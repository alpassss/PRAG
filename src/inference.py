import os
import gc
import json
import random
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete

def collect_available_adapters(load_adapter_path, filename, train_sample):
    """
    Collect all available adapter paths from the training set.
    Returns a list of (data_id, passage_id, adapter_path) tuples.
    """
    available_adapters = []
    data_dir = os.path.join(load_adapter_path, filename)
    if not os.path.exists(data_dir):
        return available_adapters
    
    for did in range(train_sample):
        data_folder = os.path.join(data_dir, f"data_{did}")
        if not os.path.exists(data_folder):
            continue
        pid = 0
        while True:
            adapter_path = os.path.join(data_folder, f"passage_{pid}")
            if os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
                available_adapters.append((did, pid, adapter_path))
                pid += 1
            else:
                break
    return available_adapters


def collect_available_passages(data_list, train_sample):
    """
    Collect all available passages from the training set.
    Returns a list of passages.
    """
    available_passages = []
    for filename, fulldata in data_list:
        for did, data in enumerate(fulldata[:train_sample]):
            if "passages" in data:
                available_passages.extend(data["passages"])
            if "augment" in data:
                for aug in data["augment"]:
                    if "passage" in aug:
                        available_passages.append(aug["passage"])
    return available_passages


def main(args):
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens = args.max_new_tokens,
    )
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)
    
    cot_name = "cot" if args.with_cot else "direct"
    load_adapter_path = os.path.join(
        ROOT_DIR, 
        "offline", 
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
    )
    output_root_dir = os.path.join(
        ROOT_DIR, 
        "output",
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
        args.inference_method, 
    )
    
    # For misinfo modes, collect available resources from training set
    available_adapters = None
    available_passages = None
    if args.inference_method == "misinfo_prag" and args.train_sample:
        print(f"### Collecting available adapters from training set (train_sample={args.train_sample}) ###")
    if args.inference_method == "misinfo_icl" and args.train_sample:
        print(f"### Collecting available passages from training set (train_sample={args.train_sample}) ###")
        available_passages = collect_available_passages(data_list, args.train_sample)
        print(f"### Found {len(available_passages)} passages ###")
    
    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(output_root_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)

        # For misinfo_prag, collect adapters for this specific file
        if args.inference_method == "misinfo_prag" and args.train_sample:
            available_adapters = collect_available_adapters(load_adapter_path, filename, args.train_sample)
            print(f"### Found {len(available_adapters)} adapters for {filename} ###")

        predict_file = os.path.join(output_dir, "predict.json")
        ret, start_with = read_complete(predict_file)

        fulldata = fulldata[start_with:] if args.sample == -1 else fulldata[start_with:args.sample]
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]

            def get_pred(model, psgs):
                text = predict(model, tokenizer, generation_config, 
                                        question, with_cot=args.with_cot, 
                                        passages=psgs)
                pred = {
                    "test_id": test_id, 
                    "question": question, 
                    "answer": answer, 
                    "text": text,
                }
                pred.update(evaluate(text, answer, args.with_cot))
                return pred

            if args.inference_method == "icl":
                ret.append(get_pred(model, psgs=passages))
            elif args.inference_method == "misinfo_plain":
                # Plain mode: direct question input without any context or LoRA
                ret.append(get_pred(model, psgs=None))
            elif args.inference_method == "misinfo_icl":
                # Misinfo ICL mode: randomly select passages from training set as context
                if available_passages and len(available_passages) > 0:
                    num_passages = min(len(passages), len(available_passages))
                    random_passages = random.sample(available_passages, num_passages)
                    pred = get_pred(model, psgs=random_passages)
                    pred["random_passages"] = random_passages
                    ret.append(pred)
                else:
                    # Fallback to using passages from data if no training passages available
                    ret.append(get_pred(model, psgs=passages))
            elif args.inference_method == "misinfo_prag":
                # Misinfo PRAG mode: randomly select LoRA weights from training set
                if available_adapters and len(available_adapters) > 0:
                    num_adapters = min(len(passages), len(available_adapters))
                    random_adapters = random.sample(available_adapters, num_adapters)
                    
                    for idx, (did, pid, adapter_path) in enumerate(random_adapters):
                        if idx == 0:
                            model = PeftModel.from_pretrained(
                                model, 
                                adapter_path,
                                adapter_name = "0", 
                                is_trainable = False
                            )
                        else:
                            model.load_adapter(adapter_path, adapter_name = str(idx))
                    
                    # merge
                    model.add_weighted_adapter(
                        adapters = [str(i) for i in range(len(random_adapters))], 
                        weights = [1] * len(random_adapters),
                        adapter_name = "merge", 
                        combination_type = "cat",
                    )
                    model.set_adapter("merge")
                    pred = get_pred(model, psgs=None)
                    pred["random_adapters"] = [(did, pid) for did, pid, _ in random_adapters]
                    ret.append(pred)
                    model.delete_adapter("merge")
                    model = model.unload()
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    # Fallback to plain mode if no adapters available
                    ret.append(get_pred(model, psgs=None))
            else:
                # Original prag/combine modes
                for pid in range(len(passages)):
                    adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}")
                    if pid == 0:
                        model = PeftModel.from_pretrained(
                            model, 
                            adapter_path,
                            adapter_name = "0", 
                            is_trainable = False
                        )
                    else:
                        model.load_adapter(adapter_path, adapter_name = str(pid)) 
                # merge
                model.add_weighted_adapter(
                    adapters = [str(i) for i in range(len(passages))], 
                    weights = [1] * len(passages),
                    adapter_name = "merge", 
                    combination_type = "cat",
                )
                model.set_adapter("merge")
                ret.append(get_pred(model, psgs=None if args.inference_method == "prag" else passages))
                model.delete_adapter("merge")
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()

        with open(predict_file, "w") as fout:
            json.dump(ret, fout, indent=4)

        ##### Evaluating #####
        metrics = ["em", "f1", "prec", "recall"]
        ret_str = ""
        for met in metrics:
            acc = sum(float(d[met]) for d in ret) / len(ret)
            acc = round(acc, 4)
            ret_str += f"{met}\t{acc}\n"
        ret_str += "\n" + json.dumps(vars(args), indent=4)
        with open(os.path.join(output_dir, "result.txt"), "w") as fout:
            fout.write(ret_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)  
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--inference_method", type=str, required=True, 
                        choices=["icl", "prag", "combine", "misinfo_prag", "misinfo_icl", "misinfo_plain"])
    parser.add_argument("--train_sample", type=int, default=None,
                        help="Number of training samples used for encoding (for misinfo modes)")
    # LoRA
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    args = parser.parse_args()
    
    # LoRA config is required except for misinfo_plain mode
    if args.inference_method != "misinfo_plain":
        assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    else:
        # Set default values for misinfo_plain if not provided
        if args.lora_rank is None:
            args.lora_rank = 2
        if args.lora_alpha is None:
            args.lora_alpha = 32
    
    # train_sample is required for misinfo_prag and misinfo_icl
    if args.inference_method in ["misinfo_prag", "misinfo_icl"]:
        assert args.train_sample is not None, "train_sample is required for misinfo modes"
    
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)
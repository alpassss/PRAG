import os
import gc
import json
import copy
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel
import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete, adapter_stats

DEBUG_ADAPTER_LOG_LIMIT = 3
DEBUG_OUTPUT_CHAR_LIMIT = 200
DEBUG_COMPARE_LIMIT_DEFAULT = 3
DEBUG_COMPARE_TEMPERATURE_DEFAULT = 0.7
DEBUG_COMPARE_TOP_P_DEFAULT = 0.8
DEBUG_COMPARE_TOP_K_DEFAULT = 20

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
    output_dir_suffix = args.inference_method
    if args.use_sampling_eval:
        output_dir_suffix = f"{output_dir_suffix}_sample"
    output_root_dir = os.path.join(
        ROOT_DIR, 
        "output",
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
        output_dir_suffix, 
    )
    compare_outputs = args.debug_compare_outputs
    compare_sampling = args.debug_compare_sampling
    compare_limit = args.debug_compare_limit
    should_compare_sampling = compare_outputs and compare_sampling
    def log_compare(label, text):
        print(f"[compare] {label}: {text[:DEBUG_OUTPUT_CHAR_LIMIT]}")
    if compare_outputs:
        print("[compare] debug output enabled; extra base inference will slow down.")
        if compare_sampling:
            print("[compare] sampling comparisons differ from greedy decoding due to stochastic token selection.")
            print("[compare] sampling debug enabled; outputs are non-deterministic.")
    if args.use_sampling_eval:
        print("[sampling] sampling evaluation enabled; metrics will be based on sampled outputs.")
    sample_generation_config = None
    needs_sampling_config = compare_sampling or args.use_sampling_eval
    if needs_sampling_config:
        sample_generation_config = copy.deepcopy(generation_config)
        sample_generation_config.update({
            "do_sample": True,
            "temperature": args.debug_compare_temperature,
            "top_p": args.debug_compare_top_p,
            "top_k": args.debug_compare_top_k,
        })
    for filename, fulldata in data_list:
        compare_count = 0
        filename = filename.split(".")[0]
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(output_root_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)

        predict_file = os.path.join(output_dir, "predict.json")
        ret, start_with = read_complete(predict_file)

        fulldata = fulldata[start_with:] if args.sample == -1 else fulldata[start_with:args.sample]
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]

            def get_pred(model, psgs, generation_override=None):
                config = generation_config if generation_override is None else generation_override
                text = predict(model, tokenizer, config, 
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
            
            def get_sample_pred(model, psgs):
                if sample_generation_config is None:
                    return get_pred(model, psgs=psgs)
                return get_pred(model, psgs, generation_override=sample_generation_config)

            psgs = None if args.inference_method == "prag" else passages
            if args.inference_method == "icl":
                pred = get_sample_pred(model, psgs=psgs) if args.use_sampling_eval else get_pred(model, psgs=psgs)
                if compare_outputs and compare_count < compare_limit:
                    log_compare("question", question)
                    pred_text = pred.get("text", "")
                    log_compare("icl", pred_text)
                    if should_compare_sampling:
                        sample_pred = get_sample_pred(model, psgs=psgs)
                        log_compare("icl(sample)", sample_pred.get("text", ""))
                    compare_count += 1
                ret.append(pred)
            else:
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
                        model.load_adapter(adapter_path, adapter_name=str(pid))
                    if pid < DEBUG_ADAPTER_LOG_LIMIT:
                        stats = adapter_stats(adapter_path)
                        print(f"[inference] load {adapter_path}: {stats}")
                # merge
                lora_model = model.base_model if hasattr(model, "base_model") else model
                lora_model.add_weighted_adapter(
                    adapters = [str(i) for i in range(len(passages))], 
                    weights = [1] * len(passages),
                    adapter_name = "merge", 
                    combination_type = "cat",
                )
                model.set_adapter("merge")
                if DEBUG_ADAPTER_LOG_LIMIT > 0:
                    try:
                        print(f"[inference] active adapters: {model.active_adapters}")
                    except (AttributeError, RuntimeError) as exc:
                        print(f"[inference] active adapters check failed: {exc}")
                pred = get_sample_pred(model, psgs=psgs) if args.use_sampling_eval else get_pred(model, psgs=psgs)
                if compare_outputs and compare_count < compare_limit:
                    with model.disable_adapter():
                        base_pred = get_pred(model, psgs=psgs)
                    # debug-only extra generation; this is intentionally expensive
                    pred_text = pred.get("text", "")
                    base_text = base_pred.get("text", "")
                    log_compare("question", question)
                    log_compare("combine(merge)", pred_text)
                    log_compare("icl(no_adapter)", base_text)
                    if should_compare_sampling:
                        sample_pred = get_sample_pred(model, psgs=psgs)
                        with model.disable_adapter():
                            sample_base_pred = get_sample_pred(model, psgs=psgs)
                        # debug-only extra generation; this is intentionally expensive
                        log_compare("combine(merge,sample)", sample_pred.get("text", ""))
                        log_compare("icl(no_adapter,sample)", sample_base_pred.get("text", ""))
                    compare_count += 1
                ret.append(pred)
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
    parser.add_argument("--inference_method", type=str, required=True, choices=["icl", "prag", "combine"])
    parser.add_argument("--debug_compare_outputs", action="store_true")
    parser.add_argument("--debug_compare_limit", type=int, default=DEBUG_COMPARE_LIMIT_DEFAULT)
    parser.add_argument("--debug_compare_sampling", action="store_true")
    parser.add_argument("--debug_compare_temperature", type=float, default=DEBUG_COMPARE_TEMPERATURE_DEFAULT)
    parser.add_argument("--debug_compare_top_p", type=float, default=DEBUG_COMPARE_TOP_P_DEFAULT)
    parser.add_argument("--debug_compare_top_k", type=int, default=DEBUG_COMPARE_TOP_K_DEFAULT)
    parser.add_argument("--use_sampling_eval", action="store_true")
    # LoRA
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)

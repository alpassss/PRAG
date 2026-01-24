import os
import gc
import time
import argparse
import torch
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig, PeftModel
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from typing import Dict, List

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, load_data, adapter_stats

DEBUG_ADAPTER_LOG_LIMIT = 1
DEBUG_TRAINABLE_LOG_LIMIT_DEFAULT = 1
DEBUG_GRAD_LOG_LIMIT_DEFAULT = 3

import numpy as np
import random

seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class TrainingData(Dataset):
    ignored_id = -100

    def __init__(self, prompt_ids, tokenizer, max_length=3000):
        self.max_length = max_length
        self.dataset = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for input_ids in prompt_ids:
            labels = input_ids.copy()
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
            input_ids += [pad_token_id] * (max_length - len(input_ids))
            labels += [self.ignored_id] * (max_length - len(labels))
            self.dataset.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })
        self.total_len = len(self.dataset)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx) -> Dict[str, list]:
        return self.dataset[idx]


class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, examples: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple(
            map(lambda x: [example[x] for example in examples], ["input_ids", "labels", "attention_mask"])
        )
        return {
            "input_ids": torch.tensor(input_ids).to(self.device),
            "labels": torch.tensor(labels).to(self.device),
            "attention_mask": torch.tensor(attention_mask).to(self.device),
        }
    

def get_train_data(aug_model, augments, tokenizer, args):
    from prompt_template import get_prompt
    prompt_ids = []
    for aug in augments:
        psg = aug["passage"]
        rew = aug[f"{aug_model}_rewrite"]
        qas = aug[f"{aug_model}_qa"]
        qpa_cnt = (len(qas) + 1) // 2
        for qid, qa in enumerate(qas):
            if qid < qpa_cnt:
                for ppp in [psg, rew]:
                    prompt_ids.append(get_prompt(tokenizer, qa["question"], 
                                                    [ppp], 
                                                    qa["answer"] if not args.with_cot else qa["full_answer"], 
                                                    with_cot=args.with_cot))
            else:
                prompt_ids.append(get_prompt(tokenizer, qa["question"], 
                                                None, 
                                                qa["answer"] if not args.with_cot else qa["full_answer"], 
                                                with_cot=args.with_cot))
    return prompt_ids


def log_trainable_parameters(model):
    """Log trainable parameter summary for LoRA debugging.

    Args:
        model: LoRA model instance to inspect.
    """
    if hasattr(model, "print_trainable_parameters"):
        print("[encode] trainable parameters:")
        model.print_trainable_parameters()
    else:
        print("[encode] trainable parameters unavailable on this model.")

def grad_norm(tensor):
    if tensor is None:
        return 0.0
    return tensor.grad.detach().float().norm().item() if tensor.requires_grad and tensor.grad is not None else 0.0

def collect_lora_norms(model, warned_adapters=None):
    norms = {"A": [], "B": [], "grad_A": [], "grad_B": []}
    warned_adapters = warned_adapters if warned_adapters is not None else set()
    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for name in module.lora_A:
                if name not in module.lora_B:
                    if name not in warned_adapters:
                        print(f"[encode] warning: lora_B missing for adapter {name}")
                        warned_adapters.add(name)
                    continue
                a = module.lora_A[name].weight
                b = module.lora_B[name].weight
                norms["A"].append(a.detach().float().norm().item())
                norms["B"].append(b.detach().float().norm().item())
                grad_a = grad_norm(a)
                grad_b = grad_norm(b)
                norms["grad_A"].append(grad_a)
                norms["grad_B"].append(grad_b)
    return norms

def log_lora_norms(label, norms):
    def safe_mean(values):
        return sum(values) / len(values) if values else 0.0
    print(
        f"[encode] {label} | "
        f"A={safe_mean(norms['A']):.6f} B={safe_mean(norms['B']):.6f} "
        f"gradA={safe_mean(norms['grad_A']):.6f} gradB={safe_mean(norms['grad_B']):.6f}"
    )

def train(question, augments, args, model, tokenizer, 
          init_adapter_path, save_path, debug_trainable=False, debug_grad=False):
    """Train LoRA adapter for a single passage and optionally log trainable params.

    Args:
        question: Question text.
        augments: Augmented passage list.
        args: Parsed CLI args.
        model: Base model to adapt.
        tokenizer: Tokenizer instance.
        init_adapter_path: Path to base LoRA weights.
        save_path: Output directory for adapter.
        debug_trainable: When True, print trainable parameter summary.
        debug_grad: When True, log LoRA A/B grad and weight norms.
    """
    prompt_ids = get_train_data(args.augment_model, augments, tokenizer, args)
    train_data = TrainingData(prompt_ids, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=False,
    )
    model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
    if debug_trainable:
        log_trainable_parameters(model)
    model.is_parallelizable = True
    model.model_parallel = True
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)
    warned_adapters = set()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            should_log_grad = debug_grad and step < args.debug_grad_limit
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            if should_log_grad:
                norms = collect_lora_norms(model, warned_adapters)
                log_lora_norms(f"grad step {step}", norms)
            optimizer.step()
            if should_log_grad:
                post_norms = collect_lora_norms(model, warned_adapters)
                log_lora_norms(f"weight step {step}", post_norms)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    model = model.unload()
    torch.cuda.empty_cache()
    gc.collect()
    return model


def main(args):
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    model, tokenizer, _generation_config = get_model(args.model_name)
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    init_adapter_path = os.path.join(
        ROOT_DIR, 
        "offline", 
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        "base_weight",
    )
    if not os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")):
        print("No LoRA base weight, creating...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['down_proj', 'gate_proj', 'up_proj'],
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0, # !!!
        )
        model = get_peft_model(model, peft_config)
        model.is_parallelizable = True
        model.model_parallel = True
        print(f'Save LoRA base weight to {init_adapter_path}')
        os.makedirs(init_adapter_path, exist_ok=True)
        model.save_pretrained(init_adapter_path)
        time.sleep(2)
        assert os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")) 

    cot_name = "cot" if args.with_cot else "direct"
    for filename, fulldata in data_list:
        filename = filename.split('.')[0] 
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(
            ROOT_DIR, 
            "offline", 
            args.model_name, 
            f"rank={args.lora_rank}_alpha={args.lora_alpha}",
            args.dataset,
            f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
            f"aug_model={args.augment_model}",
            filename,
        )
        os.makedirs(output_dir, exist_ok=True)
        fulldata = fulldata if args.sample == -1 else fulldata[:args.sample]
        for did, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            augment = data["augment"]
            for pid in range(len(augment)):
                save_path = os.path.join(output_dir, f"data_{did}", f"passage_{pid}")
                if os.path.exists(os.path.join(save_path, "adapter_model.safetensors")):
                    continue
                debug_trainable = args.debug_trainable and pid < args.debug_trainable_limit
                debug_grad = args.debug_grad and pid < args.debug_grad_limit
                model = train(data["question"], [augment[pid]], args, model, tokenizer, 
                            init_adapter_path, save_path, debug_trainable=debug_trainable, debug_grad=debug_grad)
                if DEBUG_ADAPTER_LOG_LIMIT > 0 and pid < DEBUG_ADAPTER_LOG_LIMIT:
                    print(f"[encode] saved {save_path}: {adapter_stats(save_path)}")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)
    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--debug_trainable", action="store_true")
    parser.add_argument("--debug_trainable_limit", type=int, default=DEBUG_TRAINABLE_LOG_LIMIT_DEFAULT)
    parser.add_argument("--debug_grad", action="store_true")
    parser.add_argument("--debug_grad_limit", type=int, default=DEBUG_GRAD_LOG_LIMIT_DEFAULT)
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)

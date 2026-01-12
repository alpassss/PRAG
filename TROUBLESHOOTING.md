# PRAG Troubleshooting Guide

## Understanding the Three Inference Modes

### 1. ICL Mode (In-Context Learning / Traditional RAG)
**Code**: `inference_method='icl'`

**How it works** (inference.py lines 75-76):
```python
if args.inference_method == "icl":
    ret.append(get_pred(model, psgs=passages))
```

- Uses the **base model** (no LoRA adapters)
- Passes retrieved passages directly in the prompt
- Prompt includes: "Passage 1: ...", "Passage 2: ...", etc.
- Model answers using in-context information

**LoRA Usage**: None
**Passages in Prompt**: Yes
**Expected Behavior**: Baseline performance using traditional RAG

---

### 2. PRAG Mode (Parametric RAG)
**Code**: `inference_method='prag'`

**How it works** (inference.py lines 77-101):
```python
else:  # prag or combine
    # Load LoRA adapters for each passage
    for pid in range(len(passages)):
        adapter_path = ...
        if pid == 0:
            model = PeftModel.from_pretrained(model, adapter_path, adapter_name="0", ...)
        else:
            model.load_adapter(adapter_path, adapter_name=str(pid))
    
    # Merge all passage adapters
    model.add_weighted_adapter(
        adapters=["0", "1", "2"],
        weights=[1, 1, 1],
        adapter_name="merge",
        combination_type="cat"
    )
    model.set_adapter("merge")
    
    # Generate with PRAG mode: NO passages in prompt
    ret.append(get_pred(model, psgs=None))  # psgs=None!
```

- Loads LoRA adapters trained on each passage
- Merges adapters using concatenation
- **NO passages in prompt** (psgs=None)
- Model relies entirely on parameterized knowledge in LoRA weights

**LoRA Usage**: Yes - merged adapter containing knowledge from all passages
**Passages in Prompt**: No
**Expected Behavior**: Should match or exceed ICL performance by using parameterized knowledge

---

### 3. COMBINE Mode (PRAG + ICL)
**Code**: `inference_method='combine'`

**How it works** (inference.py lines 77-101):
```python
else:  # prag or combine
    # ... same adapter loading and merging as PRAG ...
    
    # Generate with COMBINE mode: WITH passages in prompt
    ret.append(get_pred(model, psgs=passages))  # psgs=passages!
```

- Same as PRAG mode but **includes passages in prompt**
- Uses both parameterized knowledge (LoRA) and in-context information (passages)
- Best of both worlds

**LoRA Usage**: Yes - merged adapter
**Passages in Prompt**: Yes
**Expected Behavior**: Should perform best, combining parametric and in-context knowledge

---

## Common Issues and Solutions

### Issue 1: COMBINE Mode = ICL Mode (LoRA Has No Effect)

**Symptoms**:
- COMBINE and ICL have identical F1 scores
- PRAG mode performs worse than ICL
- Changing num_train_epochs has no effect

**Possible Causes**:

1. **LoRA weights not being loaded**
   - Check if adapter files exist: `offline/{model}/rank={r}_alpha={a}/{dataset}/lr={lr}_epoch={e}_direct/aug_model={model}/total/data_0/passage_0/adapter_model.safetensors`
   - Run the diagnostic script: `python src/diagnostic_test.py --model_name=... --dataset=... --num_train_epochs=... --lora_rank=... --lora_alpha=...`

2. **LoRA weights contain no useful information**
   - Training data might be insufficient (only ~5 examples per adapter)
   - Training might not be working (check for errors during encode)
   - Learning rate might be too small or too large

3. **Adapter merging not working**
   - The `combination_type="cat"` might have issues with PEFT version
   - Try changing to `combination_type="linear"` in inference.py line 94

4. **Model not using active adapter**
   - After `model.set_adapter("merge")`, the adapter should be active
   - Check PEFT version matches requirements.txt: `peft==0.13.2`

---

### Issue 2: PRAG Mode Much Worse Than ICL

**Symptoms**:
- PRAG F1 << ICL F1 (e.g., 0.0287 vs 0.0999)

**Explanation**:
- PRAG mode has **no passages in the prompt**
- Relies entirely on LoRA-parameterized knowledge
- If LoRA training failed or adapters aren't loading, model has no knowledge to answer

**Solutions**:
1. Verify LoRA weights were trained properly
2. Check adapter loading (use diagnostic script)
3. Try COMBINE mode instead - it should at least match ICL if LoRA fails

---

### Issue 3: Path Mismatches

**Important**: The paths for encoding and inference must match exactly!

**Encoding saves to**:
```
offline/{model}/rank={r}_alpha={a}/{dataset}/lr={lr}_epoch={e}_{cot}/aug_model={aug_model}/{filename}/data_{did}/passage_{pid}/
```

**Inference loads from**:
```
offline/{model}/rank={r}_alpha={a}/{dataset}/lr={lr}_epoch={e}_{cot}/aug_model={aug_model}/{filename}/data_{test_id}/passage_{pid}/
```

**Critical parameters that must match**:
- `model_name`
- `lora_rank` and `lora_alpha`
- `dataset`
- `learning_rate` and `num_train_epochs`
- `with_cot` (affects `{cot}` = "cot" or "direct")
- `augment_model` (if None, defaults to model_name)
- `data_type` (affects `{filename}`)

---

## Data Structure Requirements

### For Encoding (encode.py)
Data must have `augment` field:
```json
{
  "question": "string",
  "answer": "string or list",
  "passages": ["passage1", "passage2", "passage3"],
  "augment": [
    {
      "pid": 0,
      "passage": "passage1",
      "{model}_rewrite": "rewritten passage",
      "{model}_qa": [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
        {"question": "q3", "answer": "a3"}
      ]
    },
    ...
  ]
}
```

### For Inference (inference.py)
Data must have `passages` field (and `augment` is not used):
```json
{
  "question": "string",
  "answer": "string or list",
  "passages": ["passage1", "passage2", "passage3"]
}
```

**Note**: The same data file can be used for both if it has all fields.

---

## Running the Diagnostic Script

```bash
python src/diagnostic_test.py \
    --model_name=qwen2.5-1.5b-instruct \
    --dataset=popqa \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32
```

This will check:
1. ✓ Data loading works correctly
2. ✓ LoRA weight files exist
3. ✓ Adapters can be loaded and merged

---

## Expected Results (from paper)

For PopQA with qwen2.5-1.5b-instruct:
- ICL: ~0.10 F1
- PRAG: ~0.19 F1 (better than ICL!)
- COMBINE: ~0.20+ F1 (best)

If your results don't match:
1. Run diagnostic script to find issues
2. Check all parameters match between encode and inference
3. Verify LoRA weights were created during encoding
4. Check for errors in training logs

---

## Key Code Locations

### Data Loading
- `src/utils.py` line 78-119: `load_data()` function

### Encoding (Training LoRA)
- `src/encode.py` line 94-121: `train()` function
- `src/encode.py` line 71-91: `get_train_data()` - generates training examples

### Inference
- `src/inference.py` line 75-101: Three modes (ICL, PRAG, COMBINE)
- `src/utils.py` line 221-237: `predict()` function

### Prompt Construction
- `src/prompt_template.py` line 59-82: `get_prompt()` function
- Line 62-64: Passages formatted as "Passage 1: ...", etc.
- If `passages=None`, contexts becomes empty string

---

## Debugging Tips

1. **Check if LoRA files exist**:
   ```bash
   find offline/ -name "adapter_model.safetensors" | head
   ```

2. **Check file sizes** (should be non-zero):
   ```bash
   find offline/ -name "adapter_model.safetensors" -exec ls -lh {} \;
   ```

3. **Count how many adapters were created**:
   ```bash
   find offline/ -name "adapter_model.safetensors" | wc -l
   ```
   For 300 samples with 3 passages each: should be 900 files

4. **Check data structure**:
   ```python
   import json
   data = json.load(open('data_aug/popqa/qwen2.5-1.5b-instruct/total.json'))
   print('augment' in data[0])  # Should be True
   print(len(data[0]['augment']))  # Should equal len(data[0]['passages'])
   ```

5. **Test single sample inference**:
   Add `--sample=1` to inference command to test just one sample

---

## Contact and Support

If issues persist after following this guide:
1. Run the diagnostic script and share output
2. Check GitHub issues for similar problems
3. Verify your data matches the expected structure
4. Ensure all paths and parameters match exactly between encode and inference

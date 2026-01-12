# Potential Fix for PRAG Inference Issues

## Issue Summary

Based on analysis of the code and known PEFT library issues, the problem where COMBINE mode performs identically to ICL mode (indicating LoRA adapters have no effect) could be caused by:

1. **PEFT adapter merging bug**: There's a known issue in PEFT where `add_weighted_adapter` may not correctly apply weights to both LoRA matrices
2. **"cat" combination type issues**: Concatenation-based merging might not work correctly in PEFT 0.13.2
3. **Adapter activation**: The merged adapter might not be properly activated during generation

## Proposed Fixes

### Fix 1: Change Combination Type from "cat" to "linear"

**File**: `src/inference.py`  
**Line**: 94

**Current code**:
```python
model.add_weighted_adapter(
    adapters = [str(i) for i in range(len(passages))], 
    weights = [1] * len(passages),
    adapter_name = "merge", 
    combination_type = "cat",  # <-- Change this
)
```

**Proposed fix**:
```python
model.add_weighted_adapter(
    adapters = [str(i) for i in range(len(passages))], 
    weights = [1.0 / len(passages)] * len(passages),  # Normalized weights
    adapter_name = "merge", 
    combination_type = "linear",  # <-- Use linear instead of cat
)
```

**Explanation**:
- "linear" uses weighted averaging instead of concatenation
- For equal contribution from all adapters, use normalized weights (1/N each)
- This avoids potential bugs with "cat" combination
- All adapters have the same rank (2), so "linear" is compatible

---

### Fix 2: Explicitly Set Adapter Before Generation

**File**: `src/inference.py`  
**Lines**: 96-97

**Current code**:
```python
model.set_adapter("merge")
ret.append(get_pred(model, psgs=None if args.inference_method == "prag" else passages))
```

**Proposed fix**:
```python
model.set_adapter("merge")
# Verify adapter is set
assert model.active_adapter == "merge", f"Expected 'merge' but got {model.active_adapter}"
ret.append(get_pred(model, psgs=None if args.inference_method == "prag" else passages))
```

**Explanation**:
- Adds assertion to verify the adapter is actually active
- Will raise an error if adapter isn't set correctly
- Helps diagnose if the issue is with adapter activation

---

### Fix 3: Upgrade PEFT Version (if possible)

**File**: `requirements.txt`  
**Line**: 4

**Current**:
```
peft==0.13.2
```

**Proposed**:
```
peft>=0.14.0
```

**Explanation**:
- PEFT 0.13.2 has known bugs with adapter merging
- Later versions (0.14.0+) have fixes for adapter merging issues
- This might resolve the problem entirely

**Note**: Test thoroughly after upgrading, as API might have changed slightly.

---

### Fix 4: Alternative Approach - Use Single Adapter Instead of Merging

Instead of merging all passage adapters, you could try using them one at a time or in sequence.

**File**: `src/inference.py`  
**Replace lines 78-101 with**:

```python
else:
    # Alternative: Average predictions from each adapter separately
    all_preds = []
    
    for pid in range(len(passages)):
        adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}")
        
        # Load single adapter
        model_with_adapter = PeftModel.from_pretrained(
            model, 
            adapter_path,
            adapter_name = "temp", 
            is_trainable = False
        )
        model_with_adapter.set_adapter("temp")
        
        # Get prediction with this adapter
        pred = get_pred(model_with_adapter, psgs=None if args.inference_method == "prag" else [passages[pid]])
        all_preds.append(pred)
        
        # Clean up
        model_with_adapter = model_with_adapter.unload()
        torch.cuda.empty_cache()
        gc.collect()
    
    # Use the first prediction (or implement voting/ensembling)
    ret.append(all_preds[0] if args.inference_method == "prag" else get_pred(model, psgs=passages))
```

**Note**: This is a workaround that avoids adapter merging entirely. It's not the intended PRAG approach but could help diagnose if merging is the issue.

---

## Testing the Fixes

After applying any fix, test with:

```bash
# Run diagnostic first
python src/diagnostic_test.py \
    --model_name=qwen2.5-1.5b-instruct \
    --dataset=popqa \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32

# Then run inference
python src/inference.py \
    --model_name=qwen2.5-1.5b-instruct \
    --dataset=popqa \
    --sample=10 \  # Test with small sample first
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=20 \
    --inference_method=combine  # Should be different from ICL now
```

## Expected Results After Fix

If the fix works:
- **ICL**: ~0.10 F1 (unchanged)
- **PRAG**: ~0.19 F1 (should improve significantly)
- **COMBINE**: ~0.20+ F1 (should be better than ICL)

## Additional Debugging

If none of the fixes work, check:

1. **LoRA weights are non-zero**:
   ```python
   import torch
   weights = torch.load("offline/.../adapter_model.safetensors")
   print({k: v.abs().mean().item() for k, v in weights.items()})
   # Should show non-zero values
   ```

2. **Training loss decreases during encoding**:
   - Add print statement in encode.py train() function to log loss
   - Loss should decrease over epochs

3. **Model is actually using adapters**:
   - Add debug prints in inference.py to check active adapter
   - Verify adapter parameters are being updated during forward pass

4. **Data is correct**:
   - Verify `augment` field exists in all samples
   - Check that augment data contains valid Q&A pairs
   - Ensure passages match between `passages` and `augment[*]['passage']`

## Recommended Approach

1. **Start with Fix 1** (change "cat" to "linear") - easiest and most likely to work
2. **Add Fix 2** (assertion) for debugging
3. **If still not working**, try Fix 4 (single adapter at a time) to isolate the issue
4. **Last resort**: Upgrade PEFT (Fix 3), but test carefully

## Need More Help?

If issues persist after trying these fixes:
1. Run diagnostic_test.py and share the output
2. Check if LoRA weight files exist and have reasonable sizes
3. Try with a very small sample (--sample=1) to simplify debugging
4. Check GitHub issues for PEFT and this repository for similar problems

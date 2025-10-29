# Memory Optimization for Training

## Problem
Training encounters CUDA OOM (Out of Memory) errors after several epochs, even though early epochs run fine. This indicates **memory fragmentation** and potential **memory leaks**.

## Root Causes

1. **PyTorch Memory Fragmentation**: PyTorch's CUDA allocator accumulates fragmented memory over time
2. **Batch References**: Batch objects not being explicitly released
3. **Checkpoint Saving**: Temporary memory spike when saving model checkpoints
4. **Cache Accumulation**: CUDA cache grows over epochs without proper cleanup

## Fixes Applied

### 1. Code-Level Fixes (in `scripts/04_train_model.py`)

#### Training Loop (`train_epoch`)
- ✅ Explicitly delete `batch` variable after use
- ✅ More frequent cache clearing (every 5 batches instead of 10)
- ✅ Periodic synchronization (every 20 batches)

#### Validation Loop (`evaluate`)
- ✅ Explicitly delete `batch` variable after use
- ✅ More frequent cache clearing (every 5 batches)

#### Epoch-Level Cleanup
- ✅ Synchronize GPU before clearing cache
- ✅ Reset peak memory stats for better monitoring
- ✅ Clear cache after saving checkpoints
- ✅ Clear cache after saving best model

### 2. Environment Variables

Set these before running training:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

**What they do:**
- `expandable_segments:True`: Allows PyTorch to expand memory segments, reducing fragmentation
- `max_split_size_mb:128`: Limits the maximum size of memory splits to 128MB

## How to Use

### Option 1: Use the Optimized Script (Recommended)

```bash
cd /personal/RNA-3E-FFI
chmod +x scripts/train_with_memory_opt.sh
bash scripts/train_with_memory_opt.sh
```

Or with custom batch size:
```bash
bash scripts/train_with_memory_opt.sh 2  # Batch size = 2
```

### Option 2: Manual Setup

```bash
# Set environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Run training
python scripts/04_train_model.py \
    --batch_size 2 \
    --use_multi_hop \
    --use_nonbonded \
    --use_gate \
    --use_layer_norm
```

## Additional Tips if Still Getting OOM

### 1. Reduce Batch Size
```bash
# Try batch_size=1 if batch_size=2 still causes OOM
python scripts/04_train_model.py --batch_size 1 ...
```

### 2. Reduce Model Size
- Decrease `--num_layers` from 4 to 3
- Decrease `--output_dim` from 1536 to 1024
- Use simpler `--hidden_irreps` like "16x0e + 8x1o + 4x2e"

### 3. Disable Some Features
```bash
# Train without layer norm (saves memory)
python scripts/04_train_model.py \
    --use_multi_hop \
    --use_nonbonded \
    --use_gate
    # Note: removed --use_layer_norm
```

### 4. Use Gradient Accumulation

Modify training code to accumulate gradients over multiple mini-batches:

```python
# In train_epoch(), accumulate gradients every N steps
accumulation_steps = 4
for batch_idx, batch in enumerate(loader):
    loss = loss / accumulation_steps
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Monitoring Memory Usage

Add this to check memory at each epoch:

```python
if device.type == 'cuda':
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

## Summary of Changes

| File | Changes |
|------|---------|
| `scripts/04_train_model.py` | More aggressive memory cleanup in train/eval loops and after checkpoint saving |
| `scripts/train_with_memory_opt.sh` | New script with optimized environment variables |
| `MEMORY_OPTIMIZATION.md` | This documentation |

## Why It Works

1. **Frequent cache clearing** prevents fragmented memory from accumulating
2. **Explicit variable deletion** ensures Python releases references immediately
3. **Synchronization** forces GPU to complete operations before cleanup
4. **Environment variables** enable PyTorch's built-in anti-fragmentation mechanisms
5. **Checkpoint cleanup** prevents memory spikes from persisting

The key insight is that even though early epochs fit in memory, **fragmentation accumulates** over time. By being more aggressive about cleanup, we prevent this accumulation.

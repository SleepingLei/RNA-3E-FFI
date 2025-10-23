# Training Script Usage Guide

## Overview
The `04_train_model.py` script now supports both training from scratch and resuming from a checkpoint.

## Usage

### 1. Train from scratch (default)
```bash
python scripts/04_train_model.py \
    --num_epochs 150 \
    --batch_size 4 \
    --lr 1e-4
```

### 2. Resume from a checkpoint
```bash
python scripts/04_train_model.py \
    --resume \
    --checkpoint models/checkpoints/best_model.pt \
    --num_epochs 200
```

### 3. Resume from a specific epoch checkpoint
```bash
python scripts/04_train_model.py \
    --resume \
    --checkpoint models/checkpoints/checkpoint_epoch_50.pt \
    --num_epochs 200
```

## Key Parameters

### Resume Training
- `--resume`: Enable resuming from checkpoint (default: False)
- `--checkpoint`: Path to checkpoint file (default: `models/checkpoints/best_model.pt`)

### Training Parameters
- `--num_epochs`: Total number of epochs to train (default: 150)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-5)
- `--patience`: Early stopping patience (default: 10)

### Model Parameters
- `--input_dim`: Input feature dimension (default: 11)
- `--hidden_irreps`: Hidden layer irreps (default: "32x0e + 16x1o + 8x2e")
- `--output_dim`: Output embedding dimension (default: 1536)
- `--num_layers`: Number of message passing layers (default: 4)

## Checkpoint Format

The checkpoint files (`.pt`) contain:
- `epoch`: Last completed epoch number
- `model_state_dict`: Model parameters
- `optimizer_state_dict`: Optimizer state
- `train_loss`: Training loss at this epoch
- `val_loss`: Validation loss at this epoch

## Notes

1. When resuming training:
   - The script loads the model and optimizer states
   - Training continues from epoch N+1 (where N is the checkpoint epoch)
   - Training history is preserved if `training_history.json` exists
   - The best validation loss is tracked across resumed sessions

2. The script automatically saves:
   - `best_model.pt`: Best model based on validation loss
   - `checkpoint_epoch_N.pt`: Checkpoints every N epochs (default: 5)
   - `training_history.json`: Training and validation loss history
   - `config.json`: Training configuration

3. If the checkpoint file doesn't exist when `--resume` is specified, the script will start training from scratch with a warning.

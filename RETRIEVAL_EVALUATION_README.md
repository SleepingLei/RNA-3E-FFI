# Retrieval Evaluation Guide

## Overview

This guide explains how to use the retrieval evaluation scripts to assess model performance on the task of finding the correct ligand from a bait library.

## Key Metrics

### 1. Normalized Rank (0-1, lower is better)
- **Formula**: `(rank - 1) / (total_baits - 1)`
- **Range**: [0, 1]
  - 0 = Perfect (rank = 1, correct ligand ranked first)
  - 1 = Worst (rank = total_baits, correct ligand ranked last)
- **Interpretation**: The normalized position of the true ligand in the ranked list

### 2. Mean Reciprocal Rank (MRR, higher is better)
- **Formula**: `1 / rank`, averaged over all samples
- **Range**: (0, 1]
- **Interpretation**: Emphasizes getting the correct answer in top positions

### 3. Recall@K (higher is better)
- **Definition**: Percentage of samples where true ligand appears in top-K predictions
- **Common values**: K = 1, 5, 10, 20, 50
- **Interpretation**:
  - Recall@1 = Top-1 Accuracy
  - Recall@5 = Success rate if we consider top-5 predictions

### 4. Top-1 Accuracy (higher is better)
- **Definition**: Percentage of samples where rank = 1
- **Interpretation**: How often the model gets it exactly right

---

## Usage

### Step 1: Run Retrieval Evaluation

```bash
python scripts/evaluate_retrieval.py \
  --checkpoint models/checkpoints/best_model.pt \
  --bait_library data/processed/ligand_embeddings_dedup.h5 \
  --splits data/splits/splits.json \
  --graph_dir data/processed/graphs \
  --output results/retrieval_results.json \
  --metric cosine \
  --recall_k 1 5 10 20 50 100
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint
- `--bait_library`: HDF5 file with bait ligand embeddings
- `--splits`: JSON file with train/val/test splits
- `--graph_dir`: Directory containing RNA pocket graphs
- `--output`: Where to save results JSON
- `--metric`: Similarity metric (cosine or euclidean)
- `--recall_k`: K values for Recall@K metric

**Expected Output:**
```
Retrieval Evaluation Results
======================================================================

Total samples:           200
Successful predictions:  195
Failed predictions:      5
Total baits in library:  150
Similarity metric:       cosine

📊 Normalized Rank (0-1, lower is better):
  Mean:   0.2345
  Median: 0.1876
  Std:    0.1543
  Range:  [0.0000, 0.9234]

🎯 Mean Reciprocal Rank (MRR, higher is better):
  MRR: 0.6234

🏆 Top-1 Accuracy:
  45.64%

Rank Statistics (absolute):
  Mean rank:   15.23
  Median rank: 8.0
  Min rank:    1
  Max rank:    142
  Std rank:    18.45

Recall@K:
  recall@1:    45.64% (89/195)
  recall@5:    72.31% (141/195)
  recall@10:   85.13% (166/195)
  recall@20:   92.82% (181/195)
  recall@50:   97.95% (191/195)
```

### Step 2: Analyze Results

```bash
python scripts/analyze_retrieval_results.py \
  --results results/retrieval_results.json \
  --output_dir results/retrieval_analysis
```

**Generated outputs:**
- `rank_distribution.png`: Histogram of absolute and normalized ranks
- `recall_curve.png`: Recall@K vs K curve
- `cumulative_distribution.png`: Cumulative distribution of normalized ranks
- `summary_report.txt`: Detailed text report

**Example analysis output:**
```
Per-Ligand Performance Analysis
======================================================================

Found 45 unique ligands

Ligand          Count    Mean Rank    Norm Rank    MRR        Top-1 Acc
----------------------------------------------------------------------
ARG             15       3.20         0.0147       0.4531     60.0%
SAM             12       5.67         0.0381       0.2987     50.0%
ATP             10       8.90         0.0597       0.1876     30.0%
...

Top 20 Worst-Performing Cases
======================================================================

Complex ID                     True Ligand  Rank     Norm Rank    Top-1 Pred
----------------------------------------------------------------------
2kx8_P12_model2               P12          142      0.9234       ARG
1uui_5MC_model0               5MC          128      0.8523       SAM
...
```

---

## Interpretation Guide

### What does a good model look like?

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| Mean Normalized Rank | < 0.1 | 0.1-0.3 | 0.3-0.5 | > 0.5 |
| MRR | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |
| Top-1 Accuracy | > 60% | 40-60% | 20-40% | < 20% |
| Recall@10 | > 90% | 70-90% | 50-70% | < 50% |

### Example Scenarios

#### Scenario A: Excellent Performance
```
Mean Normalized Rank: 0.08
MRR: 0.78
Top-1 Accuracy: 65%
Recall@10: 92%
```
**Interpretation**: Model reliably ranks true ligand in top positions. 65% perfect predictions, 92% get it in top-10.

#### Scenario B: Good but Improvable
```
Mean Normalized Rank: 0.23
MRR: 0.52
Top-1 Accuracy: 38%
Recall@10: 75%
```
**Interpretation**: Model shows good discrimination but not always perfect. Consider using top-5 or top-10 for virtual screening.

#### Scenario C: Poor Performance
```
Mean Normalized Rank: 0.56
MRR: 0.21
Top-1 Accuracy: 12%
Recall@10: 35%
```
**Interpretation**: Model struggles to distinguish true ligand from baits. May need:
- More training data
- Better feature engineering
- Different loss function
- Architecture improvements

---

## Troubleshooting

### Issue 1: Low Recall@1 but High Recall@10
**Symptom**: Top-1 Accuracy 20%, but Recall@10 is 80%

**Diagnosis**: Model captures ligand binding patterns but lacks precision

**Solutions**:
- Use contrastive loss (InfoNCE) instead of MSE
- Increase model capacity
- Add hard negative mining

### Issue 2: Some ligands perform much worse
**Symptom**: Per-ligand analysis shows high variance

**Diagnosis**: Dataset imbalance or certain ligands are harder to predict

**Solutions**:
- Check if poorly-performing ligands have fewer training examples
- Use class-balanced sampling
- Add ligand-specific features

### Issue 3: Model predicts same ligand for many pockets
**Symptom**: Top-1 predictions dominated by 1-2 ligands

**Diagnosis**: Model collapsed to predicting common ligands

**Solutions**:
- Check bait library composition (should be balanced)
- Use temperature scaling in similarity computation
- Add diversity penalty in training

---

## Advanced Usage

### Custom Bait Library

Create your own bait library for specific screening:

```python
import h5py
import numpy as np

# Load full ligand embeddings
with h5py.File('data/processed/ligand_embeddings_dedup.h5', 'r') as f_in:
    # Select specific ligands for bait library
    bait_ligands = ['ARG', 'SAM', 'ATP', 'GTP', 'NAD', ...]

    # Create custom bait library
    with h5py.File('data/custom_bait_library.h5', 'w') as f_out:
        for ligand_id in bait_ligands:
            if ligand_id in f_in:
                f_out.create_dataset(ligand_id, data=f_in[ligand_id][:])

# Use in evaluation
python scripts/evaluate_retrieval.py \
  --bait_library data/custom_bait_library.h5 \
  ...
```

### Batch Comparison

Compare multiple models:

```bash
# Model 1
python scripts/evaluate_retrieval.py \
  --checkpoint models/model_v1/best_model.pt \
  --output results/retrieval_v1.json

# Model 2
python scripts/evaluate_retrieval.py \
  --checkpoint models/model_v2/best_model.pt \
  --output results/retrieval_v2.json

# Compare
python scripts/compare_retrieval_results.py \
  --results results/retrieval_v1.json results/retrieval_v2.json \
  --labels "Baseline" "Improved" \
  --output results/comparison.png
```

---

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@software{rna_3e_ffi_retrieval,
  title={RNA-3E-FFI Retrieval Evaluation Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/RNA-3E-FFI}
}
```

---

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

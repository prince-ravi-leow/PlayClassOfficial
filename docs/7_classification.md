# Classification

LOCO (Leave-One-Cage-Out) cross-validation over the built dataset. Each fold
trains on 4 cages and evaluates on the held-out cage; a disjoint test set is
scored at the end. Best result: **0.777 macro-averaged F1** (TemporalCNNv2 on
morphokinematic features + V-JEPA 2.1 embeddings).

## Commands

- **Command:** `pixi run train [options]`
- **Requires:** dataset pipeline outputs (steps 4–5)
- **Inputs:** `data/dataset/` (tracks, features, embeddings)
- **Outputs:** `data/results/eval_classification/{timestamp}_{model}/`

```sh
# Features only (MLP baseline)
pixi run train --model mlp --input features --exclude social

# Best model: TemporalCNNv2 + V-JEPA 2.1 temporal embeddings
pixi run train \
    --model temporal_cnn2 \
    --input features+embeddings_vjepa21_vitl_temporal \
    --exclude social --dropout 0.0 --n-segments 24

# Dry run (first fold only, 1 batch, no checkpoints)
pixi run train --model mlp --input features --dry-run
```

## Models

| `--model`       | Backbone        | Temporal                     |
| --------------- | --------------- | ---------------------------- |
| `linear`        | `SimpleLinear`  | No                           |
| `mlp`           | `SimpleMLP`     | No                           |
| `temporal_mlp`  | `TemporalMLP`   | Yes (gated attention)        |
| `temporal_cnn2` | `TemporalCNNv2` | Yes (GELU bottleneck + conv) |

Non-temporal models mean-pool embeddings and use windowed feature stats.
Temporal models segment-pool embeddings and use binned feature tensors. **Hybrid
mode**: passing `features+embeddings` to a temporal model concatenates flat
windowed features before the classification head.

## Input specification

`--input` accepts any combination of `features`, `embeddings`, and named
embedding files:

| `--input`                                   | Loaded files                             |
| ------------------------------------------- | ---------------------------------------- |
| `features`                                  | `features_windowed.parquet`              |
| `embeddings`                                | `embeddings.pt`                          |
| `embeddings_vjepa21_vitl_temporal`          | `embeddings_vjepa21_vitl_temporal.pt`    |
| `features+embeddings_vjepa21_vitl_temporal` | both; embeddings temporal, features flat |

## Output layout

```
data/results/eval_classification/{timestamp}_{model}/
├── cfg.json                          # full CLI args for reproducibility
├── fold_0_C1/
│   ├── checkpoints/                  # best val_loss checkpoint
│   └── lightning_logs/version_0/metrics.csv
├── fold_1_C2/ ...
├── loco_summary.csv                  # scalar metrics per fold + MEAN/STD/POOLED
├── loco_train_confusion_matrix.{npy,txt}
├── loco_val_confusion_matrix.{npy,txt}
├── loco_test_confusion_matrix.{npy,txt}   # summed across folds
├── loco_test_confusion_matrixs.npy        # per-fold CMs, shape (n_folds, C, C)
└── loco_test_recall.txt                   # per-fold per-class recall
```

## Key options

| Flag                | Default         | Description                          |
| ------------------- | --------------- | ------------------------------------ |
| `--n-segments`      | 12              | Temporal bins (temporal models only) |
| `--dropout`         | model default   | Override model dropout               |
| `--label-smoothing` | 0.1             | Cross-entropy label smoothing        |
| `--class-weights`   | `inv-sqrt`      | `inv-sqrt`, `inv-freq`, or `none`    |
| `--exclude social`  | —               | Drop social behaviour class          |
| `--dataset-dir`     | `data/dataset/` | Override dataset path                |

## Post-hoc evaluation

Generate per-sample predictions and per-fold confusion matrices from existing
checkpoints, or assemble the segment sweep and ablation tables:

```sh
# Evaluate a single run (produces CMs, recall, predictions.csv)
pixi run eval_classification evaluate <run_dir>

# Evaluate all runs (skips those with existing outputs)
pixi run eval_classification evaluate --all

# Assemble Supp Tables 3 and 4 (no GPU needed)
pixi run eval_classification tables
```

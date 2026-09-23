# Analysis

Analysis notebooks live in `notebook/`. Open and run them in Jupyter — all paths
are relative to `notebook/`, all figure outputs go to `img/` (figures and cell
outputs are only saved when `save_output = True` (first cell); otherwise they
are only displayed inline).

## Notebooks

| Notebook                    | Data inputs                                                                                                                                                                                                           | Outputs (`img/`)                                                                                                                                                                                                    |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fig1_dataset.ipynb`        | `data/dataset/tracks.parquet`, `data/dataset/labels.parquet`, `data/videos/day_28/`, `facebook/sam3` (Hugging Face)                                                                                                   | None (panels display inline)                                                                                                                                                                                        |
| `fig2_clustering.ipynb`     | `data/dataset/labels.parquet`, `data/dataset/features_windowed.parquet`, `data/results/clustering/Z_dropna.npy` (optional), `data/results/clustering/grid_dropna.csv`                                                 | `fig2_cluster_heatmap.pdf/png`, `fig2_tsne_{ethogram,subbehaviours,cluster}.pdf/png`, `fig2_radar.pdf/png`, `figS_loadings.pdf/png`, `figS_cluster_vs_k.pdf/png`, `data/results/clustering/cluster_assignments.csv` |
| `fig3_classification.ipynb` | `data/dataset/labels.parquet`, `data/dataset/features_windowed.parquet`, `data/dataset/embeddings_vjepa21_vitl_temporal.pt`, six runs in `data/results/eval_classification/` (see [below](#fig3-classification-runs)) | `fig3_metrics.pdf/png`, `fig3_metrics_uar.pdf/png`, `fig3_heatmap.pdf/png`, `fig3_heatmap_v2.pdf/png`, `fig3_heatmap_v3.pdf/png`, `fig3_shap_feature_groups.pdf`                                                    |
| `figS_tracker_eval.ipynb`   | `data/results/eval_tracking/results/metrics_aggregate.csv`, `data/results/eval_tracking/results/metrics_per_video.csv`                                                                                                | `figS_tracker_eval_hota.pdf/png`, `figS_tracker_eval_metrics_agg.pdf/png`                                                                                                                                           |

### fig3 classification runs

`fig3_classification.ipynb` hard-codes six run directories under
`data/results/eval_classification/`:

| Key            | Model              | Run directory                   |
| -------------- | ------------------ | ------------------------------- |
| `features_mlp` | Features (MLP)     | `20260916_123403_mlp`           |
| `dinov3`       | DINOv3             | `20260916_093850_temporal_cnn2` |
| `vjepa2`       | V-JEPA 2           | `20260916_101306_temporal_cnn2` |
| `vjepa21`      | V-JEPA 2.1         | `20260916_102516_temporal_cnn2` |
| `videoprism`   | VideoPrism         | `20260916_095809_temporal_cnn2` |
| `best`         | Feat. + V-JEPA 2.1 | `20260916_110657_temporal_cnn2` |

Each run needs `loco_summary.csv`, `loco_test_confusion_matrices.npy` and
`loco_test_recall.txt`. The `best` run also needs `predictions.csv` (panel b
heatmaps) and `fold_*/checkpoints/*.ckpt` (panel c SHAP). All but
`loco_summary.csv` come from `eval_classification evaluate` (see
[below](#classification-evaluation-and-tables)).

## Clustering grid

Runs the k × solver silhouette grid for the dropna variant and writes
`Z_dropna.npy` and `grid_dropna.csv` to `data/results/clustering/`. Run this
before the notebook:

```sh
pixi run clustering_grid
```

## Classification evaluation and tables

Assemble the segment sweep (Supp Table 3) and ablation (Supp Table 4) CSVs from
existing `loco_summary.csv` files (no GPU needed):

```sh
pixi run eval_classification tables
```

To backfill per-fold confusion matrices, recall, and per-sample predictions from
checkpoints (requires GPU):

```sh
pixi run eval_classification evaluate --all
```

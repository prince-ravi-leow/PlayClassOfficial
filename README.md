# PlayClass

A pipeline for play behaviour recognition in videos of poultry with tracking,
postprocessing, feature extraction and classification, along with downstream
analysis.

## Installation

### Prerequisites

All code was developed and tested on Ubuntu 24.04 (linux-64) with CUDA 12.6.

### Dependencies

- Install [pixi](https://pixi.sh), a conda-based package manager:
```sh
curl -fsSL https://pixi.sh/install.sh | bash
```

- Run the following commands to setup the main dependencies for this project.

```sh
git submodule update --init --recursive
pixi install
```

- To install SAM 3, you will need apply for approval at:
  <https://huggingface.co/facebook/sam3>
- Pixi environments (every environment builds on `default`):
  - `default`: PyTorch (CUDA 12.6), `transformers`, Lightning, scikit-learn,
    SHAP. Used for building the dataset, DINOv3/V-JEPA 2 embeddings,
    classification and analysis.
  - `tracker`: adds Ultralytics (YOLO), OmegaConf and motmetrics. Used for
    tracking (SAM 3, YOLO) and tracker evaluation.
  - `gs2`: Grounded-SAM-2, used for tracker benchmarking. Install it once with
    `pixi run -e gs2 setup_gs2`.
  - `videoprism`: adds JAX + VideoPrism for VideoPrism embeddings. Solved
    separately so JAX doesn't clash with PyTorch.

All commands use `pixi` task definitions from `pixi.toml`.
Scripts in `scripts/` (one-time setup) and `pipeline/` are the executables; `src/` holds reusable library modules.

```
data/
  labels/          Registration protocol Excel files (behaviour labels + bird info)
  tracking/        Symlinks to tracking run output dirs (gitignored)
  postprocessing/  Version-controlled per-video postprocessing JSONs + parquets (day_28/, day_29/)
  tracker_eval/    Version-controlled tracker benchmark artefacts (video manifest, keyframes, ablation configs, scored results)
```

| Stage                                          | Docs                                                 | Environment             |
| ---------------------------------------------- | ---------------------------------------------------- | ----------------------- |
| 1. Data (TO-DO: Zenodo deposit in preparation) | [data/README.md](data/README.md)                     | —                       |
| 2. Tracking                                    | [docs/2_tracking.md](docs/2_tracking.md)             | `tracker`, `gs2`        |
| 3. Tracker evaluation _(optional)_             | [docs/3_tracker_eval.md](docs/3_tracker_eval.md)     | `tracker`, `gs2`        |
| 4. Postprocessing                              | [docs/4_postprocessing.md](docs/4_postprocessing.md) | `default`               |
| 5. Build dataset                               | [docs/5_dataset.md](docs/5_dataset.md)               | `default`               |
| 6. Embeddings                                  | [docs/6_embeddings.md](docs/6_embeddings.md)         | `default`, `videoprism` |
| 7. Classification                              | [docs/7_classification.md](docs/7_classification.md) | `default`               |

### Analysis

| Stage                                                                 | Docs                                     | Environment |
| --------------------------------------------------------------------- | ---------------------------------------- | ----------- |
| 8. Analysis (clustering, classification figures, feature attribution) | [docs/8_analysis.md](docs/8_analysis.md) | `default`   |

### Tests

```sh
pixi run test_features                           # Feature extraction unit tests
pixi run test_labels                             # Label parsing unit tests
pixi run test_pooling                            # Temporal pooling module unit tests
pixi run test_dataset_integrity                  # Data integrity checks on a built dataset (needs `data/dataset/`)
pixi run -e tracker test_tracking_postprocessing # Postprocessing logic unit tests
pixi run -e tracker test_tracker                 # SAM 3 smoke test on a short clip (GPU)
```

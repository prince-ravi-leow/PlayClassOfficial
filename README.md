# PlayClass

[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22896784.svg)](https://doi.org/10.5281/zenodo.22896784)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22896874.svg)](https://doi.org/10.5281/zenodo.22896874)

Repository for _Hybrid morphokinematic and learnable video representations for
analysis of chicken play behaviour_, Under review, 2026.

A pipeline for play behaviour recognition in videos of poultry with tracking,
postprocessing, feature extraction and classification, along with downstream
analysis.

## Installation

### Prerequisites

All code was developed and tested on Ubuntu 24.04 (linux-64) with CUDA 12.6.

All environments and commands are managed via [pixi](https://pixi.sh) and
defined in `pixi.toml`.

To install pixi, run:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Dependencies

Run the following commands to setup the main dependencies for this project.

```sh
git submodule update --init --recursive
pixi install
```

Additional environments are needed for specific pipeline stages:

```bash
pixi install -e tracker    # SAM3; tracker evaluation
pixi install -e gs2        # Grounded-SAM-2; tracker evaluation
pixi install -e videoprism # JAX
pixi run -e gs2 setup_gs2  # Setup Grounded-SAM 2 (download weights + extra pkgs)
pixi run setup_vjepa2      # Setup V-JEPA 2 (download VJP 2 and 2.1 weights)
```

- To install SAM 3, you will need apply for approval at:
  <https://huggingface.co/facebook/sam3>

### Data

Research data (videos, labels, pipeline outputs) is hosted on
[Zenodo](https://doi.org/10.5281/zenodo.22896874). Download and unpack into the
repository root:

```bash
tar -xzf playclass_zenodo.tar.gz
```

See [data/README.md](data/README.md) for the full directory structure.

## Usage

See [docs/](docs) for detailed instructions for the pipeline scripts.

Files in `scripts/` and `pipeline/` hold user-facing scripts. Files in `src/`
holds reusable library modules.

| Stage                                                                 | Docs                                                 | Environment             |
| --------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------- |
| 1. Data                                                               | [data/README.md](data/README.md)                     | —                       |
| 2. Tracking                                                           | [docs/2_tracking.md](docs/2_tracking.md)             | `tracker`, `gs2`        |
| 3. Tracker evaluation _(optional)_                                    | [docs/3_tracker_eval.md](docs/3_tracker_eval.md)     | `tracker`, `gs2`        |
| 4. Postprocessing                                                     | [docs/4_postprocessing.md](docs/4_postprocessing.md) | `tracker`               |
| 5. Build dataset                                                      | [docs/5_dataset.md](docs/5_dataset.md)               | `default`               |
| 6. Embeddings                                                         | [docs/6_embeddings.md](docs/6_embeddings.md)         | `default`, `videoprism` |
| 7. Classification                                                     | [docs/7_classification.md](docs/7_classification.md) | `default`               |
| 8. Analysis (clustering, classification figures, feature attribution) | [docs/8_analysis.md](docs/8_analysis.md)             | `default`               |

### Tests

```sh
pixi run test_features                    # Feature extraction unit tests
pixi run test_labels                      # Label parsing unit tests
pixi run test_pooling                     # Pooling unit tests
pixi run test_val_rotation                # Cross-validation tests
pixi run test_tracking_postprocessing     # Tracking postprocessing unit tests
pixi run test_tracker                     # Tracker unit tests (requires CUDA)
pixi run test_dataset_integrity           # Data integrity checks (requires a built dataset)
```

## Citation

```bibtex
@article{leow2026hybrid,
  title  = {Hybrid morphokinematic and learnable video representations for analysis of chicken play behaviour},
  author = {Leow, Prince Ravi and Scheidwasser, Neil and Oscarsson, Rebecca and Jensen, Per and Bhatt, Samir and Duch{\^e}ne, David A.},
  year   = {2026},
  note   = {Under review}
}
```

## License

[MIT](LICENSE)

"""k x solver silhouette grid for the dropna variant.

Writes the grid CSV and scaled array to the clustering output directory.
Run once per dataset release; the figure notebook reads the outputs.

    pixi run clustering_grid
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src._config import (
    DEFAULT_CLUSTERING_DIR,
    DEFAULT_DATASET_DIR,
    K_RANGE,
    RANDOM_SEED,
)
from src.clustering import CLUSTERERS, ablate_k

KEY = ["video_id", "bird_id", "window"]


def parse_args():
    parser = ArgumentParser(
        description="k x solver silhouette grid for the dropna variant.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing labels.parquet and features_windowed.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CLUSTERING_DIR,
        help="Directory to write grid CSV and scaled array.",
    )
    return parser.parse_args()


def grid_search(X):
    """Run a k x solver silhouette grid search over all CLUSTERERS."""
    solvers = list(CLUSTERERS.keys())
    rows = []
    for solver in solvers:
        sens_k = ablate_k(X, solver=solver, k_range=K_RANGE, random_state=RANDOM_SEED)
        for k_val in K_RANGE:
            rows.append(
                {
                    "solver": solver,
                    "k": k_val,
                    "silhouette": sens_k.loc[k_val, "silhouette"],
                }
            )
    grid = pd.DataFrame(rows)

    pivot = grid.pivot(index="k", columns="solver", values="silhouette").round(3)
    print(pivot.to_markdown())

    mean_sil = grid.groupby("solver")["silhouette"].mean()
    best_solver = mean_sil.idxmax()
    print(
        f"Best solver by mean silhouette: {best_solver} ({mean_sil[best_solver]:.3f})\n"
    )
    return grid, best_solver


def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    lab = pd.read_parquet(
        args.dataset_dir / "labels.parquet", columns=KEY + ["behav_label"]
    )
    lab = lab[lab.behav_label != "social"].reset_index(drop=True)

    fw = lab.merge(
        pd.read_parquet(args.dataset_dir / "features_windowed.parquet"),
        on=KEY,
        how="left",
    )
    feat_cols = [c for c in fw.columns if c not in {*KEY, "behav_label", "n_frames"}]

    X = fw[feat_cols].astype(float)
    print(f"X: {X.shape}\n")

    Z_dropna = StandardScaler().fit_transform(X.dropna())
    print(f"dropna: {Z_dropna.shape[0]} rows\n")

    np.save(out / "Z_dropna.npy", Z_dropna)
    print(f"Saved Z_dropna.npy to {out}/\n")

    print(f"{'=' * 60}\n  dropna  ({Z_dropna.shape})\n{'=' * 60}\n")
    grid, best_solver = grid_search(Z_dropna)
    grid.to_csv(out / "grid_dropna.csv", index=False)
    print(f"Best solver: {best_solver}")
    print(f"\nAll results saved to {out}/")


if __name__ == "__main__":
    main()

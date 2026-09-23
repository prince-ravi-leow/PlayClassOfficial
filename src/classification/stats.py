"""LOCO/LOVO cross-validation aggregation: scalar summary CSV, summed and per-fold confusion matrices, and per-class recall."""

from pathlib import Path

import numpy as np
import pandas as pd

from loguru import logger


def format_confusion_matrix(cm: np.ndarray, labels: list[str]) -> str:
    """Format a confusion matrix as an aligned text table."""
    header = f"{'':>12s}" + "".join(f"{l:>12s}" for l in labels)
    rows = [header]
    for i, row_data in enumerate(cm):
        rows.append(f"{labels[i]:>12s}" + "".join(f"{v:>12d}" for v in row_data))
    return "\n".join(rows)


def aggregate_scalars(
    fold_results: list[dict],
    scalar_keys: list[str],
    run_dir: Path,
    prefix: str = "loco",
):
    """Build summary CSV with MEAN/STD rows and log per-metric stats."""
    rows = [{k: r[k] for k in scalar_keys} for r in fold_results]
    df = pd.DataFrame(rows)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    mean_row = {c: df[c].mean() for c in numeric_cols}
    std_row = {c: df[c].std() for c in numeric_cols}
    for c in df.columns:
        if c not in numeric_cols:
            mean_row[c] = "MEAN" if c == df.columns[0] else ""
            std_row[c] = "STD" if c == df.columns[0] else ""

    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    summary_path = run_dir / f"{prefix}_summary.csv"
    df.to_csv(summary_path, index=False)
    logger.info(f"{prefix.upper()} summary saved to {summary_path}")

    for col in numeric_cols:
        values = [r[col] for r in fold_results if col in r]
        logger.info(
            f"{prefix.upper()} {col}: {np.mean(values):.4f} +/- {np.std(values):.4f}"
        )


def _macro_f1_from_cm(cm: np.ndarray) -> float:
    """Compute macro F1 from a confusion matrix."""
    n_classes = cm.shape[0]
    f1s = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return np.mean(f1s)


def _save_per_fold_cms(
    fold_results: list[dict],
    key: str,
    run_dir: Path,
    prefix: str,
):
    """Stack per-fold confusion matrices and save as ``{prefix}_{key}s.npy``."""
    stacked = np.stack([r[key] for r in fold_results])
    out_path = run_dir / f"{prefix}_{key}s.npy"
    np.save(out_path, stacked)
    logger.info(f"Per-fold {key}s saved to {out_path} — shape {stacked.shape}")


def _save_recall_txt(
    fold_results: list[dict],
    key: str,
    run_dir: Path,
    labels: list[str],
    prefix: str,
):
    """Derive per-fold per-class recall from CMs and save as a text table."""
    n_folds = len(fold_results)
    recalls = {c: np.zeros(n_folds) for c in labels}
    for i, r in enumerate(fold_results):
        cm = r[key]
        for j, c in enumerate(labels):
            row_sum = cm[j].sum()
            recalls[c][i] = cm[j, j] / row_sum if row_sum > 0 else 0.0

    header = f"{'fold':<8s}" + "".join(f"{c:>14s}" for c in labels)
    lines = [header]
    for i in range(n_folds):
        lines.append(
            f"{'fold_' + str(i):<8s}"
            + "".join(f"{recalls[c][i]:>14.4f}" for c in labels)
        )
    lines.append(
        f"{'MEAN':<8s}" + "".join(f"{recalls[c].mean():>14.4f}" for c in labels)
    )
    lines.append(f"{'STD':<8s}" + "".join(f"{recalls[c].std():>14.4f}" for c in labels))

    split = key.replace("_confusion_matrix", "")
    out_path = run_dir / f"{prefix}_{split}_recall.txt"
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Per-fold recall saved to {out_path}")


def aggregate_confusion_matrices(
    fold_results: list[dict],
    cm_keys: list[str],
    run_dir: Path,
    labels: list[str],
    prefix: str = "loco",
) -> dict[str, float]:
    """Sum and write confusion matrices across folds.

    Also saves per-fold CM stacks (``*_confusion_matrixs.npy``) and per-fold
    recall tables (``*_recall.txt``) for test splits.

    Returns pooled macro F1 per split, e.g. ``{"test_macro_f1": 0.74, ...}``.
    """
    pooled_f1s = {}
    for key in cm_keys:
        summed_cm = sum(r[key] for r in fold_results)

        # Save as .npy for programmatic reuse
        np.save(run_dir / f"{prefix}_{key}.npy", summed_cm)

        # Save as txt for human inspection
        cm_text = format_confusion_matrix(summed_cm, labels)
        cm_path = run_dir / f"{prefix}_{key}.txt"
        cm_path.write_text(f"# Summed {key} (rows=true, cols=predicted):\n{cm_text}\n")
        logger.info(f"Summed {key} saved to {cm_path}")

        # Per-fold CMs and recall for test splits
        if "test" in key:
            _save_per_fold_cms(fold_results, key, run_dir, prefix)
            _save_recall_txt(fold_results, key, run_dir, labels, prefix)

        # Compute pooled F1 from summed CM
        macro_f1 = _macro_f1_from_cm(summed_cm)
        split = key.replace("_confusion_matrix", "")
        pooled_f1s[f"{split}_macro_f1"] = macro_f1

        if "test" in key:
            logger.info(f"\n{'='*60}")
            logger.info(f"POOLED TEST MACRO F1: {macro_f1:.4f}")
            logger.info(f"{'='*60}")
            logger.info(f"\n{cm_text}\n")

    return pooled_f1s


def aggregate_metrics(
    fold_results: list[dict], run_dir: Path, labels: list[str], prefix: str = "lovo"
) -> None:
    """Aggregate per-fold metrics and write summary files.

    Parameters
    ----------
    fold_results : list[dict]
        Per-fold dicts from ``evaluate_fold``.
    run_dir : Path
        Output directory.
    labels : list[str]
        Class names for confusion matrix formatting.
    prefix : str
        Filename prefix (``"loco"`` or ``"lovo"``).
    """
    scalar_keys = []
    cm_keys = []
    for k, v in fold_results[0].items():
        if isinstance(v, (int, float, str)):
            scalar_keys.append(k)
        elif isinstance(v, np.ndarray):
            cm_keys.append(k)
        else:
            raise ValueError(f"Unexpected type {type(v)} for metric '{k}'")

    aggregate_scalars(fold_results, scalar_keys, run_dir, prefix=prefix)
    pooled_f1s = aggregate_confusion_matrices(
        fold_results, cm_keys, run_dir, labels, prefix=prefix
    )

    # Append POOLED row to summary CSV
    summary_path = run_dir / f"{prefix}_summary.csv"
    df = pd.read_csv(summary_path)
    pooled_row = {df.columns[0]: "POOLED"}
    pooled_row.update(pooled_f1s)
    df = pd.concat([df, pd.DataFrame([pooled_row])], ignore_index=True)
    df.to_csv(summary_path, index=False)

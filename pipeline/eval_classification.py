"""Post-hoc evaluation and table assembly for classification runs.

Subcommands
-----------
evaluate
    Load checkpoints for a run and produce per-fold confusion matrices,
    per-fold recall, and per-sample predictions.

tables
    Assemble Supp Table 3 (segment sweep) and Supp Table 4 (ablation)
    from the ``loco_summary.csv`` and ``cfg.json`` files across all runs.

Usage::

    pixi run eval_classification evaluate <run_dir>
    pixi run eval_classification evaluate --all
    pixi run eval_classification tables
"""

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src._config import DEFAULT_CHECKPOINT_DIR, LABEL_ORDER

# ---------------------------------------------------------------------------
# evaluate: checkpoint-based inference
# ---------------------------------------------------------------------------

INPUT_TO_MODEL = {
    "embeddings_dinov3_vitb": ("DINOv3", "B"),
    "embeddings_dinov3_vitl": ("DINOv3", "L"),
    "embeddings_vjepa2_vitl_temporal": ("V-JEPA 2", "L"),
    "embeddings_videoprism_vitb_temporal": ("VideoPrism", "B"),
    "embeddings_videoprism_vitl_temporal": ("VideoPrism", "L"),
    "embeddings_vjepa21_vitb_temporal": ("V-JEPA 2.1", "B"),
    "embeddings_vjepa21_vitl_temporal": ("V-JEPA 2.1", "L"),
}


def _load_cfg(run_dir: Path) -> dict:
    return json.loads((run_dir / "cfg.json").read_text())


def _build_dm_and_model_cls(cfg: dict):
    """Reconstruct datamodule and model class from a run config."""
    import torch

    from src.classification.datamodule import BehaviourDataModule
    from src.classification.model_selection import LOCO
    from src.classification.models import MODEL_REGISTRY
    from src.classification.trainer import BehaviourClassifier
    from src.classification.utils import parse_input

    use_features, use_embeddings, embeddings_files = parse_input(cfg["input"])
    backbone_cls, temporal = MODEL_REGISTRY[cfg["model"]]
    splitter = LOCO()

    dm = BehaviourDataModule(
        dataset_dir=cfg.get("dataset_dir", "data/dataset"),
        splitter=splitter,
        batch_size=int(cfg.get("batch_size", 32)),
        exclude=cfg.get("exclude"),
        use_features=use_features,
        use_embeddings=use_embeddings,
        temporal=temporal,
        embeddings_files=embeddings_files or ["embeddings.pt"],
        n_segments=int(cfg.get("n_segments", 12)),
    )
    dm.prepare_data()

    return dm, backbone_cls, cfg


def _make_model(dm, backbone_cls, cfg):
    from src.classification.trainer import BehaviourClassifier

    backbone_kwargs = {}
    if dm.flat_dim > 0:
        backbone_kwargs["d_flat"] = dm.flat_dim
    if cfg.get("dropout") is not None:
        backbone_kwargs["dropout"] = float(cfg["dropout"])
    if cfg.get("d_hidden") is not None:
        backbone_kwargs["d_hidden"] = int(cfg["d_hidden"])

    return BehaviourClassifier(
        backbone=backbone_cls(dm.data_dim, dm.n_classes, **backbone_kwargs),
        n_classes=dm.n_classes,
        lr=float(cfg.get("lr", 1e-3)),
        class_weights=dm.class_weights,
        label_smoothing=float(cfg.get("label_smoothing", 0.1)),
        feature_dropout=float(cfg.get("feature_dropout", 0.0)),
    )


def _ckpt_for_fold(run_dir: Path, fold_idx: int, test_id: str) -> Path:
    ckpts = sorted(
        (run_dir / f"fold_{fold_idx}_{test_id}" / "checkpoints").glob("*.ckpt")
    )
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint for fold {fold_idx} ({test_id}) under {run_dir}"
        )
    return sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]


def evaluate_run(run_dir: Path, device: str = "cuda:0", predictions: bool = True):
    """Run inference on all folds, saving CMs, recall, and optionally predictions."""
    import torch
    import torch.nn.functional as F

    from src.classification.stats import _save_per_fold_cms, _save_recall_txt

    run_dir = Path(run_dir)
    cfg = _load_cfg(run_dir)
    dm, backbone_cls, cfg = _build_dm_and_model_cls(cfg)

    exclude = cfg.get("exclude")
    label_order = LABEL_ORDER.copy()
    if exclude:
        exclude_list = [exclude] if isinstance(exclude, str) else exclude
        label_order = [l for l in label_order if l not in exclude_list]

    labels_df = None
    if predictions:
        dataset_dir = Path(cfg.get("dataset_dir", "data/dataset"))
        labels_df = pd.read_parquet(dataset_dir / "labels.parquet")
        if exclude:
            labels_df = labels_df[
                ~labels_df["behav_label"].isin(exclude_list)
            ].reset_index(drop=True)

    class_names = list(dm._dataset.label_encoder.lab2ind.keys())
    torch_device = torch.device(device)
    folds = list(dm.splitter.split(dm.video_ids))

    fold_cms = []
    pred_rows = []

    for fold_idx, (test_id, val_id) in enumerate(folds):
        dm.set_fold(test_id=test_id, val_id=val_id)
        dm.setup()

        model = _make_model(dm, backbone_cls, cfg).to(torch_device)
        ckpt = torch.load(
            _ckpt_for_fold(run_dir, fold_idx, test_id),
            map_location=torch_device,
            weights_only=True,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        cm = torch.zeros(dm.n_classes, dm.n_classes, dtype=torch.long)
        sample_indices = dm.test_ds.indices if predictions else None
        offset = 0

        with torch.no_grad():
            for batch in dm.test_dataloader():
                batch_size = len(batch["label"])
                batch = {k: v.to(torch_device) for k, v in batch.items()}
                flat = batch.get("flat")
                logits = (
                    model.backbone(batch["data"], flat=flat)
                    if flat is not None
                    else model.backbone(batch["data"])
                )
                preds = logits.argmax(dim=-1)
                for t, p in zip(batch["label"], preds):
                    cm[t, p] += 1

                if predictions:
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    true = batch["label"].cpu().numpy()
                    for j in range(batch_size):
                        dataset_idx = sample_indices[offset + j]
                        row = {
                            "video_id": labels_df.iloc[dataset_idx]["video_id"],
                            "bird_id": labels_df.iloc[dataset_idx]["bird_id"],
                            "window": labels_df.iloc[dataset_idx]["window"],
                            "true_label": class_names[int(true[j])],
                        }
                        for ci, cn in enumerate(class_names):
                            row[f"prob_{cn}"] = probs[j, ci]
                        pred_rows.append(row)
                    offset += batch_size

        fold_cms.append(cm.numpy())
        print(f"  Fold {fold_idx} ({test_id}): done")

        del model, ckpt
        torch.cuda.empty_cache()

    # Save per-fold CMs
    stacked = np.stack(fold_cms)
    np.save(run_dir / "loco_test_confusion_matrices.npy", stacked)
    print(f"  Saved loco_test_confusion_matrices.npy — shape {stacked.shape}")

    # Save per-fold recall
    fold_dicts = [{"test_confusion_matrix": cm} for cm in fold_cms]
    _save_recall_txt(fold_dicts, "test_confusion_matrix", run_dir, label_order, "loco")
    print(f"  Saved loco_test_recall.txt")

    # Save predictions
    if predictions and pred_rows:
        df = pd.DataFrame(pred_rows)
        prob_cols = [f"prob_{c}" for c in class_names]
        df["pred_label"] = (
            df[prob_cols].idxmax(axis=1).str.replace("prob_", "", regex=False)
        )
        df.to_csv(run_dir / "predictions.csv", index=False)
        print(f"  Saved predictions.csv — {len(df)} rows")


def cmd_evaluate(args):
    """Subcommand: evaluate one or all runs."""
    eval_dir = Path(args.eval_dir)

    if args.all:
        run_dirs = sorted(eval_dir.iterdir())
    else:
        run_dirs = [Path(args.run_dir)]

    for run_dir in run_dirs:
        if not (run_dir / "cfg.json").exists():
            continue
        # Skip if already has all outputs (unless --force)
        has_cms = (run_dir / "loco_test_confusion_matrices.npy").exists()
        has_recall = (run_dir / "loco_test_recall.txt").exists()
        has_preds = (run_dir / "predictions.csv").exists()
        if has_cms and has_recall and has_preds and not args.force:
            print(f"{run_dir.name}: all outputs exist, skipping (use --force to redo)")
            continue

        print(f"\n{run_dir.name}")
        evaluate_run(
            run_dir,
            device=args.device,
            predictions=args.predictions,
        )

    print("\nDone!")


# ---------------------------------------------------------------------------
# tables: assemble sweep and ablation tables from loco_summary.csv files
# ---------------------------------------------------------------------------

K_VALUES = [4, 8, 12, 16, 24, 32, 48, 64]


def _read_run_summary(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Return (cfg, summary_df) for a single run."""
    cfg = _load_cfg(run_dir)
    summary = pd.read_csv(run_dir / "loco_summary.csv")
    return cfg, summary


def _pooled_f1_and_std(summary: pd.DataFrame) -> tuple[float, float]:
    """Extract pooled F1 and fold-level SD from a loco_summary.csv."""
    pooled_row = summary[summary.iloc[:, 0] == "POOLED"]
    pooled_f1 = float(pooled_row["test_macro_f1"].iloc[0]) * 100

    std_row = summary[summary.iloc[:, 0] == "STD"]
    fold_std = float(std_row["test_macro_f1"].iloc[0]) * 100

    return pooled_f1, fold_std


def build_sweep_table(eval_dir: Path) -> pd.DataFrame:
    """Build the segment sweep table (Supp Table 3) from all run dirs."""
    rows = []

    for run_dir in sorted(eval_dir.iterdir()):
        cfg_path = run_dir / "cfg.json"
        summary_path = run_dir / "loco_summary.csv"
        if not cfg_path.exists() or not summary_path.exists():
            continue

        cfg, summary = _read_run_summary(run_dir)
        k = cfg.get("n_segments")
        input_str = cfg["input"]
        model_name = cfg["model"]

        # Features-only MLP baseline (no temporal segments)
        if input_str == "features" and model_name == "mlp":
            pooled_f1, fold_std = _pooled_f1_and_std(summary)
            rows.append({
                "model": "Morphokinematic features",
                "backbone": "---",
                "classifier": "MLP",
                "k": None,
                "f1": pooled_f1,
                "std": fold_std,
                "run_dir": run_dir.name,
            })
            continue

        # Embedding-based runs
        emb_key = input_str.replace("features+", "")
        if emb_key not in INPUT_TO_MODEL:
            continue

        display_model, backbone = INPUT_TO_MODEL[emb_key]
        pooled_f1, fold_std = _pooled_f1_and_std(summary)

        # Hybrid model
        if input_str.startswith("features+"):
            display_model = f"Hybrid: Feat. + {display_model}"
            classifier = "MLP" if model_name == "mlp" else "1D-CNN"
        else:
            classifier = "1D-CNN"

        rows.append({
            "model": display_model,
            "backbone": backbone,
            "classifier": classifier,
            "k": k,
            "f1": pooled_f1,
            "std": fold_std,
            "run_dir": run_dir.name,
        })

    return pd.DataFrame(rows)


def build_ablation_table(eval_dir: Path) -> pd.DataFrame:
    """Build the ablation table (Supp Table 4) from the hybrid runs."""
    # Find all hybrid runs (features + vjepa21 vitl, K=24, temporal_cnn2)
    full_model = None
    ablation_rows = []

    for run_dir in sorted(eval_dir.iterdir()):
        cfg_path = run_dir / "cfg.json"
        summary_path = run_dir / "loco_summary.csv"
        if not cfg_path.exists() or not summary_path.exists():
            continue

        cfg, summary = _read_run_summary(run_dir)
        input_str = cfg["input"]

        if "vjepa21_vitl" not in input_str and input_str != "features":
            continue

        pooled_f1, fold_std = _pooled_f1_and_std(summary)
        model_name = cfg["model"]
        k = cfg.get("n_segments")
        epochs = cfg.get("epochs", 5)
        class_weights = cfg.get("class_weights", "inv-sqrt")
        label_smoothing = cfg.get("label_smoothing", 0.1)

        # Full model: features+vjepa21_vitl_temporal, temporal_cnn2, K=24, 5 epochs, inv-sqrt, ls=0.1
        is_full = (
            input_str == "features+embeddings_vjepa21_vitl_temporal"
            and model_name == "temporal_cnn2"
            and k == 24
            and epochs == 5
            and class_weights == "inv-sqrt"
            and label_smoothing == 0.1
        )

        if is_full:
            full_model = {"f1": pooled_f1, "std": fold_std}
            ablation_rows.append({
                "configuration": "Full model",
                "original": "-",
                "f1": pooled_f1,
                "std": fold_std,
                "delta": None,
            })
            continue

        # Identify which ablation this is
        if input_str == "features+embeddings_vjepa21_vitl_temporal" and k == 24:
            if epochs == 1:
                config = "Training: 1 epoch"
                original = "5 epochs"
            elif class_weights == "none":
                config = "Class weights: no"
                original = "yes"
            elif label_smoothing == 0.0:
                config = "Label smoothing: no"
                original = "yes"
            else:
                continue

            ablation_rows.append({
                "configuration": config,
                "original": original,
                "f1": pooled_f1,
                "std": fold_std,
                "delta": None,
            })
        elif (
            input_str == "features+embeddings_vjepa21_vitl_temporal"
            and model_name == "mlp"
        ):
            ablation_rows.append({
                "configuration": "Architecture: MLP classifier",
                "original": "1D-CNN",
                "f1": pooled_f1,
                "std": fold_std,
                "delta": None,
            })

    # Compute deltas
    if full_model:
        for row in ablation_rows:
            if row["configuration"] != "Full model":
                row["delta"] = row["f1"] - full_model["f1"]

    return pd.DataFrame(ablation_rows)


def cmd_tables(args):
    """Subcommand: assemble sweep and ablation tables."""
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = build_sweep_table(eval_dir)
    sweep_path = out_dir / "table_segment_sweep.csv"
    sweep.to_csv(sweep_path, index=False)
    print(f"Segment sweep table: {sweep_path} ({len(sweep)} rows)")

    # Pretty-print sweep pivot
    embedding_runs = sweep[sweep["k"].notna()].copy()
    if not embedding_runs.empty:
        embedding_runs["cell"] = embedding_runs.apply(
            lambda r: f"{r['f1']:.1f} ±{r['std']:.1f}", axis=1
        )
        pivot = embedding_runs.pivot_table(
            index=["model", "backbone"],
            columns="k",
            values="cell",
            aggfunc="first",
        )
        pivot = pivot.reindex(columns=K_VALUES)
        print(f"\n{pivot.to_string()}\n")

    # Features baseline
    baseline = sweep[sweep["k"].isna()]
    if not baseline.empty:
        for _, r in baseline.iterrows():
            print(f"{r['model']}: {r['f1']:.1f} ±{r['std']:.1f}")

    ablation = build_ablation_table(eval_dir)
    ablation_path = out_dir / "table_ablation.csv"
    ablation.to_csv(ablation_path, index=False)
    print(f"\nAblation table: {ablation_path} ({len(ablation)} rows)")
    print(ablation.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = ArgumentParser(
        description="Post-hoc evaluation and table assembly for classification runs.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # evaluate
    ev = sub.add_parser(
        "evaluate",
        help="Run checkpoint-based evaluation for a run.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    ev.add_argument("run_dir", nargs="?", help="Path to a single run directory.")
    ev.add_argument(
        "--all", action="store_true", help="Evaluate all runs in --eval-dir."
    )
    ev.add_argument(
        "--eval-dir", default=DEFAULT_CHECKPOINT_DIR, help="Root eval directory."
    )
    ev.add_argument("--device", default="cuda:0", help="Torch device.")
    ev.add_argument(
        "--no-predictions",
        dest="predictions",
        action="store_false",
        help="Skip saving predictions.csv (faster).",
    )
    ev.add_argument(
        "--force", action="store_true", help="Re-run even if outputs exist."
    )

    # tables
    tb = sub.add_parser(
        "tables",
        help="Assemble sweep and ablation tables from existing summaries.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    tb.add_argument(
        "--eval-dir", default=DEFAULT_CHECKPOINT_DIR, help="Root eval directory."
    )
    tb.add_argument(
        "--output-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory to write table CSVs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "tables":
        cmd_tables(args)


if __name__ == "__main__":
    main()

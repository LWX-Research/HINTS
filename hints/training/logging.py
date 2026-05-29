from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from ..settings import Settings


def setup_logger(settings: Settings) -> tuple[logging.Logger, Path]:
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = settings.logs_dir / f"{run_ts}_3d"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    logger = logging.getLogger(f"HINTS_3D_{run_ts}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.info("===== HINTS 3D Training =====")
    logger.info("Log dir: %s", log_dir)
    for key, value in settings.to_dict().items():
        logger.info("Config: %s = %s", key, value)

    config_path = log_dir / "config.json"
    config_path.write_text(json.dumps(settings.to_dict(), indent=2, default=str) + "\n")

    return logger, log_dir


def save_fold_metrics_csv(fold: int, records: list[dict], log_dir: Path) -> None:
    path = log_dir / f"fold{fold}_metrics.csv"
    keys = ["epoch", "loss", "cox", "contra", "train_ci", "train_auc", "val_ci", "val_auc"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def log_epoch_info(
    logger: logging.Logger,
    settings: Settings,
    fold: int,
    epoch: int,
    loss: float,
    cox: float,
    contra: float,
    train_c: float,
    train_auc: float,
    valid_c: float,
    valid_auc: float,
) -> dict:
    parts = [f"Fold {fold} Epoch {epoch + 1}/{settings.epochs}"]
    parts.append(f"Loss={loss:.4f}")
    parts.append(f"Cox={cox:.4f}")
    parts.append(f"Contra={contra:.4f}")
    if not np.isnan(train_c):
        parts.append(f"Train CI={train_c:.4f}")
    if not np.isnan(train_auc):
        parts.append(f"Train AUC={train_auc:.4f}")
    parts.append(f"Val CI={valid_c:.4f}")
    parts.append(f"Val AUC={valid_auc:.4f}")
    logger.info(" | ".join(parts))
    return {
        "epoch": epoch + 1,
        "loss": round(loss, 6),
        "cox": round(cox, 6),
        "contra": round(contra, 6),
        "train_ci": round(train_c, 4) if not np.isnan(train_c) else None,
        "train_auc": round(train_auc, 4) if not np.isnan(train_auc) else None,
        "val_ci": round(valid_c, 4) if not np.isnan(valid_c) else None,
        "val_auc": round(valid_auc, 4) if not np.isnan(valid_auc) else None,
    }


def report_results(logger: logging.Logger, fold_metrics: list, log_dir: Path) -> None:
    logger.info("\n===== Final Results =====")
    cis, aucs = [], []
    folds_info = []
    for fold, train_c, train_auc, best_ci, best_auc in fold_metrics:
        logger.info("Fold %d: Best CI=%.4f Best AUC=%.4f", fold, best_ci, best_auc)
        cis.append(best_ci)
        aucs.append(best_auc)
        folds_info.append({
            "fold": fold,
            "best_ci": round(best_ci, 4),
            "best_auc": round(best_auc, 4),
        })
    if cis and aucs:
        logger.info("Average CI=%.4f ± %.4f", np.mean(cis), np.std(cis))
        logger.info("Average AUC=%.4f ± %.4f", np.mean(aucs), np.std(aucs))

    results = {
        "folds": folds_info,
        "mean_ci": round(float(np.mean(cis)), 4) if cis else None,
        "std_ci": round(float(np.std(cis)), 4) if cis else None,
        "mean_auc": round(float(np.mean(aucs)), 4) if aucs else None,
        "std_auc": round(float(np.std(aucs)), 4) if aucs else None,
    }
    (log_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

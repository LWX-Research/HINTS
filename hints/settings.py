from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    if os.environ.get(env_name) in {None, "", "0"}:
        os.environ[env_name] = "1"

import torch


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in {None, ""} else default


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in {None, ""} else default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in {None, ""}:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _get_env_str(name: str, default: str | None) -> str | None:
    value = os.environ.get(name)
    return value if value not in {None, ""} else default


@dataclass
class Settings:
    # ---- Paths ----
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    dataset_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    device: str = field(init=False)

    graph_build_mode: str = field(default_factory=lambda: _get_env_str("HINTS_GRAPH_BUILD_MODE", "auto"))

    # ---- 3D supervoxel ----
    n_segments: int = field(default_factory=lambda: _get_env_int("HINTS_N_SEGMENTS", 200))
    roi_margin: int = field(default_factory=lambda: _get_env_int("HINTS_ROI_MARGIN", 5))

    # ---- Supervoxel ----
    supervoxel_backend: str = field(default_factory=lambda: _get_env_str("HINTS_SUPERVOXEL_BACKEND", "skimage"))
    slic_compactness: float = field(default_factory=lambda: _get_env_float("HINTS_SLIC_COMPACTNESS", 0.1))
    slic_sigma: float = field(default_factory=lambda: _get_env_float("HINTS_SLIC_SIGMA", 1.0))

    # ---- Training ----
    k_fold: int = field(default_factory=lambda: _get_env_int("HINTS_K_FOLD", 5))
    seed: int = field(default_factory=lambda: _get_env_int("HINTS_SEED", 111111))
    batch_size: int = field(default_factory=lambda: _get_env_int("HINTS_BATCH_SIZE", 4))
    num_workers: int = field(default_factory=lambda: _get_env_int("HINTS_NUM_WORKERS", 4))
    pin_memory: bool = field(default_factory=lambda: _get_env_bool("HINTS_PIN_MEMORY", torch.cuda.is_available()))
    epochs: int = field(default_factory=lambda: _get_env_int("HINTS_EPOCHS", 50))
    learning_rate: float = field(default_factory=lambda: _get_env_float("HINTS_LR", 0.01))
    eta_min: float = field(default_factory=lambda: _get_env_float("HINTS_ETA_MIN", 0.1))
    lambda_1: float = field(default_factory=lambda: _get_env_float("HINTS_LAMBDA_1", 0.0))
    eval_interval: int = field(default_factory=lambda: _get_env_int("HINTS_EVAL_INTERVAL", 10))
    eval_train_metrics: bool = field(default_factory=lambda: _get_env_bool("HINTS_EVAL_TRAIN_METRICS", False))

    # ---- Model (compatible with V1 downstream) ----
    in_channels: int = 6  # cz, cy, cx, voxel_count, mean, std
    num_prototypes: int = field(default_factory=lambda: _get_env_int("HINTS_NUM_PROTOTYPES", 3))
    tau: float = field(default_factory=lambda: _get_env_float("HINTS_TAU", 0.01))
    contra_weight: float = field(default_factory=lambda: _get_env_float("HINTS_CONTRA_WEIGHT", 1.0))
    hidden_dim: int = field(default_factory=lambda: _get_env_int("HINTS_HIDDEN_DIM", 32))
    num_nodes: int = field(default_factory=lambda: _get_env_int("HINTS_NUM_NODES", 128))

    # ---- Pretrained ----
    pretrained_path: str | None = field(default_factory=lambda: os.environ.get("HINTS_PRETRAINED_PATH"))

    # ---- Unified data (canonical format) ----
    unified_dir: str | None = field(default_factory=lambda: _get_env_str("HINTS_UNIFIED_DIR", None))

    # ---- Legacy raw data (deprecated — use convert + unified_dir instead) ----
    raw_data_dir: str | None = field(default_factory=lambda: _get_env_str("HINTS_RAW_DIR", None))
    preprocess_output_dir: str | None = field(default_factory=lambda: _get_env_str("HINTS_PREPROCESS_OUTPUT_DIR", None))

    def __post_init__(self) -> None:
        if self.unified_dir is not None:
            self.dataset_dir = Path(self.unified_dir)
        elif self.raw_data_dir is not None:
            self.dataset_dir = Path(self.raw_data_dir)
        else:
            dataset_override = os.environ.get("HINTS_DATASET_DIR")
            if dataset_override:
                self.dataset_dir = Path(dataset_override)
            else:
                self.dataset_dir = self.base_dir.parent / "dataset" / "MICCAI_BraTS2020_TrainingData"
        self.logs_dir = self.base_dir / "outputs" / "logs"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.pretrained_path is not None and self.pretrained_path.lower() == "none":
            self.pretrained_path = None
        if self.preprocess_output_dir is None:
            self.preprocess_output_dir = str(self.base_dir.parent / "dataset" / "Hints-3D-BraTS")

    @property
    def label_path(self) -> str | None:
        # Unified dir: only look there, no fallthrough to other datasets
        if self.unified_dir is not None:
            u = Path(self.unified_dir)
            for name in ("survival.csv", "1352.csv", "survival_info.csv"):
                candidate = u / name
                if candidate.exists():
                    return str(candidate)
            return None

        candidates: list[Path] = []
        if self.raw_data_dir is not None:
            raw = Path(self.raw_data_dir)
            candidates.extend([
                raw / "survival.csv", raw / "1352.csv", raw / "survival_info.csv",
                raw.parent / "survival.csv", raw.parent / "1352.csv", raw.parent / "survival_info.csv",
            ])
        if self.preprocess_output_dir is not None:
            pp = Path(self.preprocess_output_dir)
            candidates.extend([pp / "survival.csv", pp / "1352.csv", pp / "survival_info.csv"])
        candidates.extend([
            self.dataset_dir / "survival.csv", self.dataset_dir / "1352.csv", self.dataset_dir / "survival_info.csv",
            self.dataset_dir.parent / "survival.csv", self.dataset_dir.parent / "1352.csv", self.dataset_dir.parent / "survival_info.csv",
        ])
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def apply_overrides(self, **kwargs) -> "Settings":
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                if key == "pretrained_path" and isinstance(value, str) and value.lower() == "none":
                    value = None
                setattr(self, key, value)
        return self

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["label_path"] = self.label_path
        payload["dataset_dir"] = str(self.dataset_dir)
        payload["logs_dir"] = str(self.logs_dir)
        payload["base_dir"] = str(self.base_dir)
        return payload

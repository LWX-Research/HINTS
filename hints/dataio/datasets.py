from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from ..settings import Settings
from .unified import UnifiedDataset, _MODALITY_FILE_MAP, _filter_edges, _pad_or_truncate_6d

def _normalize_labels(survival_path: str) -> pd.DataFrame:
    """Load and normalize survival labels to standard format (MRN, dead, deadtime).

    Handles both NPC format (MRN, dead, deadtime) and BraTS format
    (Brats20ID, Survival_days) with automatic detection.
    """
    import re

    raw = pd.read_csv(survival_path)
    cols = set(raw.columns)

    if "Brats20ID" in cols:
        out = pd.DataFrame()
        out["MRN"] = raw["Brats20ID"].astype(str)

        _alive_re = re.compile(r"\d+")

        def _parse_survival(val) -> tuple[int, float]:
            s = str(val).strip()
            if "ALIVE" in s.upper():
                nums = _alive_re.findall(s)
                days = float(nums[0]) if nums else 0.0
                return 0, days / 30.4375
            return 1, float(s) / 30.4375

        parsed = raw["Survival_days"].apply(_parse_survival)
        out["dead"] = [p[0] for p in parsed]
        out["deadtime"] = [p[1] for p in parsed]
        out = out[out["deadtime"] > 0]
        out["MRN"] = out["MRN"].astype(str)
        return out.set_index("MRN")

    survival = raw.copy()
    survival["MRN"] = survival["MRN"].astype(str)
    return survival.set_index("MRN")




class Graph3DSurvivalDataset(Dataset):
    """Loads pre-computed per-modality 3D graph files from offline preprocessing."""

    def __init__(
        self,
        graph_dir: str | Path,
        survival_path: str | None,
        num_nodes: int,
        transform=None,
    ):
        super().__init__(transform=transform)
        self.graph_dir = Path(graph_dir)
        self.num_nodes = num_nodes

        if survival_path is None or not Path(survival_path).exists():
            raise FileNotFoundError(
                "No survival label file found. Place survival.csv, survival_info.csv, or 1352.csv "
                "beside the data directory, or set HINTS_RAW_DIR to point to a dataset with labels."
            )
        self.survival = _normalize_labels(survival_path)

        graphs_dir = self.graph_dir / "graphs"
        t1_files = {f.stem[:-3] for f in graphs_dir.glob("*_t1.pt")}
        self.case_ids = sorted(
            cid for cid in t1_files
            if cid in self.survival.index
            and (graphs_dir / f"{cid}_t1c.pt").exists()
            and (graphs_dir / f"{cid}_t2.pt").exists()
        )

    def len(self) -> int:
        return len(self.case_ids)

    def get(self, idx: int) -> Data:
        case_id = self.case_ids[idx]
        row = self.survival.loc[case_id]

        data = Data(
            mrn=case_id,
            num_nodes=self.num_nodes,
            dead=torch.tensor(int(row["dead"]), dtype=torch.int),
            deadtime=torch.tensor(float(row["deadtime"]), dtype=torch.float32),
        )

        graphs_dir = self.graph_dir / "graphs"
        for model_key, file_key in _MODALITY_FILE_MAP.items():
            fpath = graphs_dir / f"{case_id}_{file_key}.pt"
            saved = torch.load(fpath, weights_only=False)
            x = _pad_or_truncate_6d(saved["x"], self.num_nodes)
            ei = _filter_edges(saved["edge_index"], self.num_nodes)
            data[f"{model_key}_x"] = x
            data[f"{model_key}_edge_index"] = ei

        return data


def _validate_metadata(meta_path: Path, settings: Settings) -> bool:
    """Check that cached graph metadata matches the current settings."""
    try:
        metadata = json.loads(meta_path.read_text())
    except Exception:
        return False
    expected = {
        "n_segments": settings.n_segments,
        "num_nodes": settings.num_nodes,
        "roi_margin": settings.roi_margin,
        "supervoxel_backend": settings.supervoxel_backend,
        "compactness": settings.slic_compactness,
        "sigma": settings.slic_sigma,
        "spatial_mode": "3d",
        "node_feature_dim": 6,
    }
    return all(
        str(metadata.get(k)) == str(v) for k, v in expected.items()
    )


def build_dataset_3d(settings: Settings):
    """Build the appropriate 3D dataset controlled by graph_build_mode.

    Valid modes:
      - "offline": require precomputed graphs; error if missing or metadata mismatch
      - "online":  build graphs from unified format on the fly
      - "auto":    try offline first, fall back to online
    """
    mode = settings.graph_build_mode
    preprocess_dir = settings.preprocess_output_dir
    has_cache = preprocess_dir is not None and Path(preprocess_dir, "graphs").exists()
    cache_ok = has_cache and _validate_metadata(Path(preprocess_dir) / "metadata.json", settings)

    if mode == "offline":
        if not has_cache:
            raise FileNotFoundError(
                f"graph_build_mode='offline' but no cached graphs found at {preprocess_dir}. "
                "Run 'python -m hints.cli preprocess-3d' first, or set HINTS_GRAPH_BUILD_MODE=auto."
            )
        if not cache_ok:
            print("[3D] WARNING: cached graph metadata mismatch — consider re-running preprocess-3d")
        print(f"[3D] Using cached graphs from {preprocess_dir}")
        return Graph3DSurvivalDataset(
            graph_dir=preprocess_dir,
            survival_path=settings.label_path,
            num_nodes=settings.num_nodes,
        )

    if mode == "online":
        if settings.unified_dir is None:
            raise ValueError(
                "graph_build_mode='online' but HINTS_UNIFIED_DIR is not set. "
                "Provide --unified-dir or set HINTS_UNIFIED_DIR."
            )
        print(f"[3D] Building graphs online from {settings.unified_dir}")
        return UnifiedDataset(
            unified_dir=settings.unified_dir,
            num_nodes=settings.num_nodes,
            n_segments=settings.n_segments,
            roi_margin=settings.roi_margin,
            compactness=settings.slic_compactness,
            sigma=settings.slic_sigma,
            supervoxel_backend=settings.supervoxel_backend,
            build_workers=settings.num_workers,
        )

    if mode == "auto":
        if has_cache:
            if not cache_ok:
                print("[3D] WARNING: cached graph metadata mismatch — consider re-running preprocess-3d")
            print(f"[3D] Using cached graphs from {preprocess_dir}")
            return Graph3DSurvivalDataset(
                graph_dir=preprocess_dir,
                survival_path=settings.label_path,
                num_nodes=settings.num_nodes,
            )
        if settings.unified_dir is not None:
            print(f"[3D] Building graphs online from {settings.unified_dir}")
            return UnifiedDataset(
                unified_dir=settings.unified_dir,
                num_nodes=settings.num_nodes,
                n_segments=settings.n_segments,
                roi_margin=settings.roi_margin,
                compactness=settings.slic_compactness,
                sigma=settings.slic_sigma,
                supervoxel_backend=settings.supervoxel_backend,
                build_workers=settings.num_workers,
            )
        raise ValueError(
            "No 3D data source available in auto mode. Set HINTS_PREPROCESS_OUTPUT_DIR for offline graphs "
            "or HINTS_UNIFIED_DIR for online graph building."
        )

    raise ValueError(
        f"Unknown graph_build_mode: '{mode}'. Valid modes: offline, online, auto."
    )

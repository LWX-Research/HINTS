"""Unified dataset format converter and loader.

Produces/consumes a canonical directory layout:

    unified_root/
      survival.csv       # MRN, dead, deadtime
      {case_id}/
        t1.npy           # (D,H,W) float32
        t1c.npy          # (D,H,W) float32
        t2.npy           # (D,H,W) float32
        mask.npy         # (D,H,W) bool  (optional — auto-computed if absent)

Supported source formats: NPC NIfTI, NPC NPY, BraTS NIfTI.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

TARGET_MODALITIES = ("t1", "t1c", "t2")
_MODALITY_FILE_MAP = {"t1": "t1", "t1ce": "t1c", "t2": "t2"}


# ---------------------------------------------------------------------------
#  Detection
# ---------------------------------------------------------------------------

def _detect_source_type(raw_dir: Path) -> str:
    """Return one of 'npc_npy', 'npc_nifti', 'brats' by inspecting the first case."""
    case_dirs = _find_case_dirs(raw_dir)
    if not case_dirs:
        raise ValueError(f"No case directories found in {raw_dir}")
    first = case_dirs[0]

    npy_files = list(first.glob("*.npy"))
    if npy_files:
        return "npc_npy"

    nii_names = [f.name for f in first.glob("*.nii*")]
    if any("t1ce" in n or "flair" in n or "seg" in n for n in nii_names):
        return "brats"
    if any("_image" in n for n in nii_names):
        return "npc_nifti"

    raise ValueError(f"Cannot detect dataset type in: {first}")


def _find_case_dirs(raw_dir: Path) -> list[Path]:
    return sorted(
        p for p in raw_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


# ---------------------------------------------------------------------------
#  Label normalisation
# ---------------------------------------------------------------------------

def _find_and_normalize_labels(raw_dir: Path) -> pd.DataFrame:
    """Search for survival labels, normalise to MRN/dead/deadtime, return DataFrame indexed by MRN."""
    candidates = [
        raw_dir / name
        for name in ("survival.csv", "1352.csv", "survival_info.csv")
    ] + [
        raw_dir.parent / name
        for name in ("survival.csv", "1352.csv", "survival_info.csv")
    ]
    label_path = None
    for c in candidates:
        if c.exists():
            label_path = c
            break
    if label_path is None:
        raise FileNotFoundError(
            "No survival label file found. Expected survival.csv, 1352.csv, or survival_info.csv "
            f"in {raw_dir} or its parent directory."
        )

    raw = pd.read_csv(label_path)
    cols = set(raw.columns)

    # BraTS format  ->  standard
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
        return out.set_index("MRN")

    # Already standard format
    survival = raw.copy()
    # force standard column names
    if "mrn" in survival.columns:
        survival = survival.rename(columns={"mrn": "MRN"})
    if "event" in survival.columns and "dead" not in survival.columns:
        survival = survival.rename(columns={"event": "dead"})
    if "time" in survival.columns and "deadtime" not in survival.columns:
        survival = survival.rename(columns={"time": "deadtime"})
    survival["MRN"] = survival["MRN"].astype(str)
    survival = survival.set_index("MRN")
    return survival[["dead", "deadtime"]]


# ---------------------------------------------------------------------------
#  Converters
# ---------------------------------------------------------------------------

def convert_to_unified(
    raw_dir: str | Path,
    output_dir: str | Path,
    source_type: str | None = None,
    case_id: str | None = None,
    max_cases: int | None = None,
) -> int:
    """Convert a raw dataset into the unified canonical format.

    Returns the number of cases converted.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_type is None:
        source_type = _detect_source_type(raw_dir)
    if source_type not in ("npc_nifti", "npc_npy", "brats"):
        raise ValueError(f"Unknown source_type: {source_type}")

    survival = _find_and_normalize_labels(raw_dir)
    survival.to_csv(output_dir / "survival.csv")

    case_dirs = _find_case_dirs(raw_dir)
    if case_id:
        case_dirs = [d for d in case_dirs if d.name == case_id]
    if max_cases:
        case_dirs = case_dirs[:max_cases]

    converted = 0
    for case_dir in tqdm(case_dirs, desc=f"Convert {source_type}", unit="case", ncols=80):
        cid = case_dir.name
        if cid not in survival.index:
            continue
        out_case = output_dir / cid
        out_case.mkdir(parents=True, exist_ok=True)
        try:
            _convert_case(case_dir, out_case, cid, source_type)
            converted += 1
        except Exception as e:
            print(f"[SKIP] {cid}: {e}")
            # Clean up partially created directory
            if out_case.exists():
                shutil.rmtree(out_case)

    print(f"Converted {converted} cases -> {output_dir}")
    return converted


def _convert_case(case_dir: Path, out_dir: Path, case_id: str, source_type: str) -> None:
    if source_type == "npc_nifti":
        _convert_npc_nifti_case(case_dir, out_dir, case_id)
    elif source_type == "npc_npy":
        _convert_npc_npy_case(case_dir, out_dir, case_id)
    elif source_type == "brats":
        _convert_brats_case(case_dir, out_dir, case_id)


# -- NPC NIfTI ---------------------------------------------------------------

def _convert_npc_nifti_case(case_dir: Path, out_dir: Path, case_id: str) -> None:
    mask_merged: np.ndarray | None = None
    for mod in ("t1", "t1c", "t2"):
        img_path = case_dir / f"{case_id}_{mod}_image.nii.gz"
        vol = nib.load(str(img_path)).get_fdata().astype(np.float32)
        np.save(out_dir / f"{mod}.npy", vol)

        mask_path = case_dir / f"{case_id}_{mod}_label1.nii.gz"
        if mask_path.exists():
            m = nib.load(str(mask_path)).get_fdata() > 0
            mask_merged = m if mask_merged is None else mask_merged | m

    _save_mask(out_dir, mask_merged, vol.shape)


# -- NPC NPY (symlink-only, no data reads) -----------------------------------

def _convert_npc_npy_case(case_dir: Path, out_dir: Path, case_id: str) -> None:
    # Verify all three image files exist before symlinking
    for mod in ("t1", "t1c", "t2"):
        img_path = case_dir / f"{case_id}_{mod}_image.npy"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing modality {mod}: {img_path}")

    for mod in ("t1", "t1c", "t2"):
        img_path = case_dir / f"{case_id}_{mod}_image.npy"
        dst = out_dir / f"{mod}.npy"
        if not dst.exists():
            os.symlink(os.path.relpath(img_path, out_dir), dst)

    # Symlink mask — use t1 label as tumour mask (all 3 modalities share same tumour region)
    mask_dst = out_dir / "mask.npy"
    if not mask_dst.exists():
        mask_src = case_dir / f"{case_id}_t1_label1.npy"
        if mask_src.exists():
            os.symlink(os.path.relpath(mask_src, out_dir), mask_dst)


# -- BraTS NIfTI -------------------------------------------------------------

_MOD_SOURCE_MAP_BRATS = {"t1": "t1", "t1c": "t1ce", "t2": "t2"}


def _convert_brats_case(case_dir: Path, out_dir: Path, case_id: str) -> None:
    ref_shape = None
    for target, source in _MOD_SOURCE_MAP_BRATS.items():
        candidates = sorted(case_dir.glob(f"{case_id}_{source}.nii*"))
        if not candidates:
            candidates = sorted(case_dir.glob(f"*_{source}.nii*"))
        if not candidates:
            raise FileNotFoundError(f"Missing {source} for {case_id}")
        vol = nib.load(str(candidates[0])).get_fdata().astype(np.float32)
        np.save(out_dir / f"{target}.npy", vol)
        ref_shape = vol.shape

    # shared segmentation mask
    seg_candidates = (
        sorted(case_dir.glob(f"{case_id}_seg.nii*"))
        or sorted(case_dir.glob("*_seg.nii*"))
        or sorted(case_dir.glob("*[Ss]eg.nii*"))
        or sorted(case_dir.glob("*[Ss]egm.nii*"))
    )
    mask: np.ndarray | None = None
    if seg_candidates:
        mask = nib.load(str(seg_candidates[0])).get_fdata() > 0
    _save_mask(out_dir, mask, ref_shape)


def _save_mask(out_dir: Path, mask: np.ndarray | None, ref_shape: tuple | None) -> None:
    if mask is not None:
        np.save(out_dir / "mask.npy", mask)
    elif ref_shape is not None:
        np.save(out_dir / "mask.npy", np.zeros(ref_shape, dtype=bool))


# ---------------------------------------------------------------------------
#  Unified reader (no per-dataset branches)
# ---------------------------------------------------------------------------

def load_case_unified(case_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Read a single case in unified canonical format.

    Returns:
        volumes:  {"t1": (D,H,W), "t1c": (D,H,W), "t2": (D,H,W)}  float32
        mask:     (D,H,W)  bool
    """
    volumes: dict[str, np.ndarray] = {}
    for mod in TARGET_MODALITIES:
        fpath = case_dir / f"{mod}.npy"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing modality file: {fpath}")
        arr = np.load(fpath)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        volumes[mod] = arr

    mask_path = case_dir / "mask.npy"
    if mask_path.exists():
        mask = np.load(mask_path)
        if mask.dtype != bool:
            mask = mask > 0
    else:
        mask = np.any([v > 0 for v in volumes.values()], axis=0)

    return volumes, mask


# ---------------------------------------------------------------------------
#  PyG Dataset for unified format
# ---------------------------------------------------------------------------

def _pad_or_truncate_6d(x: torch.Tensor, target_nodes: int) -> torch.Tensor:
    if x is None or x.numel() == 0:
        return torch.zeros((target_nodes, 6), dtype=torch.float32)
    current = x.size(0)
    if current >= target_nodes:
        return x[:target_nodes]
    padded = torch.zeros((target_nodes, x.size(1)), dtype=x.dtype)
    padded[:current] = x
    return padded


def _filter_edges(ei: torch.Tensor, max_nodes: int) -> torch.Tensor:
    if ei.numel() == 0:
        return ei
    mask = (ei[0] < max_nodes) & (ei[1] < max_nodes)
    return ei[:, mask]


class UnifiedDataset(Dataset):
    """Online 3D graph building from a unified-format directory.

    No per-dataset branching — all cases follow the same canonical layout.
    """

    def __init__(
        self,
        unified_dir: str | Path,
        num_nodes: int,
        n_segments: int = 200,
        roi_margin: int = 5,
        compactness: float = 0.1,
        sigma: float = 1.0,
        supervoxel_backend: str = "skimage",
        build_workers: int = 1,
        transform=None,
    ):
        super().__init__(transform=transform)
        self.unified_dir = Path(unified_dir)
        self.num_nodes = num_nodes
        self.n_segments = n_segments
        self.roi_margin = roi_margin
        self.compactness = compactness
        self.sigma = sigma
        self.supervoxel_backend = supervoxel_backend
        self.build_workers = max(1, int(build_workers))

        # Labels
        label_path = self.unified_dir / "survival.csv"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing survival.csv in {self.unified_dir}")
        survival = pd.read_csv(label_path)
        survival["MRN"] = survival["MRN"].astype(str)
        self.survival = survival.set_index("MRN")

        self.case_dirs = sorted(
            d for d in self.unified_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
            and d.name in self.survival.index
        )
        self._mem_cache: dict[int, Data] = {}

    def len(self) -> int:
        return len(self.case_dirs)

    def get(self, idx: int) -> Data:
        if idx in self._mem_cache:
            return self._mem_cache[idx]

        from ..preprocess.pipeline_3d import build_graphs_from_unified

        case_dir = self.case_dirs[idx]
        graphs = build_graphs_from_unified(
            case_dir=case_dir,
            n_segments=self.n_segments,
            roi_margin=self.roi_margin,
            compactness=self.compactness,
            sigma=self.sigma,
            supervoxel_backend=self.supervoxel_backend,
            num_nodes=self.num_nodes,
        )
        row = self.survival.loc[case_dir.name]
        data = Data(
            mrn=case_dir.name,
            num_nodes=self.num_nodes,
            dead=torch.tensor(int(row["dead"]), dtype=torch.int),
            deadtime=torch.tensor(float(row["deadtime"]), dtype=torch.float32),
        )
        for model_key, file_key in _MODALITY_FILE_MAP.items():
            g = graphs[file_key]
            x = _pad_or_truncate_6d(g.x, self.num_nodes)
            ei = _filter_edges(g.edge_index, self.num_nodes)
            data[f"{model_key}_x"] = x
            data[f"{model_key}_edge_index"] = ei

        self._mem_cache[idx] = data
        return data

    def warm_cache(self) -> None:
        if len(self._mem_cache) >= self.len():
            return
        from concurrent.futures import ProcessPoolExecutor

        args_list = [
            (case_dir, self.n_segments, self.roi_margin, self.compactness, self.sigma,
             self.supervoxel_backend, self.num_nodes)
            for case_dir in self.case_dirs
        ]
        if self.build_workers <= 1 or self.len() <= 1:
            for idx in tqdm(range(self.len()), desc="3D graph", unit="case", ncols=80):
                self.get(idx)
            return

        with ProcessPoolExecutor(max_workers=self.build_workers) as pool:
            results = list(tqdm(
                pool.map(_build_graph_worker, args_list),
                total=self.len(), desc="3D graph", unit="case", ncols=80,
            ))
            for idx, (case_dir, graphs) in enumerate(results):
                row = self.survival.loc[case_dir.name]
                data = Data(
                    mrn=case_dir.name,
                    num_nodes=self.num_nodes,
                    dead=torch.tensor(int(row["dead"]), dtype=torch.int),
                    deadtime=torch.tensor(float(row["deadtime"]), dtype=torch.float32),
                )
                for model_key, file_key in _MODALITY_FILE_MAP.items():
                    g = graphs[file_key]
                    x = _pad_or_truncate_6d(g.x, self.num_nodes)
                    ei = _filter_edges(g.edge_index, self.num_nodes)
                    data[f"{model_key}_x"] = x
                    data[f"{model_key}_edge_index"] = ei
                self._mem_cache[idx] = data


def _build_graph_worker(args):
    case_dir, n_segments, roi_margin, compactness, sigma, backend, num_nodes = args
    from ..preprocess.pipeline_3d import build_graphs_from_unified

    return case_dir, build_graphs_from_unified(
        case_dir=case_dir,
        n_segments=n_segments,
        roi_margin=roi_margin,
        compactness=compactness,
        sigma=sigma,
        supervoxel_backend=backend,
        num_nodes=num_nodes,
    )

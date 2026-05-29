from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.segmentation import slic
from torch_geometric.data import Data
from tqdm import tqdm

from ..dataio.unified import _find_case_dirs, load_case_unified, TARGET_MODALITIES

NODE_FEATURE_DIM = 6  # cz, cy, cx, voxel_count, mean, std


# ---- Public API ----

def run_3d_preprocessing(
    unified_dir: str | Path,
    output_dir: str | Path,
    n_segments: int = 200,
    num_nodes: int | None = None,
    roi_margin: int = 5,
    compactness: float = 0.1,
    sigma: float = 1.0,
    supervoxel_backend: str = "skimage",
    case_id: str | None = None,
    max_cases: int | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Offline 3D preprocessing from unified-format directory.

    Produces per-case per-modality .pt graph files, manifest.csv, and metadata.json.
    """
    unified_dir = Path(unified_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_nodes is None:
        num_nodes = n_segments

    (output_dir / "graphs").mkdir(parents=True, exist_ok=True)

    # Copy survival labels
    label_path = unified_dir / "survival.csv"
    if label_path.exists():
        import shutil
        shutil.copy2(label_path, output_dir / "survival.csv")

    case_dirs = _find_case_dirs(unified_dir)
    if case_id:
        case_dirs = [d for d in case_dirs if d.name == case_id]
    if max_cases:
        case_dirs = case_dirs[:max_cases]

    records: list[dict] = []

    for case_dir in tqdm(case_dirs, desc="Preprocessing 3D", unit="case", ncols=80):
        cid = case_dir.name
        # Check if all 3 modality graphs already exist
        all_exist = all(
            (output_dir / "graphs" / f"{cid}_{mod}.pt").exists()
            for mod in TARGET_MODALITIES
        )
        if all_exist and not overwrite:
            try:
                t1_data = torch.load(output_dir / "graphs" / f"{cid}_t1.pt", weights_only=False)
                records.append(_extract_record(cid, t1_data))
                continue
            except Exception:
                pass

        try:
            t_start = time.time()
            graphs = build_graphs_from_unified(
                case_dir=case_dir,
                n_segments=n_segments,
                roi_margin=roi_margin,
                compactness=compactness,
                sigma=sigma,
                supervoxel_backend=supervoxel_backend,
                num_nodes=num_nodes,
            )
            elapsed = time.time() - t_start
            for mod, graph_data in graphs.items():
                graph_data["elapsed"] = elapsed
                torch.save(graph_data, output_dir / "graphs" / f"{cid}_{mod}.pt")
            records.append(_extract_record(cid, graphs["t1"]))
        except Exception as e:
            print(f"[SKIP] {cid}: {e}")
            continue

    if not records:
        print("No cases processed.")
        return pd.DataFrame()

    summary = pd.DataFrame(records)
    manifest_path = output_dir / "manifest.csv"
    summary.to_csv(manifest_path, index=False)

    metadata = {
        "spatial_mode": "3d",
        "n_segments": n_segments,
        "num_nodes": num_nodes,
        "roi_margin": roi_margin,
        "compactness": compactness,
        "sigma": sigma,
        "supervoxel_backend": supervoxel_backend,
        "node_feature_dim": NODE_FEATURE_DIM,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Processed {len(records)} cases → {output_dir}")
    return summary


def _extract_record(case_id: str, data: Data | dict) -> dict:
    x = data.x if isinstance(data, Data) else data.get("x")
    return {
        "case_id": case_id,
        "num_nodes": x.size(0) if x is not None else 0,
        "active_nodes": (getattr(data, "active_nodes", 0) if isinstance(data, Data) else data.get("active_nodes", 0)),
        "roi_shape": "x".join(map(str, (getattr(data, "roi_shape", []) if isinstance(data, Data) else data.get("roi_shape", [])))),
        "elapsed": round(float(getattr(data, "elapsed", 0.0) if isinstance(data, Data) else data.get("elapsed", 0.0)), 2),
    }


# ---------------------------------------------------------------------------
#  Graph building from unified format (single entry point, no branching)
# ---------------------------------------------------------------------------

def build_graphs_from_unified(
    case_dir: Path,
    n_segments: int,
    roi_margin: int,
    compactness: float,
    sigma: float,
    supervoxel_backend: str,
    num_nodes: int,
) -> dict[str, Data]:
    """Build per-modality 3D graphs from a unified-format case directory.

    All modalities share the same supervoxel topology (single SLIC run on
    stacked volumes restricted to the joint tumour mask).

    Returns dict mapping modality name -> Data (with x, edge_index, active_nodes).
    """
    volumes, mask = load_case_unified(case_dir)

    # Align to minimum shape across modalities (safety — unified format should be pre-aligned)
    min_shape = tuple(min(v.shape[d] for v in volumes.values()) for d in range(3))
    volumes = {m: v[:min_shape[0], :min_shape[1], :min_shape[2]] for m, v in volumes.items()}
    mask = mask[:min_shape[0], :min_shape[1], :min_shape[2]]

    bbox = _compute_bbox_3d(mask, margin=roi_margin)
    volumes_cropped = {m: _crop_volume_3d(v, bbox) for m, v in volumes.items()}
    mask_cropped = _crop_volume_3d(mask, bbox)
    volumes_norm = {m: _normalize_volume(volumes_cropped[m]) for m in volumes}

    labels = _build_3d_supervoxels(
        volumes_norm, mask_cropped, n_segments=n_segments,
        compactness=compactness, sigma=sigma, backend=supervoxel_backend,
    )
    adjacency = _build_adjacency_3d(labels)

    graphs: dict[str, Data] = {}
    for mod in TARGET_MODALITIES:
        g = _build_graph_3d(volumes_norm[mod], labels, adjacency, num_nodes)
        g.roi_shape = list(mask_cropped.shape)
        g.original_shape = list(mask.shape)
        g.bbox = [(s.start, s.stop) for s in bbox]
        graphs[mod] = g

    return graphs


def _compute_bbox_3d(mask: np.ndarray, margin: int = 5) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2])
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    z_slice = slice(max(0, z_min - margin), min(mask.shape[0], z_max + margin + 1))
    y_slice = slice(max(0, y_min - margin), min(mask.shape[1], y_max + margin + 1))
    x_slice = slice(max(0, x_min - margin), min(mask.shape[2], x_max + margin + 1))
    return z_slice, y_slice, x_slice


def _crop_volume_3d(volume: np.ndarray, bbox: tuple[slice, slice, slice]) -> np.ndarray:
    return volume[bbox].copy()


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    valid = volume > 0
    if not valid.any():
        return np.zeros_like(volume, dtype=np.float32)
    low, high = np.percentile(volume[valid], [1, 99])
    if high <= low:
        return np.zeros_like(volume, dtype=np.float32)
    normalized = (volume - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _build_3d_supervoxels(
    volumes: dict[str, np.ndarray],
    mask: np.ndarray,
    n_segments: int,
    compactness: float,
    sigma: float,
    backend: str,
) -> np.ndarray:
    """Run 3D SLIC supervoxel segmentation inside the ROI."""
    modalities = ("t1", "t1c", "t2")
    multi_channel = np.stack([volumes[m] for m in modalities], axis=-1).astype(np.float32, copy=False)
    if backend == "skimage":
        slic_kwargs = dict(
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=1,
            channel_axis=-1,
            enforce_connectivity=False,
        )
        if mask.any():
            slic_kwargs["mask"] = mask
        labels = slic(multi_channel, **slic_kwargs)
    else:
        raise ValueError(f"Unsupported 3D supervoxel backend: {backend}")
    if mask.any():
        labels[~mask] = 0
    return labels.astype(np.int64)


def _build_adjacency_3d(labels: np.ndarray) -> list[tuple[int, int]]:
    """Build 3D adjacency from voxel contacts (6-connectivity)."""
    edge_parts: list[np.ndarray] = []

    def _collect(a: np.ndarray, b: np.ndarray) -> None:
        boundary = (a != b) & (a > 0) & (b > 0)
        if not boundary.any():
            return
        pair_arr = np.stack((a[boundary] - 1, b[boundary] - 1), axis=1)
        pair_arr.sort(axis=1)
        edge_parts.append(np.unique(pair_arr, axis=0))

    if labels.shape[0] > 1:
        _collect(labels[:-1, :, :], labels[1:, :, :])
    if labels.shape[1] > 1:
        _collect(labels[:, :-1, :], labels[:, 1:, :])
    if labels.shape[2] > 1:
        _collect(labels[:, :, :-1], labels[:, :, 1:])

    if not edge_parts:
        return []
    edges = np.unique(np.concatenate(edge_parts, axis=0), axis=0)
    return [tuple(map(int, row)) for row in edges]


def _build_graph_3d(
    volume: np.ndarray,
    labels: np.ndarray,
    adjacency: list[tuple[int, int]],
    num_nodes: int,
) -> Data:
    """Build a 3D graph with 6-dim node features.

    Features: cz, cy, cx, voxel_count, mean, std
    """
    active = np.unique(labels[labels > 0]).astype(np.int64)
    x = np.zeros((num_nodes, NODE_FEATURE_DIM), dtype=np.float32)
    active_count = min(active.size, num_nodes)

    if active_count > 0:
        valid_mask = labels > 0
        flat_labels = labels[valid_mask].astype(np.int64, copy=False)
        label_ids = active[:active_count]
        remapped = np.searchsorted(label_ids, flat_labels)

        keep = remapped < active_count
        remapped = remapped[keep]

        voxel_count = np.bincount(remapped, minlength=active_count).astype(np.float32)
        coords = np.argwhere(valid_mask).astype(np.float32, copy=False)[keep]
        coord_sums = np.vstack([
            np.bincount(remapped, weights=coords[:, dim], minlength=active_count)
            for dim in range(3)
        ]).T
        centroids = coord_sums / np.maximum(voxel_count[:, None], 1.0)

        values = volume[valid_mask].astype(np.float32, copy=False)[keep]
        sum_vals = np.bincount(remapped, weights=values, minlength=active_count).astype(np.float32)
        sum_sq = np.bincount(remapped, weights=values * values, minlength=active_count).astype(np.float32)
        means = sum_vals / np.maximum(voxel_count, 1.0)
        variances = np.maximum(sum_sq / np.maximum(voxel_count, 1.0) - means * means, 0.0)

        features = np.zeros((active_count, NODE_FEATURE_DIM), dtype=np.float32)
        features[:, :3] = centroids
        features[:, 3] = voxel_count
        features[:, 4] = means
        features[:, 5] = np.sqrt(variances)
        x[:active_count] = features

    edge_pairs = [(s, t) for s, t in adjacency if s < active_count and t < active_count]
    edge_pairs.extend((i, i) for i in range(active_count, num_nodes))
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=torch.from_numpy(x), edge_index=edge_index)
    data.active_nodes = active_count
    return data



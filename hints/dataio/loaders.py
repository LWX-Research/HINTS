from __future__ import annotations

from torch.utils.data import Subset
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from ..settings import Settings


def create_data_loaders_3d(
    dataset: Dataset,
    train_idx,
    val_idx,
    settings: Settings,
) -> tuple[DataLoader, DataLoader]:
    common_kwargs = {
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "pin_memory": settings.pin_memory,
        "drop_last": False,
        "persistent_workers": settings.num_workers > 0,
    }
    train_loader = DataLoader(
        Subset(dataset, train_idx), shuffle=True, **common_kwargs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), shuffle=False, **common_kwargs,
    )
    return train_loader, val_loader

from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_environment() -> None:
    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(env_name) in {None, "", "0"}:
            os.environ[env_name] = "1"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

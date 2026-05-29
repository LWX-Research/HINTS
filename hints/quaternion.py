from __future__ import annotations

import torch
import torch.nn as nn


class QuaternionLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape * 4, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        return self.norm(x.reshape(*x.shape[:-2], -1)).reshape(original_shape)


class QuaternionLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, normalize: str = "both", eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim, 4))
        if normalize in {"component", "both"}:
            self.comp_norm = QuaternionLayerNorm(out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.weight)
        if self.normalize in {"magnitude", "both"}:
            with torch.no_grad():
                self.weight.data = self.magnitude_normalize(self.weight.data)

    def magnitude_normalize(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=-1, keepdim=True) + self.eps
        return x / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-3)
        weight_expanded = self.weight.unsqueeze(0)
        products = torch.stack(
            [
                x_expanded[..., 0] * weight_expanded[..., 0] - x_expanded[..., 1] * weight_expanded[..., 1] - x_expanded[..., 2] * weight_expanded[..., 2] - x_expanded[..., 3] * weight_expanded[..., 3],
                x_expanded[..., 0] * weight_expanded[..., 1] + x_expanded[..., 1] * weight_expanded[..., 0] + x_expanded[..., 2] * weight_expanded[..., 3] - x_expanded[..., 3] * weight_expanded[..., 2],
                x_expanded[..., 0] * weight_expanded[..., 2] - x_expanded[..., 1] * weight_expanded[..., 3] + x_expanded[..., 2] * weight_expanded[..., 0] + x_expanded[..., 3] * weight_expanded[..., 1],
                x_expanded[..., 0] * weight_expanded[..., 3] + x_expanded[..., 1] * weight_expanded[..., 2] - x_expanded[..., 2] * weight_expanded[..., 1] + x_expanded[..., 3] * weight_expanded[..., 0],
            ],
            dim=-1,
        )
        return torch.sum(products, dim=-2)


def construct_hamilton_matrix(q: torch.Tensor) -> torch.Tensor:
    batch_size, prototype_count, hidden_dim, _ = q.shape
    w, x, y, z = q.unbind(-1)
    matrix = torch.zeros((batch_size, 2 * prototype_count, 2 * hidden_dim), dtype=q.dtype, device=q.device)
    matrix[:, 0::2, 0::2] = w
    matrix[:, 0::2, 1::2] = x
    matrix[:, 1::2, 0::2] = -y
    matrix[:, 1::2, 1::2] = z
    return matrix


def qsvd_reconstruct(matrix: torch.Tensor, rank_q: int = 2) -> torch.Tensor:
    try:
        if matrix.is_cuda:
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False, driver="gesvd")
        else:
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    except torch.linalg.LinAlgError:
        if not matrix.is_cuda:
            raise
        u, s, vh = torch.linalg.svd(matrix.cpu(), full_matrices=False)
        u = u.to(matrix.device)
        s = s.to(matrix.device)
        vh = vh.to(matrix.device)

    rank_r = 2 * rank_q
    s_trunc = s[..., :rank_r]
    u_trunc = u[..., :rank_r]
    vh_trunc = vh[..., :rank_r, :]
    return u_trunc @ (s_trunc.unsqueeze(-1) * vh_trunc)


def hamilton_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    w = matrix[:, 0::2, 0::2]
    x = matrix[:, 0::2, 1::2]
    y = -matrix[:, 1::2, 0::2]
    z = matrix[:, 1::2, 1::2]
    quaternion = torch.stack([w, x, y, z], dim=-1)
    return quaternion / torch.norm(quaternion, dim=-1, keepdim=True)


def qsvd_reconstruction(quaternion: torch.Tensor, rank_q: int = 1) -> torch.Tensor:
    matrix = construct_hamilton_matrix(quaternion)
    reconstructed = qsvd_reconstruct(matrix, rank_q)
    return hamilton_to_quaternion(reconstructed)

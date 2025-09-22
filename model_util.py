import torch
import torch.nn as nn
import torch.nn.functional as F

class QuaternionLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape * 4, eps=eps, elementwise_affine=affine)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(*x.shape[:-2], -1)
        x = self.norm(x)
        return x.reshape(orig_shape)

class QuaternionBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features * 4, eps=eps, affine=affine)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x = self.bn(x)
        return x.reshape(orig_shape)

class QuaternionLinear(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 normalize: str = "both",  # "none", "magnitude", "component", "both"
                 norm_type: str = "layer",  # "layer" or "batch"
                 eps: float = 1e-8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize = normalize
        self.eps = eps
        
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim, 4))
        
        if normalize in ["component", "both"]:
            if norm_type == "layer":
                self.comp_norm = QuaternionLayerNorm(out_dim)
            elif norm_type == "batch":
                self.comp_norm = QuaternionBatchNorm(out_dim)
            else:
                raise ValueError(f"Unsupported norm_type: {norm_type}")
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        
        if self.normalize in ["magnitude", "both"]:
            with torch.no_grad():
                self.weight.data = self.magnitude_normalize(self.weight.data)

    def magnitude_normalize(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=-1, keepdim=True) + self.eps
        return x / norm

    def component_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.comp_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-3)
        weight_expanded = self.weight.unsqueeze(0)
        products = torch.stack([
            x_expanded[..., 0]*weight_expanded[..., 0] - x_expanded[..., 1]*weight_expanded[..., 1] - x_expanded[..., 2]*weight_expanded[..., 2] - x_expanded[..., 3]*weight_expanded[..., 3],
            x_expanded[..., 0]*weight_expanded[..., 1] + x_expanded[..., 1]*weight_expanded[..., 0] + x_expanded[..., 2]*weight_expanded[..., 3] - x_expanded[..., 3]*weight_expanded[..., 2],
            x_expanded[..., 0]*weight_expanded[..., 2] - x_expanded[..., 1]*weight_expanded[..., 3] + x_expanded[..., 2]*weight_expanded[..., 0] + x_expanded[..., 3]*weight_expanded[..., 1],
            x_expanded[..., 0]*weight_expanded[..., 3] + x_expanded[..., 1]*weight_expanded[..., 2] - x_expanded[..., 2]*weight_expanded[..., 1] + x_expanded[..., 3]*weight_expanded[..., 0]
        ], dim=-1)
        output = torch.sum(products, dim=-2)
        
        return output


def construct_hamilton_matrix(q: torch.Tensor) -> torch.Tensor:
    B, P, D, _ = q.shape
    w, x, y, z = q.unbind(-1)
    A = torch.zeros((B, 2*P, 2*D), dtype=q.dtype, device=q.device)
    A[:, 0::2, 0::2] = w
    A[:, 0::2, 1::2] = x
    A[:, 1::2, 0::2] = -y
    A[:, 1::2, 1::2] = z
    return A

def qsvd_reconstruct(A: torch.Tensor, rank_q: int = 2) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    rank_r = 2 * rank_q
    S_trunc = S[..., :rank_r]
    U_trunc = U[..., :rank_r]
    Vh_trunc = Vh[..., :rank_r, :]
    A_recon = U_trunc @ (S_trunc.unsqueeze(-1) * Vh_trunc)
    return A_recon

def hamilton_to_quaternion(A: torch.Tensor) -> torch.Tensor:
    w = A[:, 0::2, 0::2]
    x = A[:, 0::2, 1::2]
    y = -A[:, 1::2, 0::2]
    z = A[:, 1::2, 1::2]
    q = torch.stack([w, x, y, z], dim=-1)
    return q / torch.norm(q, dim=-1, keepdim=True)

def qsvd_reconstruction(q: torch.Tensor, rank_q: int = 1) -> torch.Tensor:
    A = construct_hamilton_matrix(q)
    A_recon = qsvd_reconstruct(A, rank_q)
    q_recon = hamilton_to_quaternion(A_recon)
    return q_recon
                
def qsvd_main_left_singular(q: torch.Tensor) -> torch.Tensor:
    B, P, D, _ = q.shape
    A = construct_hamilton_matrix(q)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    assert U.shape == (B, 2*P, 2*P), f"Invalid U shape: {U.shape} vs {(B, 2*P, 2*P)}"
    u_vectors = U[..., :2]
    flattened = u_vectors.reshape(B, 2*P*2)
    quat_vectors = flattened.reshape(B, P, 4)

    return quat_vectors

# utils/propagation_utils.py
import torch
from typing import Tuple

@torch.no_grad()
def compute_lag_spectrum(attn: torch.Tensor, rel_t_index: torch.Tensor, T_w: int, reduce_heads: bool=True) -> torch.Tensor:
    # attn: [B_win, H, N, N], rel_t_index: [N, N] in [0..2*T_w-2]
    B, H, N, _ = attn.shape
    bins = 2*T_w - 1
    idx = rel_t_index.view(-1)                     # [N*N]
    weights = attn.view(B, H, -1)                  # [B,H,N*N]
    hist = torch.zeros(B, H, bins, device=attn.device)
    for b in range(B):
        for h in range(H):
            hist[b, h] = torch.bincount(idx, weights=weights[b, h], minlength=bins)
    p_delta = hist / (hist.sum(dim=-1, keepdim=True) + 1e-9)
    return p_delta.mean(dim=1) if reduce_heads else p_delta  # [B,bins] or [B,H,bins]

@torch.no_grad()
def compute_direction_field(attn: torch.Tensor, rel_t_index: torch.Tensor, rel_xyz: torch.Tensor, positive_only: bool=True) -> torch.Tensor:
    # 返回窗口级方向向量场: [B_win, H, N, 3]
    B, H, N, _ = attn.shape
    center = rel_t_index.max() // 2
    mask_pos = (rel_t_index > center) if positive_only else torch.ones_like(rel_t_index, dtype=torch.bool)
    mask_pos = mask_pos.unsqueeze(0).unsqueeze(0)       # [1,1,N,N]
    disp = rel_xyz.float()                               # [3,N,N]
    dist = torch.linalg.norm(disp, dim=0) + 1e-8        # [N,N]
    unit = (disp / dist).unsqueeze(0).unsqueeze(0)      # [1,1,3,N,N]
    w = attn * mask_pos                                  # [B,H,N,N]
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
    vec = (w.unsqueeze(2) * unit).sum(dim=-1)            # [B,H,3,N]
    return vec.permute(0, 1, 3, 2).contiguous()          # [B,H,N,3]

@torch.no_grad()
def compute_speed_map(attn: torch.Tensor, rel_t_index: torch.Tensor, rel_xyz: torch.Tensor,
                      voxel_spacing: Tuple[float,float,float], TR: float,
                      positive_only: bool=True, reduce_heads: bool=True) -> torch.Tensor:
    # 返回窗口级速度: [B_win, H, N] 或 [B_win, N]
    B, H, N, _ = attn.shape
    center = rel_t_index.max() // 2
    delta = (rel_t_index - center).float()
    valid = (delta > 0).float() if positive_only else (delta.abs() > 0).float()

    sd, sh, sw = voxel_spacing
    disp = rel_xyz.float()                                 # [3,N,N]
    disp_mm = torch.stack([disp[0]*sd, disp[1]*sh, disp[2]*sw], dim=0)  # [3,N,N]
    dist_mm = torch.linalg.norm(disp_mm, dim=0)                                 # [N,N]
    time_s = (delta.abs() + 1e-9) * TR                                         # [N,N]
    base = (dist_mm / time_s) * valid                                          # [N,N]
    base = base.unsqueeze(0).unsqueeze(0)                                      # [1,1,N,N]

    w = attn * valid.unsqueeze(0).unsqueeze(0)                                 # [B,H,N,N]
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
    speed = (w * base).sum(dim=-1)                                             # [B,H,N]
    return speed.mean(dim=1) if reduce_heads else speed                        # [B,N] / [B,H,N]
"""
Our code is based on the following code.
https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html#SwinUNETR
"""

import itertools
import os
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp

from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

from .patchembedding import PatchEmbed
from typing import Optional, Dict

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "window_partition",
    "window_reverse",
    "WindowAttention4D",
    "SwinTransformerBlock4D",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer4D",
]

import torch
import torch.nn.functional as F

def window_partition(x, window_size):
    """
    x: [B, Dp, Hp, Wp, Tp, C]  (已经 pad/shift 处理之后的特征)
    window_size: (Wd, Wh, Ww, Wt)

    返回:
      x_windows:  [B*nW, N, C]   (N = Wd*Wh*Ww*Wt)
      win_starts: [B*nW, 3]      每个窗口在 pad 后全局坐标系中的 (d,h,w) 起点 (体素索引)
      rel_offsets:[N, 3]         窗口内每个 token 相对于起点的 (d,h,w) 体素偏移
    """
    Wd, Wh, Ww, Wt = window_size
    B, Dp, Hp, Wp, Tp, C = x.shape
    assert Dp % Wd == 0 and Hp % Wh == 0 and Wp % Ww == 0 and Tp % Wt == 0, "input dims must be divisible"

    # 计算每个维度的窗口个数
    nD = Dp // Wd
    nH = Hp // Wh
    nW = Wp // Ww
    nT = Tp // Wt
    nWin = nD * nH * nW * nT

    # 把 x 切成窗口格
    x_reshaped = x.view(B,
                        nD, Wd,
                        nH, Wh,
                        nW, Ww,
                        nT, Wt,
                        C)
    # 重排让窗口内部 token 连成一维
    # 顺序与 _build_rel_indices 中 meshgrid(indexing="ij") 保持一致: (d,h,w,t)
    x_windows = x_reshaped.permute(0, 1, 3, 5, 7,     # B, nD, nH, nW, nT,
                                   2, 4, 6, 8,       #    Wd, Wh, Ww, Wt,
                                   9).contiguous()   #    C
    # 现在把 (nD,nH,nW,nT) 合到 batch 维
    x_windows = x_windows.view(B * nWin, Wd * Wh * Ww * Wt, C)  # [B*nWin, N, C]

    # --- win_starts: 每个窗口的 (d,h,w) 起点(体素索引) ---
    d_starts = torch.arange(0, Dp, Wd, device=x.device)
    h_starts = torch.arange(0, Hp, Wh, device=x.device)
    w_starts = torch.arange(0, Wp, Ww, device=x.device)
    t_starts = torch.arange(0, Tp, Wt, device=x.device)  # 只用来枚举窗口，不写入 win_starts

    grid = torch.stack(torch.meshgrid(d_starts, h_starts, w_starts, t_starts, indexing="ij"), dim=0)  # [4, nD, nH, nW, nT]
    grid = grid.view(4, -1).t()  # [nWin, 4] = (d0,h0,w0,t0)
    win_starts = grid[:, :3]     # [nWin, 3] = (d0,h0,w0)
    # 扩成 [B*nWin, 3]
    win_starts = win_starts.unsqueeze(0).expand(B, nWin, 3).reshape(B * nWin, 3).contiguous()

    # --- rel_offsets: 窗口内每个 token 的 (d,h,w) 相对偏移 ---
    offs_d = torch.arange(Wd, device=x.device)
    offs_h = torch.arange(Wh, device=x.device)
    offs_w = torch.arange(Ww, device=x.device)
    offs_t = torch.arange(Wt, device=x.device)
    rel = torch.stack(torch.meshgrid(offs_d, offs_h, offs_w, offs_t, indexing="ij"), dim=0)  # [4,Wd,Wh,Ww,Wt]
    rel = rel.view(4, -1).t()  # [N, 4] = (d,h,w,t)
    rel_offsets = rel[:, :3].contiguous()  # [N, 3]

    return x_windows, win_starts, rel_offsets


def window_reverse(windows, window_size, dims):
    """
    反切窗，把 (B*nWin, N, C) 还原成 (B, Dp, Hp, Wp, Tp, C)
    window_size: (Wd, Wh, Ww, Wt)
    dims:        [B, Dp, Hp, Wp, Tp]
    """
    Wd, Wh, Ww, Wt = window_size
    B, Dp, Hp, Wp, Tp = dims
    nd, nh, nw, nt = Dp // Wd, Hp // Wh, Wp // Ww, Tp // Wt
    N = Wd * Wh * Ww * Wt
    C = windows.shape[-1]

    x = windows.view(B, nd, nh, nw, nt, Wd, Wh, Ww, Wt, C) \
             .permute(0, 1, 5, 2, 6, 3, 7, 4, 8, 9) \
             .contiguous() \
             .view(B, Dp, Hp, Wp, Tp, C)
    return x


'''
def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Partition tokens into their respective windows

     Args:
        x: input tensor (B, D, H, W, T, C)

        window_size: local window size.


    Returns:
        windows: (B*num_windows, window_size*window_size*window_size*window_size, C)
    """
    x_shape = x.size()

    b, d, h, w, t, c = x_shape
    x = x.view(
        b,
        d // window_size[0],  # number of windows in depth dimension
        window_size[0],  # window size in depth dimension
        h // window_size[1],  # number of windows in height dimension
        window_size[1],  # window size in height dimension
        w // window_size[2],  # number of windows in width dimension
        window_size[2],  # window size in width dimension
        t // window_size[3],  # number of windows in time dimension
        window_size[3],  # window size in time dimension
        c,
    )
    windows = (
        x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8, 9)
        .contiguous()
        .view(-1, window_size[0] * window_size[1] * window_size[2] * window_size[3], c)
    )
    return windows

def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor (B*num_windows, window_size, window_size, C)
        window_size: local window size.
        dims: dimension values.

    Returns:
        x: (B, D, H, W, T, C)
    """

    b, d, h, w, t = dims
    x = windows.view(
        b,
        torch.div(d, window_size[0], rounding_mode="floor"),
        torch.div(h, window_size[1], rounding_mode="floor"),
        torch.div(w, window_size[2], rounding_mode="floor"),
        torch.div(t, window_size[3], rounding_mode="floor"),
        window_size[0],
        window_size[1],
        window_size[2],
        window_size[3],
        -1,
    )
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4, 8, 9).contiguous().view(b, d, h, w, t, -1)

    return x
'''

def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class WindowAttention4D(nn.Module):
    """
    4D 窗口注意力（Swin），加入“相对时间偏置”；相对索引按当前窗口动态构建，避免形状不匹配。
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # 仅作默认值；真正以 forward(meta) 为准
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # 运行期开关/缓存（评估可取出注意力）
        self.return_attn: bool = False
        self.last_attn = None
        self.last_meta = None

        # 动态缓存（避免重复构造）
        self._cached_key = None            # (Dw,Hw,Ww,Tw)
        self._rel_t_index = None           # [N,N]
        self._rel_xyz = None               # [3,N,N]
        # self._time_bias_table = None       # nn.Parameter(2*Tw-1, heads)

        self._time_bias_table = nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        self.register_parameter("_time_bias_table", self._time_bias_table)

        self.last_win_starts = None  # [B_win, 3]  每个窗口左上角(或起点)的全局体素索引 (d,h,w)
        self.last_rel_offsets = None  # [N, 3]     窗口内token的相对偏移 (d,h,w)

    def _build_rel_indices(self, Dw, Hw, Ww, Tw, device, dtype):
        """
        与 window_partition 一致的展平顺序 (D,H,W,T) 来构造：
          - relative_time_index: [N,N] in [0 .. 2*Tw-2]
          - relative_xyz: [3,N,N]，分别是 Δx, Δy, Δz（体素单位）
        """
        coords_d = torch.arange(Dw, device=device)
        coords_h = torch.arange(Hw, device=device)
        coords_w = torch.arange(Ww, device=device)
        coords_t = torch.arange(Tw, device=device)
        grid = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, coords_t, indexing="ij"), dim=0)  # [4,D,H,W,T]
        grid = grid.view(4, -1)  # [4, N]，顺序与 window_partition 一致
        d_coords, h_coords, w_coords, t_coords = grid[0], grid[1], grid[2], grid[3]
        N = grid.shape[1]

        # 时间相对索引
        rel_t = t_coords[:, None] - t_coords[None, :]                 # [N,N] ∈ [-(Tw-1) .. +(Tw-1)]
        rel_t_index = (rel_t + (Tw - 1)).to(torch.long)               # shift 到 [0 .. 2*Tw-2]

        # 空间相对位移（体素单位，后续可结合 voxel_spacing 转换到 mm）
        rel_d = (d_coords[:, None] - d_coords[None, :]).to(dtype)     # [N,N]
        rel_h = (h_coords[:, None] - h_coords[None, :]).to(dtype)
        rel_w = (w_coords[:, None] - w_coords[None, :]).to(dtype)
        rel_xyz = torch.stack([rel_w, rel_h, rel_d], dim=0)           # [3,N,N]，对应 x(宽), y(高), z(深)

        return rel_t_index, rel_xyz

    def forward(self, x, mask, meta: Optional[dict] = None):
        """
        x:    (B_win, N, C)
        mask: (num_windows, N, N) 或 None
        meta: 必须包含 {"window_size": (Dw,Hw,Ww,Tw), "dims": [B,Dp,Hp,Wp,Tp]}
        """
        if meta is None or "window_size" not in meta:
            raise RuntimeError("WindowAttention4D.forward 需要 meta['window_size']")

        Dw, Hw, Ww, Tw = meta["window_size"]
        b_, n, c = x.shape

        # qkv & 注意力分数
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_win, heads, N, N)

        # --- 动态/缓存的相对索引 ---
        key = (Dw, Hw, Ww, Tw)
        if self._cached_key != key or self._rel_t_index is None or self._rel_t_index.shape[0] != n:
            self._rel_t_index, self._rel_xyz = self._build_rel_indices(Dw, Hw, Ww, Tw, attn.device, attn.dtype)
            self._cached_key = key

        # --- 可学习的时间偏置表（依赖 Tw 与 heads）---
        # if (self._time_bias_table is None) or (self._time_bias_table.shape[0] != (2 * Tw - 1)):
        #     self._time_bias_table = nn.Parameter(
        #         torch.zeros(2 * Tw - 1, self.num_heads, device=attn.device, dtype=attn.dtype),
        #         requires_grad=True
        #     )
        #     trunc_normal_(self._time_bias_table, std=0.02)

        if (self._time_bias_table.shape[0] != (2 * Tw - 1)) or (not self._time_bias_table.requires_grad):
            self._time_bias_table = nn.Parameter(
                torch.zeros(2 * Tw - 1, self.num_heads, device=attn.device, dtype=attn.dtype),
                requires_grad=True
            )
            trunc_normal_(self._time_bias_table, std=0.02)
            self.register_parameter("_time_bias_table", self._time_bias_table)

        tb = self._time_bias_table[self._rel_t_index.view(-1)]         # [(N*N), heads]
        time_bias = tb.view(n, n, self.num_heads).permute(2, 0, 1).unsqueeze(0)  # [1, heads, N, N]
        attn = attn + time_bias

        # 窗口注意力 mask
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.to(attn.dtype).unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # 仅评估/可视化时缓存注意力 & 元数据 & 相对索引
        # 仅评估/可视化时缓存注意力 & 元数据 & 相对索引
        # 仅评估/可视化时缓存注意力 & 元数据 & 相对索引 & 全局坐标
        # 仅评估/可视化时缓存注意力 & 元数据 & 相对索引 & 全局坐标
        if self.return_attn:
            # 1) 缓存注意力与基本 meta
            self.last_attn = attn.detach()
            self.last_meta = {
                "window_size": (Dw, Hw, Ww, Tw),
                "dims": meta.get("dims", None),  # [B, Dp, Hp, Wp, Tp]
            }
            # 2) 缓存相对索引
            self.relative_time_index = self._rel_t_index.detach()  # [N, N]
            self.relative_xyz = self._rel_xyz.detach()  # [3, N, N] (x,y,z)=(w,h,d)

            # 3) 先从 meta 里取全局坐标信息并缓存到模块
            win_starts = meta.get("win_starts", None)  # [B_win, 3]  (d,h,w)
            rel_offsets = meta.get("rel_offsets", None)  # [N, 3]      (d,h,w)
            self.last_win_starts = None
            self.last_rel_offsets = None
            if win_starts is not None and rel_offsets is not None:
                try:
                    ws = torch.as_tensor(win_starts, device=attn.device)
                    ro = torch.as_tensor(rel_offsets, device=attn.device)
                    assert ws.ndim == 2 and ws.shape[
                        1] == 3, f"win_starts shape should be [B_win,3], got {tuple(ws.shape)}"
                    assert ro.ndim == 2 and ro.shape[
                        1] == 3, f"rel_offsets shape should be [N,3], got {tuple(ro.shape)}"
                    self.last_win_starts = ws.detach()
                    self.last_rel_offsets = ro.detach()
                except Exception as e:
                    if os.environ.get("SWIN4D_VERBOSE", "0") == "1":
                        print(f"[WARN] meta win_starts/rel_offsets invalid: {e}")

            # 4) 现在再调试打印（放在赋值之后才有意义）
            # if self.last_win_starts is None:
            #     print("[DEBUG] WindowAttention4D: last_win_starts is None (meta didn't carry it)")
            # else:
            #     try:
            #         mn = self.last_win_starts.min(dim=0).values.tolist()
            #         mx = self.last_win_starts.max(dim=0).values.tolist()
            #         print(f"[DEBUG] cached last_win_starts min/max(d,h,w): {mn} .. {mx}")
            #     except Exception:
            #         pass

        # 加权求值并投影
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



'''
class WindowAttention4D(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

       
    def forward(self, x, mask):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.to(attn.dtype).unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

'''

class SwinTransformerBlock4D(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention4D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix):
        b, d, h, w, t, c = x.shape
        window_size, shift_size = get_window_size((d, h, w, t), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_d0 = pad_h0 = pad_w0 = pad_t0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_h1 = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_w1 = (window_size[2] - w % window_size[2]) % window_size[2]
        pad_t1 = (window_size[3] - t % window_size[3]) % window_size[3]
        x = F.pad(x, (0, 0, pad_t0, pad_t1, pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1))  # last tuple first in
        _, dp, hp, wp, tp, _ = x.shape
        dims = [b, dp, hp, wp, tp]
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2], -shift_size[3]), dims=(1, 2, 3, 4)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # x_windows = window_partition(shifted_x, window_size)
        # #attn_windows = self.attn(x_windows, mask=attn_mask)
        # attn_meta = {'window_size': window_size, 'dims': dims}  # dims = [B, Dp, Hp, Wp, Tp]
        # attn_windows = self.attn(x_windows, mask=attn_mask, meta=attn_meta)

        # 1) 切窗：拿到 token + 每个窗口的全局起点 + 窗口内相对偏移(体素, d,h,w)
        x_windows, win_starts, rel_offsets = window_partition(
            shifted_x, window_size
        )  # (B*nWin, N, C), (B*nWin,3), (N,3)

        # if not hasattr(self, "_debug_printed"):
        #     self._debug_printed = True
        #     try:
        #         print(f"[DEBUG] win_starts min/max (d,h,w): "
        #               f"{win_starts[:, 0].min().item()}..{win_starts[:, 0].max().item()}, "
        #               f"{win_starts[:, 1].min().item()}..{win_starts[:, 1].max().item()}, "
        #               f"{win_starts[:, 2].min().item()}..{win_starts[:, 2].max().item()}")
        #         print(f"[DEBUG] rel_offsets unique per axis: "
        #               f"d={torch.unique(rel_offsets[:, 0])[:8]}, "
        #               f"h={torch.unique(rel_offsets[:, 1])[:8]}, "
        #               f"w={torch.unique(rel_offsets[:, 2])[:8]}")
        #     except Exception as e:
        #         print("[DEBUG] print win_starts/rel_offsets failed:", e)

        # 2) 若使用了 shift，把 roll 的位移“加回去”（映射回 pad 后的全局坐标系）
        #    dims = [B, dp, hp, wp, tp]
        if any(i > 0 for i in shift_size):
            dp, hp, wp, tp = dims[1], dims[2], dims[3], dims[4]
            win_starts = win_starts.clone()
            win_starts[:, 0] = (win_starts[:, 0] + shift_size[0]) % dp  # d
            win_starts[:, 1] = (win_starts[:, 1] + shift_size[1]) % hp  # h
            win_starts[:, 2] = (win_starts[:, 2] + shift_size[2]) % wp  # w

        # 3) 组装 meta 并传给注意力
        attn_meta = {
            'window_size': window_size,  # (Wd,Wh,Ww,Tw)
            'dims': dims,  # [B, Dp, Hp, Wp, Tp] (pad 后)
            'win_starts': win_starts,  # (B*nWin, 3) 全局起点(体素, d,h,w)
            'rel_offsets': rel_offsets,  # (N, 3)     窗口内相对偏移(体素, d,h,w)
        }
        attn_windows = self.attn(x_windows, mask=attn_mask, meta=attn_meta)

        # 4) 反切回去
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2], shift_size[3]), dims=(1, 2, 3, 4)
            )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_h1 > 0 or pad_w1 > 0 or pad_t1 > 0:
            x = x[:, :d, :h, :w, :t, :].contiguous()

        return x

    def forward_part2(self, x):
        x = self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3, c_multiplier: int = 2
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim

        # Skip dimension reduction on the temporal dimension

        self.reduction = nn.Linear(8 * dim, c_multiplier * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        x_shape = x.size()
        b, d, h, w, t, c = x_shape
        x = torch.cat(
            [x[:, i::2, j::2, k::2, :, :] for i, j, k in itertools.product(range(2), range(2), range(2))],
            -1,
        )

        x = self.norm(x)
        x = self.reduction(x)

        return x


MERGING_MODE = {"mergingv2": PatchMergingV2}


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    d, h, w, t = dims
    img_mask = torch.zeros((1, d, h, w, t, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                for t in slice(-window_size[3]), slice(-window_size[3], -shift_size[3]), slice(-shift_size[3], None):
                    img_mask[:, d, h, w, t, :] = cnt
                    cnt += 1

    # mask_windows = window_partition(img_mask, window_size)
    # mask_windows = mask_windows.squeeze(-1)

    mask_windows, _, _ = window_partition(img_mask, window_size)  # 只用第一个返回值
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        c_multiplier: int = 2,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock4D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size), c_multiplier=c_multiplier
            )

    def forward(self, x):
        b, c, d, h, w, t = x.size()
        window_size, shift_size = get_window_size((d, h, w, t), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w t -> b d h w t c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        tp = int(np.ceil(t / window_size[3])) * window_size[3]
        attn_mask = compute_mask([dp, hp, wp, tp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, t, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w t c -> b c d h w t")

        return x


# Basic layer for full attention,
# the only difference is that there is no window shifting
class BasicLayer_FullAttention(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        c_multiplier: int = 2,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock4D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.no_shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size), c_multiplier=c_multiplier
            )

    def forward(self, x):
        b, c, d, h, w, t = x.size()
        window_size, shift_size = get_window_size((d, h, w, t), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w t -> b d h w t c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        tp = int(np.ceil(t / window_size[3])) * window_size[3]
        attn_mask = None
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, t, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w t c -> b c d h w t")

        return x


class PositionalEmbedding(nn.Module):
    """
    Absolute positional embedding module
    """

    def __init__(
        self, dim: int, patch_dim: tuple
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            patch_num: total number of patches per time frame
            time_num: total number of time frames
        """

        super().__init__()
        self.dim = dim
        self.patch_dim = patch_dim
        d, h, w, t = patch_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, d, h, w, 1))
        self.time_embed = nn.Parameter(torch.zeros(1, dim, 1, 1, 1, t))

        
        trunc_normal_(self.pos_embed, std=0.02)
        
        trunc_normal_(self.time_embed, std=0.02)


    def forward(self, x):
        b, c, d, h, w, t = x.shape

        x = x + self.pos_embed
        # only add time_embed up to the time frame of the input in case the input size changes
        x = x + self.time_embed[:, :, :, :, :, :t]

        return x

class SwinTransformer4D(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        img_size: Tuple,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        first_window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 4,
        c_multiplier: int = 2,
        last_layer_full_MSA: bool = False,
        downsample="mergingv2",
        num_classes=2,
        to_float: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).


            c_multiplier: multiplier for the feature length after patch merging
        """

        super().__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.first_window_size = first_window_size
        self.patch_size = patch_size
        self.to_float = to_float
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            flatten=False,
            spatial_dims=spatial_dims,
        )
        grid_size = self.patch_embed.grid_size
        self.grid_size = grid_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        #patch_num = int((img_size[0]/patch_size[0]) * (img_size[1]/patch_size[1]) * (img_size[2]/patch_size[2]))
        #time_num = int(img_size[3]/patch_size[3])
        patch_dim =  ((img_size[0]//patch_size[0]), (img_size[1]//patch_size[1]), (img_size[2]//patch_size[2]), (img_size[3]//patch_size[3]))

        #print img, patch size, patch dim
        print("img_size: ", img_size)
        print("patch_size: ", patch_size)
        print("patch_dim: ", patch_dim)
        self.pos_embeds = nn.ModuleList()
        pos_embed_dim = embed_dim
        for i in range(self.num_layers):
            self.pos_embeds.append(PositionalEmbedding(pos_embed_dim, patch_dim))
            pos_embed_dim = pos_embed_dim * c_multiplier
            patch_dim = (patch_dim[0]//2, patch_dim[1]//2, patch_dim[2]//2, patch_dim[3])

        # build layer
        self.layers = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
    
        layer = BasicLayer(
            dim=int(embed_dim),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=self.first_window_size,
            drop_path=dpr[sum(depths[:0]) : sum(depths[: 0 + 1])],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            c_multiplier=c_multiplier,
            downsample=down_sample_mod if 0 < self.num_layers - 1 else None,
            use_checkpoint=use_checkpoint,
        )
        self.layers.append(layer)

        # exclude last layer
        for i_layer in range(1, self.num_layers - 1):
            layer = BasicLayer(
                dim=int(embed_dim * (c_multiplier**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=down_sample_mod if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        if not last_layer_full_MSA:
            layer = BasicLayer(
                dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
                depth=depths[(self.num_layers - 1)],
                num_heads=num_heads[(self.num_layers - 1)],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[: (self.num_layers - 1)]) : sum(depths[: (self.num_layers - 1) + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        else:
            #################Full MSA for last layer#####################

            self.last_window_size = (
                self.grid_size[0] // int(2 ** (self.num_layers - 1)),
                self.grid_size[1] // int(2 ** (self.num_layers - 1)),
                self.grid_size[2] // int(2 ** (self.num_layers - 1)),
                self.window_size[3],
            )

            layer = BasicLayer_FullAttention(
                dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
                depth=depths[(self.num_layers - 1)],
                num_heads=num_heads[(self.num_layers - 1)],
                # change the window size to the entire grid size
                window_size=self.last_window_size,
                drop_path=dpr[sum(depths[: (self.num_layers - 1)]) : sum(depths[: (self.num_layers - 1) + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

            #############################################################

        self.num_features = int(embed_dim * c_multiplier ** (self.num_layers - 1))

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        # self.head = nn.Linear(self.num_features, 1) if num_classes == 2 else num_classes # moved this part to clf_mlp or reg_mlp


    def forward(self, x):

        #print model parameters
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data.shape)

        if self.to_float:
            # converting tensor to float
            x = x.float()
        x = self.patch_embed(x)
        x = self.pos_drop(x)  # (b, c, h, w, d, t)

        for i in range(self.num_layers):
            x = self.pos_embeds[i](x)
            x = self.layers[i](x.contiguous())

        # moved this part to clf_mlp or reg_mlp

        # x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        # x = self.norm(x)  # B L C
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        # x = self.head(x)

        return x

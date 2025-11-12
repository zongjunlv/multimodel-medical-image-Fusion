"""
State Space 3D (SS3D) implementation for MedMamba
Extended from SS2D to support 3D volumetric data (medical volumes, videos, etc.)
"""
import time
import math
from functools import partial
from typing import Optional, Callable
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass


def flops_selective_scan_ref_3d(B=1, L=4096, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    FLOPS calculation for 3D selective scan
    Similar to 2D version but with 3D spatial dimensions
    
    Args:
        B: batch size
        L: sequence length (D*H*W for 3D)
        D: model dimension 
        N: state dimension
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    
    assert not with_complex

    flops = 0
    
    # Core einsum operations for 3D
    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    
    # Sequential processing
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 

    # Optional components
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    
    return flops


class SS3D(nn.Module):
    """
    3D State Space Model for volumetric data processing
    
    Extends SS2D to handle 3D inputs with shape (B, D, H, W, C)
    Uses 6 scanning directions for comprehensive 3D context modeling
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        scan_directions=6,  # 6 directions for 3D: +/-x, +/-y, +/-z
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.scan_directions = scan_directions  # Number of scanning directions

        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # 3D Convolution instead of 2D
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # Projection layers for each scanning direction
        self.x_proj = tuple(
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.scan_directions)
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # Delta time projections
        self.dt_projs = tuple(
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.scan_directions)
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        # State parameters
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.scan_directions, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.scan_directions, merge=True)

        # Forward core function
        self.forward_core = self.forward_corev0

        # Output layers
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        """Initialize delta time projection"""
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        """Initialize A matrix (state transition matrix)"""
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        """Initialize D parameter (skip connection)"""
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def create_3d_scan_sequences(self, x):
        """
        Create 6 different scanning sequences for 3D volume
        
        Args:
            x: input tensor of shape (B, C, D, H, W)
            
        Returns:
            xs: tensor of shape (B, K, C, L) where K=6, L=D*H*W
        """
        B, C, D, H, W = x.shape
        L = D * H * W
        
        # Reshape to sequence: (B, C, D*H*W)
        x_dhw = x.view(B, C, L)
        
        # Create 6 scanning directions for 3D
        sequences = []
        
        # 1. Forward depth-first (D->H->W)
        seq1 = x.view(B, C, L)  # Already in D*H*W order
        sequences.append(seq1)
        
        # 2. Reverse depth-first
        seq2 = torch.flip(seq1, dims=[-1])
        sequences.append(seq2)
        
        # 3. Height-first (H->D->W) 
        x_hdw = x.permute(0, 1, 3, 2, 4).contiguous().view(B, C, L)
        sequences.append(x_hdw)
        
        # 4. Reverse height-first
        seq4 = torch.flip(x_hdw, dims=[-1])
        sequences.append(seq4)
        
        # 5. Width-first (W->D->H)
        x_wdh = x.permute(0, 1, 4, 2, 3).contiguous().view(B, C, L)
        sequences.append(x_wdh)
        
        # 6. Reverse width-first
        seq6 = torch.flip(x_wdh, dims=[-1])
        sequences.append(seq6)
        
        # Stack all sequences: (B, K=6, C, L)
        xs = torch.stack(sequences, dim=1)
        
        return xs

    def reconstruct_from_sequences(self, ys, B, D, H, W):
        """
        Reconstruct 3D output from 6 scanning sequences
        
        Args:
            ys: output from selective scan, shape (B, K=6, C, L)
            B, D, H, W: original dimensions
            
        Returns:
            y: reconstructed output of shape (B, D, H, W, C)
        """
        K, C, L = ys.shape[1], ys.shape[2], ys.shape[3]
        
        # Process each scanning direction
        y1 = ys[:, 0].view(B, C, D, H, W)  # Forward depth-first
        y2 = torch.flip(ys[:, 1], dims=[-1]).view(B, C, D, H, W)  # Reverse depth-first
        
        # Height-first sequences
        y3_hdw = ys[:, 2].view(B, C, H, D, W)
        y3 = y3_hdw.permute(0, 1, 3, 2, 4).contiguous()  # H->D->W back to D->H->W
        
        y4_hdw = torch.flip(ys[:, 3], dims=[-1]).view(B, C, H, D, W) 
        y4 = y4_hdw.permute(0, 1, 3, 2, 4).contiguous()
        
        # Width-first sequences  
        y5_wdh = ys[:, 4].view(B, C, W, D, H)
        y5 = y5_wdh.permute(0, 1, 3, 4, 2).contiguous()  # W->D->H back to D->H->W
        
        y6_wdh = torch.flip(ys[:, 5], dims=[-1]).view(B, C, W, D, H)
        y6 = y6_wdh.permute(0, 1, 3, 4, 2).contiguous()
        
        # Combine all directions
        y_combined = y1 + y2 + y3 + y4 + y5 + y6
        
        # Convert to (B, D, H, W, C) format
        y = y_combined.permute(0, 2, 3, 4, 1).contiguous()
        
        return y

    def forward_corev0(self, x: torch.Tensor):
        """
        Core forward pass using selective scan
        
        Args:
            x: input tensor of shape (B, C, D, H, W)
            
        Returns:
            y: output tensor of shape (B, D, H, W, C)
        """
        self.selective_scan = selective_scan_fn
        
        B, C, D, H, W = x.shape
        L = D * H * W
        K = self.scan_directions

        # Create scanning sequences
        xs = self.create_3d_scan_sequences(x)  # (B, K, C, L)

        # Project to get delta, B, C parameters
        x_dbl = torch.einsum("b k c l, k d c -> b k d l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        # Project delta times
        dts = torch.einsum("b k r l, k c r -> b k c l", dts, self.dt_projs_weight)

        # Prepare tensors for selective scan
        xs = xs.float().view(B, -1, L)  # (b, k * c, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * c, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * c)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * c, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * c)

        # Selective scan
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        
        assert out_y.dtype == torch.float

        # Reconstruct 3D output
        y = self.reconstruct_from_sequences(out_y, B, D, H, W)
        
        return y

    def forward_corev1(self, x: torch.Tensor):
        """Alternative forward core using selective_scan_fn_v1"""
        self.selective_scan = selective_scan_fn_v1

        B, C, D, H, W = x.shape
        L = D * H * W
        K = self.scan_directions

        # Create scanning sequences
        xs = self.create_3d_scan_sequences(x)  # (B, K, C, L)

        # Project to get delta, B, C parameters
        x_dbl = torch.einsum("b k c l, k d c -> b k d l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        # Project delta times
        dts = torch.einsum("b k r l, k c r -> b k c l", dts, self.dt_projs_weight)

        # Prepare tensors for selective scan
        xs = xs.float().view(B, -1, L)  # (b, k * c, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * c, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * c)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * c, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * c)

        # Selective scan
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        
        assert out_y.dtype == torch.float

        # Reconstruct 3D output
        y = self.reconstruct_from_sequences(out_y, B, D, H, W)
        
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Forward pass for SS3D
        
        Args:
            x: input tensor of shape (B, D, H, W, C)
            
        Returns:
            out: output tensor of shape (B, D, H, W, C)
        """
        B, D, H, W, C = x.shape

        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, d, h, w, c_inner)

        # Convert to (B, C, D, H, W) for 3D convolution
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))  # (b, c_inner, d, h, w)
        
        # Apply core selective scan
        y = self.forward_core(x)  # (b, d, h, w, c_inner)
        
        # Normalize and apply gating
        y = self.out_norm(y)
        y = y * F.silu(z)
        
        # Final projection
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
            
        return out


# Utility function to create different SS3D variants
def create_ss3d_tiny(d_model=96, **kwargs):
    """Create a tiny SS3D model"""
    return SS3D(d_model=d_model, d_state=8, **kwargs)


def create_ss3d_small(d_model=192, **kwargs):  
    """Create a small SS3D model"""
    return SS3D(d_model=d_model, d_state=16, **kwargs)


def create_ss3d_base(d_model=384, **kwargs):
    """Create a base SS3D model"""
    return SS3D(d_model=d_model, d_state=16, **kwargs)


def create_ss3d_large(d_model=768, **kwargs):
    """Create a large SS3D model"""
    return SS3D(d_model=d_model, d_state=32, **kwargs)

"""
3D Patch Embedding layers for MedMamba
Extended from 2D patch embedding to support volumetric data
"""
import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed3D(nn.Module):
    """
    3D Volume to Patch Embedding
    
    Converts 3D volumetric data (like medical volumes, videos) into patch sequences
    """
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None, **kwargs):
        """
        Args:
            patch_size (int or tuple): Patch token size. Default: 4.
            in_chans (int): Number of input volume channels. Default: 1 (for medical images).
            embed_dim (int): Number of linear projection output channels. Default: 96.
            norm_layer (nn.Module, optional): Normalization layer. Default: None
        """
        super().__init__()
        
        # Handle different patch size formats
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif len(patch_size) == 2:
            # If only 2D patch size given, use same for depth
            patch_size = (patch_size[0], patch_size[0], patch_size[1])
            
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # 3D Convolution for patch projection
        self.proj = nn.Conv3d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Optional normalization
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, C, D, H, W)
            
        Returns:
            output tensor of shape (B, D', H', W', embed_dim)
            where D' = D//patch_size[0], H' = H//patch_size[1], W' = W//patch_size[2]
        """
        # Apply 3D convolution: (B, C, D, H, W) -> (B, embed_dim, D', H', W')
        x = self.proj(x)
        
        # Permute to (B, D', H', W', embed_dim) format
        x = x.permute(0, 2, 3, 4, 1)
        
        # Apply normalization if specified
        if self.norm is not None:
            x = self.norm(x)
            
        return x


class PatchEmbed3DVideo(nn.Module):
    """
    Specialized 3D Patch Embedding for video data
    
    Treats temporal dimension differently from spatial dimensions
    """
    def __init__(self, 
                 spatial_patch_size=4, 
                 temporal_patch_size=2,
                 in_chans=3, 
                 embed_dim=96, 
                 norm_layer=None, 
                 **kwargs):
        """
        Args:
            spatial_patch_size (int): Spatial patch size for H, W dimensions
            temporal_patch_size (int): Temporal patch size for T dimension  
            in_chans (int): Number of input channels (3 for RGB video)
            embed_dim (int): Output embedding dimension
            norm_layer: Optional normalization layer
        """
        super().__init__()
        
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # 3D convolution with different kernel sizes for temporal vs spatial
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
            stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size)
        )
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Args:
            x: input video tensor of shape (B, C, T, H, W)
            
        Returns:
            output tensor of shape (B, T', H', W', embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        x = x.permute(0, 2, 3, 4, 1)  # (B, T', H', W', embed_dim)
        
        if self.norm is not None:
            x = self.norm(x)
            
        return x


class AdaptivePatchEmbed3D(nn.Module):
    """
    Adaptive 3D Patch Embedding that can handle variable input sizes
    
    Useful for medical volumes that may have different resolutions
    """
    def __init__(self, 
                 target_patch_size=(4, 4, 4),
                 in_chans=1, 
                 embed_dim=96, 
                 norm_layer=None,
                 adaptive_method='interpolate',
                 **kwargs):
        """
        Args:
            target_patch_size: Target patch size for uniform processing
            in_chans: Number of input channels
            embed_dim: Output embedding dimension
            norm_layer: Optional normalization
            adaptive_method: Method for handling variable sizes ('interpolate' or 'pad')
        """
        super().__init__()
        
        self.target_patch_size = target_patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.adaptive_method = adaptive_method
        
        # Fixed size projection
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=target_patch_size,
            stride=target_patch_size
        )
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def _adapt_input_size(self, x):
        """Adapt input to work with target patch size"""
        B, C, D, H, W = x.shape
        target_d, target_h, target_w = self.target_patch_size
        
        # Calculate padding or interpolation needed
        pad_d = (target_d - D % target_d) % target_d
        pad_h = (target_h - H % target_h) % target_h
        pad_w = (target_w - W % target_w) % target_w
        
        if self.adaptive_method == 'pad':
            # Pad to make divisible by patch size
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        elif self.adaptive_method == 'interpolate':
            # Interpolate to nearest divisible size
            new_d = D + pad_d
            new_h = H + pad_h  
            new_w = W + pad_w
            if new_d != D or new_h != H or new_w != W:
                x = torch.nn.functional.interpolate(
                    x, size=(new_d, new_h, new_w), 
                    mode='trilinear', align_corners=False
                )
        
        return x

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, C, D, H, W) - any size
            
        Returns:
            output tensor of shape (B, D', H', W', embed_dim)
        """
        # Adapt input size
        x = self._adapt_input_size(x)
        
        # Apply projection
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = x.permute(0, 2, 3, 4, 1)  # (B, D', H', W', embed_dim)
        
        if self.norm is not None:
            x = self.norm(x)
            
        return x


class MultiScale3DPatchEmbed(nn.Module):
    """
    Multi-scale 3D patch embedding for hierarchical feature extraction
    
    Creates multiple patch sizes for multi-resolution processing
    """
    def __init__(self,
                 patch_sizes=[(2,2,2), (4,4,4), (8,8,8)],
                 in_chans=1,
                 embed_dims=[64, 96, 128],
                 norm_layer=None,
                 **kwargs):
        """
        Args:
            patch_sizes: List of patch sizes for different scales
            in_chans: Number of input channels
            embed_dims: Embedding dimensions for each scale
            norm_layer: Optional normalization layer
        """
        super().__init__()
        
        assert len(patch_sizes) == len(embed_dims), "Number of patch sizes must match embed dims"
        
        self.patch_sizes = patch_sizes
        self.embed_dims = embed_dims
        self.num_scales = len(patch_sizes)
        
        # Create projections for each scale
        self.projections = nn.ModuleList([
            nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            for patch_size, embed_dim in zip(patch_sizes, embed_dims)
        ])
        
        # Optional normalization for each scale
        if norm_layer is not None:
            self.norms = nn.ModuleList([
                norm_layer(embed_dim) for embed_dim in embed_dims
            ])
        else:
            self.norms = None

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, C, D, H, W)
            
        Returns:
            list of output tensors for each scale
        """
        outputs = []
        
        for i, proj in enumerate(self.projections):
            # Apply projection at each scale
            out = proj(x)  # (B, embed_dim, D', H', W')
            out = out.permute(0, 2, 3, 4, 1)  # (B, D', H', W', embed_dim)
            
            # Apply normalization if available
            if self.norms is not None:
                out = self.norms[i](out)
                
            outputs.append(out)
        
        return outputs


class SeparablePatchEmbed3D(nn.Module):
    """
    Separable 3D Patch Embedding using factorized convolutions
    
    More efficient for large 3D volumes by separating spatial and temporal/depth convolutions
    """
    def __init__(self, 
                 patch_size=(4,4,4), 
                 in_chans=1, 
                 embed_dim=96, 
                 norm_layer=None,
                 **kwargs):
        """
        Args:
            patch_size: 3D patch size (depth, height, width)
            in_chans: Number of input channels
            embed_dim: Output embedding dimension  
            norm_layer: Optional normalization layer
        """
        super().__init__()
        
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
            
        self.patch_size = patch_size
        d_patch, h_patch, w_patch = patch_size
        
        # Factorize 3D convolution into separate operations
        mid_dim = embed_dim // 2
        
        # Depth-wise convolution
        self.depth_conv = nn.Conv3d(
            in_chans, mid_dim,
            kernel_size=(d_patch, 1, 1),
            stride=(d_patch, 1, 1)
        )
        
        # Spatial convolution  
        self.spatial_conv = nn.Conv3d(
            mid_dim, embed_dim,
            kernel_size=(1, h_patch, w_patch),
            stride=(1, h_patch, w_patch)
        )
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, C, D, H, W)
            
        Returns:
            output tensor of shape (B, D', H', W', embed_dim)
        """
        # Apply factorized convolutions
        x = self.depth_conv(x)    # Process depth dimension
        x = self.spatial_conv(x)  # Process spatial dimensions
        
        # Permute to desired format
        x = x.permute(0, 2, 3, 4, 1)
        
        if self.norm is not None:
            x = self.norm(x)
            
        return x

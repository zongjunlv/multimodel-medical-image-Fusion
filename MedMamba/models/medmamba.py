"""
MedMamba: Vision Mamba for Medical Image Classification
"""
import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath

from .layers import (
    PatchEmbed2D, PatchMerging2D, VSSLayer,
    SS2D, flops_selective_scan_ref
)


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 4, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96,192,384,768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in SS_Conv_SSM, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, SS_Conv_SSM initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_backbone(x)
        x = x.permute(0,3,1,2)
        x = self.avgpool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.head(x)
        return x


def create_medmamba_tiny(num_classes=1000, depths=None, dims=None, **kwargs):
    """Create MedMamba-Tiny model"""
    # Use provided parameters or defaults
    if depths is None:
        depths = [2, 2, 4, 2]
    if dims is None:
        dims = [96, 192, 384, 768]
    
    model = VSSM(
        depths=depths,
        dims=dims,
        num_classes=num_classes,
        **kwargs
    )
    return model


def create_medmamba_small(num_classes=1000, depths=None, dims=None, **kwargs):
    """Create MedMamba-Small model"""
    # Use provided parameters or defaults
    if depths is None:
        depths = [2, 2, 8, 2]
    if dims is None:
        dims = [96, 192, 384, 768]
    
    model = VSSM(
        depths=depths,
        dims=dims,
        num_classes=num_classes,
        **kwargs
    )
    return model


def create_medmamba_base(num_classes=1000, depths=None, dims=None, **kwargs):
    """Create MedMamba-Base model"""
    # Use provided parameters or defaults
    if depths is None:
        depths = [2, 2, 12, 2]
    if dims is None:
        dims = [128, 256, 512, 1024]
    
    model = VSSM(
        depths=depths,
        dims=dims,
        num_classes=num_classes,
        **kwargs
    )
    return model

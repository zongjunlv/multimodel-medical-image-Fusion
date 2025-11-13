import math

from monai.networks.blocks.cablock import FeedForward
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from monai.networks.nets import DenseNet121, resnet18

from .layers import (
    PatchEmbed3D, PatchMerging3D, VSS3DLayer,
    SS3D
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Resnet18 = resnet18(
            spatial_dims = 3,
            n_input_channels = 1,
            num_classes = 3,
            pretrained= True,
            feed_forward = False,
            shortcut_type = 'A',
            bias_downsample = True
        )
        self.fc = nn.Linear(512, 3)
    def forward(self, x):
        x = self.Resnet18(x)
        x = self.fc(x)
        return x



class VSSM3D(nn.Module):
    """
    3D Vision State Space Model for volumetric medical image classification
    
    Processes 3D medical volumes (CT, MRI, etc.) using state space models
    with 6-directional scanning for comprehensive 3D context modeling
    """
    def __init__(self, 
                 patch_size=4, 
                 in_chans=1, 
                 num_classes=3, 
                 depths=[2, 2, 4, 2],
                 dims=[96, 192, 384, 768], 
                 d_state=16, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 patch_norm=True,
                 use_checkpoint=False,
                 scan_directions=6,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.scan_directions = scan_directions

        # 3D Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )

        # Absolute position embedding (optional)
        self.ape = False
        if self.ape:
            # This would need to be implemented for 3D case
            # For now, we disable it
            pass
        
        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSS3DLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                attn_drop=attn_drop_rate, 
                scan_directions=scan_directions,
            )
            self.layers.append(layer)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool3d(1)  # 3D global average pooling
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):  # 3D convolutions
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _init_weights(self, m: nn.Module):
        """Initialize model weights"""
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
        """
        Forward pass through the backbone
        
        Args:
            x: input tensor of shape (B, C, D, H, W)
            
        Returns:
            features: output features of shape (B, D', H', W', C_out)
        """
        # Patch embedding: (B, C, D, H, W) -> (B, D', H', W', embed_dim)
        x = self.patch_embed(x)
        
        # Add position embedding if enabled
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # Forward through layers
        for layer in self.layers:
            x = layer(x)
            
        return x

    def forward(self, x):
        """
        Forward pass for classification
        
        Args:
            x: input tensor of shape (B, C, D, H, W)
            
        Returns:
            logits: classification logits of shape (B, num_classes)
        """
        # Get backbone features
        x = self.forward_backbone(x)  # (B, D', H', W', C)
        
        # Convert to (B, C, D', H', W') for pooling
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        # Global average pooling: (B, C, D', H', W') -> (B, C, 1, 1, 1)
        x = self.avgpool(x)
        
        # Flatten: (B, C, 1, 1, 1) -> (B, C)
        x = torch.flatten(x, start_dim=1)
        
        # Classification head
        x = self.head(x)
        
        return x


# Factory functions for different model sizes
def create_medmamba3d_tiny(num_classes=3, **kwargs):
    """Create MedMamba3D-Tiny model"""
    model = VSSM3D(
        depths=[2, 2, 4, 2],
        dims=[96, 192, 384, 768],
        num_classes=num_classes,
        **kwargs
    )
    return model


def create_medmamba3d_small(num_classes=3, **kwargs):
    """Create MedMamba3D-Small model"""
    model = VSSM3D(
        depths=[2, 2, 8, 2],
        dims=[96, 192, 384, 768],
        num_classes=num_classes,
        **kwargs
    )
    return model


def create_medmamba3d_base(num_classes=3):
    """Create MedMamba3D-Base model"""
    model = VSSM3D(
        depths=[2, 2, 12, 2],
        dims=[128, 256, 512, 1024],
        num_classes=num_classes
    )
    return model


def create_medmamba3d_large(num_classes=3, **kwargs):
    """Create MedMamba3D-Large model"""
    model = VSSM3D(
        depths=[2, 2, 16, 2],
        dims=[192, 384, 768, 1536],
        num_classes=num_classes,
        **kwargs
    )
    return model


class MedMamba3DClassifier(nn.Module):
    """
    Specialized 3D MedMamba for medical image classification
    
    Includes medical-specific preprocessing and postprocessing
    """
    def __init__(self,
                 model_size='base',
                 num_classes=3,
                 in_chans=1,
                 input_size=(64, 64, 64),
                 normalization='instance',
                 dropout_rate=0.1,
                 **kwargs):
        """
        Args:
            model_size: Size of the model ('tiny', 'small', 'base', 'large')
            num_classes: Number of output classes
            in_chans: Number of input channels
            input_size: Expected input size (D, H, W)
            normalization: Type of normalization ('batch', 'instance', 'group')
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Input normalization
        if normalization == 'instance':
            self.input_norm = nn.InstanceNorm3d(in_chans)
        elif normalization == 'batch':
            self.input_norm = nn.BatchNorm3d(in_chans)
        elif normalization == 'group':
            self.input_norm = nn.GroupNorm(num_groups=1, num_channels=in_chans)
        else:
            self.input_norm = nn.Identity()
        
        # Create backbone model
        model_creators = {
            'tiny': create_medmamba3d_tiny,
            'small': create_medmamba3d_small,
            'base': create_medmamba3d_base,
            'large': create_medmamba3d_large,
        }
        
        if model_size not in model_creators:
            raise ValueError(f"Model size {model_size} not supported. Choose from {list(model_creators.keys())}")
        
        self.backbone = model_creators[model_size](
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=dropout_rate,
            **kwargs
        )
        
    def preprocess(self, x):
        """
        Medical image preprocessing
        
        Args:
            x: raw input tensor
            
        Returns:
            preprocessed tensor
        """
        # Normalize input
        x = self.input_norm(x)
        
        # Clip extreme values (common in medical imaging)
        x = torch.clamp(x, min=-3, max=3)
        
        return x
    
    def forward(self, x):
        """
        Forward pass with medical preprocessing
        
        Args:
            x: input tensor of shape (B, C, D, H, W)
            
        Returns:
            logits or probabilities
        """
        # Preprocess
        x = self.preprocess(x)
        
        # Forward through backbone
        logits = self.backbone(x)
        
        return logits


# Example usage and testing function
def test_medmamba3d():
    """Test function for MedMamba3D models"""
    print("Testing MedMamba3D models...")
    
    # Test different model sizes
    models = {
        'tiny': create_medmamba3d_tiny(num_classes=10),
        'small': create_medmamba3d_small(num_classes=10),
        'base': create_medmamba3d_base(num_classes=10),
    }
    
    # Create test input (batch_size=2, channels=1, depth=32, height=32, width=32)
    test_input = torch.randn(2, 1, 32, 32, 32)
    
    for name, model in models.items():
        print(f"\nTesting {name} model:")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        try:
            model.eval()
            with torch.no_grad():
                output = model(test_input)
            print(f"  Output shape: {output.shape}")
            print(f"  ✓ Forward pass successful")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
    
    # Test medical classifier
    print(f"\nTesting MedMamba3DClassifier:")
    classifier = MedMamba3DClassifier(
        model_size='tiny',
        num_classes=3,
        input_size=(32, 32, 32)
    )
    
    try:
        classifier.eval()
        with torch.no_grad():
            output = classifier(test_input)
        print(f"  Medical classifier output shape: {output.shape}")
        print(f"  ✓ Medical classifier successful")
    except Exception as e:
        print(f"  ✗ Medical classifier failed: {e}")


if __name__ == "__main__":
    test_medmamba3d()

"""
Configuration file for MedMamba training
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Data configuration"""
    train_root: str = "the path of your train set"
    val_root: str = "the path of your validation set"
    batch_size: int = 32
    num_workers: int = 0  # Use 0 to avoid multiprocessing issues
    img_size: Tuple[int, int] = (224, 224)
    
    # Medical image processing options
    use_medical_transforms: bool = False  # Use standard torchvision transforms instead of MONAI
    augmentation_level: str = "light"  # "light", "medium", "heavy"
    
    
    # Data transforms parameters
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "medmamba"
    num_classes: int = 6
    
    # MedMamba specific parameters
    patch_size: int = 4
    in_chans: int = 3
    depths: List[int] = field(default_factory=lambda: [2, 2, 4, 2])
    dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    d_state: int = 16
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 100
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    optimizer: str = "adam"
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    save_best_only: bool = True
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 15  # Stop if no improvement for N epochs
    early_stopping_metric: str = "f1"  # Metric to monitor: "accuracy" or "f1"
    
    # Device
    device: str = "cuda:0"
    
    # Logging
    print_freq: int = 10
    

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        # Ensure save directory exists
        os.makedirs(self.training.save_dir, exist_ok=True)
        
        # Auto-detect number of workers if not specified
        if self.data.num_workers == 8:
            self.data.num_workers = min([
                os.cpu_count(), 
                self.data.batch_size if self.data.batch_size > 1 else 0, 
                8
            ])


def get_config() -> Config:
    """Get default configuration"""
    return Config()


def get_medmamba_tiny_config() -> Config:
    """Get MedMamba-Tiny configuration"""
    config = Config()
    config.model.depths = [2, 2, 4, 2]
    config.model.dims = [96, 192, 384, 768]
    return config


def get_medmamba_small_config() -> Config:
    """Get MedMamba-Small configuration"""
    config = Config()
    config.model.depths = [2, 2, 8, 2]
    config.model.dims = [96, 192, 384, 768]
    return config


def get_medmamba_base_config() -> Config:
    """Get MedMamba-Base configuration"""
    config = Config()
    config.model.depths = [2, 2, 12, 2]
    config.model.dims = [128, 256, 512, 1024]
    return config


def get_medical_image_config() -> Config:
    """Get optimized configuration for medical image analysis"""
    config = Config()
    
    # Optimize for medical images
    config.data.use_medical_transforms = True
    config.data.augmentation_level = "light"  # Conservative augmentation
    config.data.img_size = 224
    config.data.batch_size = 16  # Smaller batch for medical images
    
    # Medical-friendly normalization (for CT/MRI)
    config.data.normalize_mean = (0.485,)  # Single channel medical images
    config.data.normalize_std = (0.229,)
    
    # Model adjustments for medical data
    config.model.in_chans = 1  # Grayscale medical images
    config.model.drop_path_rate = 0.05  # Reduced dropout for medical data
    
    # Training adjustments
    config.training.learning_rate = 0.0001
    config.training.weight_decay = 1e-4
    
    return config




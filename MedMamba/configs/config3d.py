"""
Configuration file for MedMamba3D training and inference
"""
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union


@dataclass
class Data3DConfig:
    """3D Data configuration for volumetric medical data"""
    # ==================== 数据配置区域 ====================
    # CSV文件路径（样本级别，包含file_path, label, class_name列）
    train_csv: Optional[str] = "dataset/ABUS_Classification/330_512_512/330_512_512_train.csv"
    val_csv: Optional[str] = "dataset/ABUS_Classification/330_512_512/330_512_512_val.csv"  # 如果为None，将使用train_csv的一部分作为验证集
    test_csv: Optional[str] = None  # 如果为None，使用目录方式加载
    
    # 数据集根目录（如果不使用CSV，则使用目录方式）
    train_root: str = "/data02/workspace/LZJ_SPACE/dataset/ABUS_Classification/110_256_256"
    val_root: str = "/data02/workspace/LZJ_SPACE/dataset/ABUS_Classification/110_256_256"
    test_root: str = "/data02/workspace/LZJ_SPACE/dataset/ABUS_Classification/330_512_512"
    
    # 训练参数
    batch_size: int = 8  # Smaller batch size for 3D data due to memory constraints
    num_workers: int = 2  # Fewer workers for 3D data loading
    val_split: float = 0.2  # 如果没有单独的验证CSV，从训练集分割的比例
    
    # 3D volume dimensions
    volume_size: Tuple[int, int, int] = (128, 128, 128)  # (D, H, W)
    
    # Data format and channels
    in_chans: int = 1  # Grayscale for medical images (CT, MRI)
    data_format: str = "NIFTI"  # NIFTI, DICOM, or NUMPY
    
    # Data preprocessing
    normalize_mean: Union[float, Tuple[float, ...]] = 0.0
    normalize_std: Union[float, Tuple[float, ...]] = 1.0
    intensity_range: Tuple[float, float] = (-1000, 1000)  # HU units for CT
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: int = 15  # degrees
    elastic_deform: bool = False
    noise_factor: float = 0.1


@dataclass
class Model3DConfig:
    """3D Model configuration for MedMamba3D"""
    # ==================== 模型配置区域 ====================
    model_name: str = "medmamba3d"
    model_size: str = "base"  # tiny, small, base, large - 根据您的checkpoint选择
    num_classes: int = 3
    class_names: List[str] = field(default_factory=lambda: ['benign', 'healthy', 'malignant'])
    
    # Input specifications
    patch_size: Union[int, Tuple[int, int, int]] = 4
    in_chans: int = 1
    
    # Architecture parameters
    depths: List[int] = field(default_factory=lambda: [2, 2, 4, 2])
    dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    d_state: int = 16           # 隐状态维度。值越大，能捕获更长程的依赖关系，但计算量增加
    scan_directions: int = 6  # 6-directional scanning for 3D
    
    # Regularization 
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    
    # 3D-specific options
    use_checkpoint: bool = True  # Important for 3D due to memory usage
    normalization: str = "instance"  # batch, instance, group
    
    # Medical-specific preprocessing
    medical_preprocessing: bool = True
    intensity_clipping: bool = True
    clip_range: Tuple[float, float] = (-3.0, 3.0)


@dataclass
class Training3DConfig:
    """3D Training configuration"""
    # ==================== 训练配置区域 ====================
    epochs: int = 200  # Longer training for 3D data
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    optimizer: str = "adamw"  # AdamW often works better for 3D
    
    # Checkpoint路径（用于测试）
    checkpoint_path: str = "/data02/workspace/LZJ_SPACE/checkpoints_3d/medmamba3d_best.pth"
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    
    # Gradient handling
    grad_clip_norm: float = 1.0
    accumulation_steps: int = 2  # Gradient accumulation for larger effective batch size
    
    # Checkpointing
    save_dir: str = "./checkpoints_3d"
    save_best_only: bool = True
    save_frequency: int = 10  # Save every N epochs
    
    # Validation
    val_frequency: int = 5  # Validate every N epochs (3D is expensive)
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 20  # Stop if no improvement for N epochs
    early_stopping_metric: str = "f1"  # Metric to monitor: "accuracy" or "f1"
    
    # Mixed precision training (recommended for 3D)
    use_amp: bool = True
    
    # Device configuration
    device: str = "cuda:0"
    multi_gpu: bool = False
    
    # Logging
    print_freq: int = 10
    log_wandb: bool = False
    wandb_project: str = "medmamba3d"


@dataclass
class Config3D:
    """Main 3D configuration class"""
    data: Data3DConfig = field(default_factory=Data3DConfig)
    model: Model3DConfig = field(default_factory=Model3DConfig)
    training: Training3DConfig = field(default_factory=Training3DConfig)
    
    def __post_init__(self):
        # Ensure directories exist
        os.makedirs(self.training.save_dir, exist_ok=True)
        
        # Auto-detect number of workers if not specified
        if self.data.num_workers == 4:
            self.data.num_workers = min([
                os.cpu_count() // 2,  # Use fewer cores for 3D data
                self.data.batch_size if self.data.batch_size > 1 else 0, 
                4
            ])
        
        # Sync model and data configurations
        self.model.in_chans = self.data.in_chans
        self.model.num_classes = self.model.num_classes
        
        # Adjust batch size for memory constraints
        if self.data.batch_size > 8:
            print(f"⚠️ Warning: Batch size {self.data.batch_size} may be too large for 3D data. Consider reducing.")


def get_config3d() -> Config3D:
    """Get default 3D configuration"""
    return Config3D()


def get_medmamba3d_tiny_config() -> Config3D:
    """Get MedMamba3D-Tiny configuration"""
    config = Config3D()
    config.model.model_size = "tiny"
    config.model.depths = [2, 2, 4, 2]
    config.model.dims = [64, 128, 256, 512]
    config.data.batch_size = 8  # Can handle larger batches
    return config


def get_medmamba3d_small_config() -> Config3D:
    """Get MedMamba3D-Small configuration"""
    config = Config3D()
    config.model.model_size = "small"
    config.model.depths = [2, 2, 8, 2]
    config.model.dims = [96, 192, 384, 768]
    config.data.batch_size = 4
    return config


def get_medmamba3d_base_config() -> Config3D:
    """Get MedMamba3D-Base configuration"""
    config = Config3D()
    config.model.model_size = "base"
    config.model.depths = [2, 2, 12, 2]
    config.model.dims = [128, 256, 512, 1024]
    config.data.batch_size = 2
    return config


def get_medmamba3d_large_config() -> Config3D:
    """Get MedMamba3D-Large configuration"""
    config = Config3D()
    config.model.model_size = "large"
    config.model.depths = [2, 2, 16, 2]
    config.model.dims = [192, 384, 768, 1536]
    config.data.batch_size = 1  # Very memory intensive
    config.training.accumulation_steps = 8  # Use gradient accumulation
    return config




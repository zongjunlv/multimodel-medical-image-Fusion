import os
import torch


from trainer import MedMambaTrainer
from data import create_data_loaders
from utils import setup_logger, log_config, log_model_info
from configs.config3d import get_config3d, get_medmamba3d_tiny_config, get_medmamba3d_small_config
from models import (
    create_medmamba_tiny, create_medmamba_small, create_medmamba_base,
    create_medmamba3d_tiny, create_medmamba3d_small, create_medmamba3d_base, create_medmamba3d_large
)


# ============================================================
# 配置区域 - 在这里修改您的训练参数
# ============================================================
USE_CONFIG = "default"  # 选择配置：'default', 'tiny', 'small', 'base'
LOG_FILE = None  # 日志文件路径，None表示只输出到控制台
QUIET_MODE = False  # 是否减少日志输出
# ============================================================



def get_dataset_config(config_type='default'):
    """Get configuration"""
    if config_type == 'tiny':
        return get_medmamba3d_tiny_config()
    elif config_type == 'small':
        return get_medmamba3d_small_config()
    else:
        return get_config3d()


def create_model(model_name, config, num_classes, **kwargs):
    """Create model based on name and configuration (automatically select 2D/3D)"""
    
    # Detect if this is a 3D configuration
    is_3d_config = hasattr(config.data, 'volume_size') and not hasattr(config.data, 'img_size')
    
    if is_3d_config:
        # Use 3D models for 3D configuration
        print("🔍 检测到3D配置，使用3D模型")
        model_creators_3d = {
            'tiny': create_medmamba3d_tiny,
            'small': create_medmamba3d_small,
            'base': create_medmamba3d_base,
            'large': create_medmamba3d_large,
        }
        
        if model_name not in model_creators_3d:
            raise ValueError(f"Unknown 3D model: {model_name}. Available: {list(model_creators_3d.keys())}")
        
        # Filter out architecture parameters that are pre-defined in 3D models
        # 3D models have hardcoded depths/dims, so we don't pass them
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['depths', 'dims']}
        
        return model_creators_3d[model_name](num_classes=num_classes, **filtered_kwargs)
    
    else:
        # Use 2D models for 2D configuration
        print("🔍 检测到2D配置，使用2D模型")
        model_creators_2d = {
            'tiny': create_medmamba_tiny,
            'small': create_medmamba_small,
            'base': create_medmamba_base,
        }
        
        if model_name not in model_creators_2d:
            if model_name == 'large':
                raise ValueError(f"Model 'large' is only available for 3D configuration. For 2D, use: {list(model_creators_2d.keys())}")
            else:
                raise ValueError(f"Unknown 2D model: {model_name}. Available: {list(model_creators_2d.keys())}")
        
        # 2D models need all architecture parameters
        return model_creators_2d[model_name](num_classes=num_classes, **kwargs)


def main():
    """Main training function"""
    # Get configuration
    config = get_dataset_config(USE_CONFIG)
    
    print("="*70)
    print("MedMamba Training")
    print("="*70)
    print(f"📝 Configuration: {USE_CONFIG}")
    print(f"🏗️  Model: {config.model.model_size}")
    print(f"📊 Classes: {config.model.num_classes} - {config.model.class_names}")
    print(f"💾 Save dir: {config.training.save_dir}")
    print("="*70)
    
    # Setup device
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Setup logger
    logger = None if QUIET_MODE else setup_logger('MedMamba', LOG_FILE)
    
    if logger:
        logger.info("Starting MedMamba training")
        log_config(logger, config)
    
    # Validate data paths
    if not os.path.exists(config.data.train_root):
        raise ValueError(f"Training data path does not exist: {config.data.train_root}")
    if not os.path.exists(config.data.val_root):
        raise ValueError(f"Validation data path does not exist: {config.data.val_root}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_classes, class_to_idx = create_data_loaders(config)
    
    # Update num_classes in config if auto-detected
    if config.model.num_classes != num_classes:
        if logger:
            logger.info(f"Auto-detected {num_classes} classes, updating config")
        config.model.num_classes = num_classes
    
    # Create model
    model = create_model(
        config.model.model_size,
        config,

        num_classes=config.model.num_classes,
        patch_size=config.model.patch_size,
        
        in_chans=config.model.in_chans,
        depths=config.model.depths,
        dims=config.model.dims,
        d_state=config.model.d_state,

        drop_rate=config.model.drop_rate,
        attn_drop_rate=config.model.attn_drop_rate,
        drop_path_rate=config.model.drop_path_rate,

        use_checkpoint=config.model.use_checkpoint
    )
    
    model.to(device)
    
    if logger:
        log_model_info(logger, model, device)
    
    # Create trainer
    trainer = MedMambaTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger
    )
    
    # Start training
    try:
        results = trainer.train()
        
        # 训练完成的详细指标已经在trainer中打印
        if not logger:
            print("\n✅ 训练流程完成！")
            
    except KeyboardInterrupt:
        if logger:
            logger.info("Training interrupted by user")
        else:
            print("Training interrupted by user")
    except Exception as e:
        if logger:
            logger.error(f"Training failed with error: {e}")
        else:
            print(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()

"""
Dataset utilities for MedMamba
"""
from cgi import test
import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from .medical_dataset import MedicalImageFolder
from .transforms import get_transforms, get_3d_medical_transforms
from .medical_dataset import Medical3DImageFolder, CSV3DMedicalDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from .medical_dataset import CSV3DMedicalDataset

def create_data_loaders(config):
    """
    Create train and validation data loaders based on configuration
    
    Args:
        config: Configuration object containing data parameters
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes, class_to_idx)
    """
    
    # Get data transforms based on configuration type
    # Check if this is a 3D configuration
    is_3d_config = hasattr(config.data, 'volume_size') and not hasattr(config.data, 'img_size')
    
    if is_3d_config:
        print("🔍 检测到3D配置，使用3D医学图像变换")
        data_transform = get_3d_medical_transforms(config)
    else:
        print("🔍 检测到2D配置，使用2D图像变换")
        data_transform = get_transforms(config)
    
    # Check if we're using CSV files
    use_csv = config.data.train_csv is not None and os.path.exists(config.data.train_csv)
    
    if use_csv:
        print("📄 使用CSV文件加载数据集")
        return _create_data_loaders_from_csv(config, data_transform, is_3d_config)
    
    # Check if we're dealing with medical images (e.g., .nii.gz files)
    use_medical_dataset = _should_use_medical_dataset(config.data.test_root)
    
    if use_medical_dataset:
        if is_3d_config:
            print("🩺 检测到3D医学图像格式，使用 Medical3DImageFolder 加载器")
            print("   📝 支持格式: .nii.gz, .nii, .mhd, .mha, .dcm")
            print("   📦 目标体积大小:", config.data.volume_size)
            # Create 3D medical datasets
            train_dataset = Medical3DImageFolder(
                root=config.data.train_root,
                transform=data_transform["train"],
                target_volume_size=config.data.volume_size
            )
            
            val_dataset = Medical3DImageFolder(
                root=config.data.val_root,
                transform=data_transform["val"],
                target_volume_size=config.data.volume_size
            )

            test_dataset = Medical3DImageFolder(
                root=config.data.test_root,
                transform=data_transform["test"],
                target_volume_size=config.data.volume_size
            )
        else:
            print("🩺 检测到2D医学图像格式，使用 MedicalImageFolder 加载器")
            print("   📝 支持格式: .nii.gz, .nii, .mhd, .mha, .dcm")
            # Create 2D medical datasets  
            train_dataset = MedicalImageFolder(
                root=config.data.train_root,
                transform=data_transform["train"]
            )
            
            val_dataset = MedicalImageFolder(
                root=config.data.val_root,
                transform=data_transform["val"]
            )

            test_dataset = MedicalImageFolder(
                root=config.data.test_root,
                transform=data_transform["test"]
            )
    else:
        print("📸 检测到标准图像格式，使用 ImageFolder 加载器")
        print("   📝 支持格式: .jpg, .jpeg, .png, .bmp, .gif, .tiff")
        # Create standard datasets
        train_dataset = datasets.ImageFolder(
            root=config.data.train_root,
            transform=data_transform["train"]
        )
        
        val_dataset = datasets.ImageFolder(
            root=config.data.val_root,
            transform=data_transform["val"]
        )

        test_dataset = datasets.ImageFolder(
            root=config.data.test_root,
            transform=data_transform["test"]
        )
    
    # Get class information
    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_to_idx)
    
    # Save class indices to JSON
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    json_str = json.dumps(idx_to_class, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    print(f"\n📊 数据集统计:")
    print(f"  🎯 训练样本: {len(train_dataset)} 个")
    print(f"  🎯 验证样本: {len(val_dataset)} 个")
    print(f"  📝 类别数量: {num_classes} 个")
    print(f"  ⚙️ 数据加载进程: {config.data.num_workers} 个")
    print(f"  📦 批次大小: {config.data.batch_size}")
    
    return train_loader, val_loader, test_loader, num_classes, class_to_idx


def _should_use_medical_dataset(root_path):
    """
    Check if the dataset contains medical image files that require MedicalImageFolder
    
    Args:
        root_path: Path to the dataset root directory
        
    Returns:
        bool: True if medical dataset should be used, False otherwise
    """
    medical_extensions = ('.nii.gz', '.nii', '.mhd', '.mha', '.dcm')
    
    # Walk through the directory to find any medical image files
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in medical_extensions):
                return True
    
    return False


def _create_data_loaders_from_csv(config, data_transform, is_3d_config):
    """
    Create data loaders from CSV files
    
    Args:
        config: Configuration object
        data_transform: Dictionary of transforms for train/val/test
        is_3d_config: Whether this is a 3D configuration
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes, class_to_idx)
    """

    
    # CSV文件中的路径已经是从项目根目录开始的相对路径，不需要额外的root_dir
    # 传入空字符串或None，让CSV3DMedicalDataset直接使用CSV中的路径
    root_dir = ""  # CSV中的file_path已经包含完整的相对路径
    
    # Load training dataset from CSV
    if is_3d_config:
        full_train_dataset = CSV3DMedicalDataset(
            csv_path=config.data.train_csv,
            root_dir=root_dir,
            transform=data_transform["train"],
            target_volume_size=config.data.volume_size
        )
    else:
        raise NotImplementedError("2D CSV dataset not yet implemented")
    
    # Create validation dataset
    if config.data.val_csv and os.path.exists(config.data.val_csv):
        # Use separate validation CSV
        if is_3d_config:
            val_dataset = CSV3DMedicalDataset(
                csv_path=config.data.val_csv,
                root_dir=root_dir,
                transform=data_transform["val"],
                target_volume_size=config.data.volume_size
            )
    else:
        # Split training dataset for validation
        print(f"📊 从训练集分割验证集 (比例: {config.data.val_split})")
        
        # Get indices for train/val split
        total_size = len(full_train_dataset)
        indices = list(range(total_size))
        
        # Stratified split based on labels
        labels = [full_train_dataset.data_frame.iloc[i]['label'] for i in indices]
        train_indices, val_indices = train_test_split(
            indices,
            test_size=config.data.val_split,
            stratify=labels,
            random_state=42
        )
        
        # Create subset datasets
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_train_dataset, val_indices)
        
        # Note: When using Subset, we need to access the original dataset for class info
        # Update: Create separate datasets for proper transform application
        # For now, use Subset
        print(f"   训练样本: {len(train_dataset)}")
        print(f"   验证样本: {len(val_dataset)}")
        
    # If we didn't split, use full dataset for training
    if not (config.data.val_csv is None or not os.path.exists(config.data.val_csv)):
        train_dataset = full_train_dataset
    
    # Create test dataset
    if config.data.test_csv and os.path.exists(config.data.test_csv):
        # Use separate test CSV
        if is_3d_config:
            test_dataset = CSV3DMedicalDataset(
                csv_path=config.data.test_csv,
                root_dir=root_dir,
                transform=data_transform["test"],
                target_volume_size=config.data.volume_size
            )
    else:
        # Use test_root directory if no CSV
        print(f"⚠️  No test CSV provided, using directory: {config.data.test_root}")
        from .medical_dataset import Medical3DImageFolder
        test_dataset = Medical3DImageFolder(
            root=config.data.test_root,
            transform=data_transform["test"],
            target_volume_size=config.data.volume_size
        )
    
    # Get class information from the dataset
    if hasattr(train_dataset, 'dataset'):
        # It's a Subset
        num_classes = len(train_dataset.dataset.classes)
        class_to_idx = train_dataset.dataset.class_to_idx
    else:
        num_classes = len(train_dataset.classes)
        class_to_idx = train_dataset.class_to_idx
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    print(f"✅ 数据加载器创建完成:")
    print(f"   训练批次: {len(train_loader)}")
    print(f"   验证批次: {len(val_loader)}")
    print(f"   测试批次: {len(test_loader)}")
    print(f"   类别数量: {num_classes}")
    
    return train_loader, val_loader, test_loader, num_classes, class_to_idx


def get_dataset_info(train_loader, val_loader):
    """
    Get basic information about the datasets
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        
    Returns:
        dict: Dataset information
    """
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    num_classes = len(train_loader.dataset.classes)
    
    return {
        'train_size': train_size,
        'val_size': val_size,
        'num_classes': num_classes,
        'classes': train_loader.dataset.classes
    }

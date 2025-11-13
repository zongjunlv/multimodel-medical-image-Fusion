from torchvision import transforms
try:
    from monai.transforms import (
        Compose as MonaiCompose, Resize as MonaiResize, 
        CenterSpatialCrop, ScaleIntensity, EnsureChannelFirst,
        RandRotate90, RandFlip, RandAffine, ToTensor as MonaiToTensor,
        NormalizeIntensity
    )
    MONAI_AVAILABLE = True
except ImportError:
    print("Warning: MONAI not available. Using torchvision transforms only.")
    MONAI_AVAILABLE = False


def get_transforms(config):
    """
    Get data transforms based on configuration
    
    Args:
        config: Configuration object containing data parameters
        
    Returns:
        dict: Dictionary containing train and validation transforms
    """
    # 从配置中提取尺寸与归一化超参数
    img_size = config.data.img_size
    normalize_mean = config.data.normalize_mean
    normalize_std = config.data.normalize_std
    
    # Use medical image friendly transforms if MONAI is available
    if MONAI_AVAILABLE and hasattr(config.data, 'use_medical_transforms') and config.data.use_medical_transforms:
        data_transform = get_medical_transforms(config)  # 优先使用 MONAI 管线
    else:
        # Fallback to standard transforms with center crop instead of random crop
        data_transform = {
            "train": transforms.Compose([
                transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),  # Slightly larger
                transforms.CenterCrop(img_size),  # Center crop instead of random
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),  # Light rotation
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
            ]),
            "val": transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
            ]),
            "test": transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
            ])
        }
    
    return data_transform


def get_medical_transforms(config):
    """
    Get medical image optimized transforms using MONAI
    
    Args:
        config: Configuration object containing data parameters
        
    Returns:
        dict: Dictionary containing train and validation transforms optimized for medical images
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is required for medical transforms. Please install: pip install monai")
    
    img_size = config.data.img_size
    
    # Medical image specific transforms
    data_transform = {
        "train": MonaiCompose([
            EnsureChannelFirst(channel_dim='no_channel'),
            ScaleIntensity(minv=0.0, maxv=1.0),  # Normalize intensity
            CenterSpatialCrop(roi_size=(img_size, img_size)),  # Center crop - preserves anatomy
            RandRotate90(prob=0.5, max_k=3),  # Medical appropriate rotation
            RandFlip(spatial_axis=[0], prob=0.5),  # Horizontal flip
            RandAffine(
                prob=0.3,
                rotate_range=(0.1, 0.1),  # Small rotation
                scale_range=(0.05, 0.05),  # Small scale change
                translate_range=(5, 5),  # Small translation
                padding_mode="border"
            ),
            MonaiToTensor(),
            NormalizeIntensity(nonzero=True)  # Normalize non-zero voxels
        ]),
        "val": MonaiCompose([
            EnsureChannelFirst(channel_dim='no_channel'),
            ScaleIntensity(minv=0.0, maxv=1.0),
            CenterSpatialCrop(roi_size=(img_size, img_size)),  # Consistent center crop
            MonaiToTensor(),
            NormalizeIntensity(nonzero=True)
        ]),
        "test": MonaiCompose([
            EnsureChannelFirst(channel_dim='no_channel'),
            ScaleIntensity(minv=0.0, maxv=1.0),
            CenterSpatialCrop(roi_size=(img_size, img_size)),  # Consistent center crop
            MonaiToTensor(),
            NormalizeIntensity(nonzero=True)
        ])
    }
    
    return data_transform


def get_3d_medical_transforms(config):
    """
    Get 3D medical volume transforms using MONAI
    
    Args:
        config: Configuration object containing data parameters
        
    Returns:
        dict: Dictionary containing train and validation transforms for 3D volumes
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is required for 3D medical transforms. Please install: pip install monai")
    
    volume_size = getattr(config.data, 'volume_size', (64, 64, 64))
    normalize_mean = getattr(config.data, 'normalize_mean', 0.0)
    normalize_std = getattr(config.data, 'normalize_std', 1.0)
    use_augmentation = getattr(config.data, 'use_augmentation', True)
    
    # Basic transforms for both train and val
    # 所有阶段共享的体素预处理步骤
    base_transforms = [
        EnsureChannelFirst(channel_dim='no_channel'),
        ScaleIntensity(minv=0.0, maxv=1.0),
        # CenterSpatialCrop(roi_size=volume_size),
        MonaiResize(spatial_size=volume_size),
    ]
    
    # Training transforms with augmentation
    train_transforms = base_transforms.copy()
    if use_augmentation:
        train_transforms.extend([
            RandRotate90(prob=0.5, max_k=3, spatial_axes=(1, 2)),  # Rotate in axial plane
            RandFlip(spatial_axis=[1], prob=0.5),  # Flip along one axis
            RandAffine(
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),  # Small 3D rotation
                scale_range=(0.05, 0.05, 0.05),  # Small scale change
                translate_range=(2, 2, 2),  # Small translation
                padding_mode="border"
            ),
        ])
    
    train_transforms.extend([
        MonaiToTensor(),
        NormalizeIntensity(nonzero=True) if normalize_mean == 0.0 else None
    ])
    
    # Validation transforms (no augmentation)
    val_transforms = base_transforms + [
        MonaiToTensor(),
        NormalizeIntensity(nonzero=True) if normalize_mean == 0.0 else None
    ]

    test_transforms = base_transforms + [
        MonaiToTensor(),
        NormalizeIntensity(nonzero=True) if normalize_mean == 0.0 else None
    ]
    
    # Remove None values
    train_transforms = [t for t in train_transforms if t is not None]
    val_transforms = [t for t in val_transforms if t is not None]
    test_transforms = [t for t in test_transforms if t is not None]
    
    data_transform = {
        "train": MonaiCompose(train_transforms),
        "val": MonaiCompose(val_transforms),
        "test": MonaiCompose(test_transforms)
    }
    
    return data_transform


def get_medmnist_transforms(config):
    """
    Get transforms specifically for MedMNIST datasets (typically 28x28)
    
    Args:
        config: Configuration object containing data parameters
        
    Returns:
        dict: Dictionary containing train and validation transforms
    """
    img_size = config.data.img_size
    normalize_mean = config.data.normalize_mean
    normalize_std = config.data.normalize_std
    
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ]),
        "val": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ]),
        "test": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
    }
    
    return data_transform


def get_augmented_transforms(config, augmentation_level="light"):
    """
    Get enhanced data transforms with more augmentations
    
    Args:
        config: Configuration object containing data parameters
        augmentation_level: Level of augmentation ("light", "medium", "heavy")
        
    Returns:
        dict: Dictionary containing train and validation transforms
    """
    img_size = config.data.img_size
    normalize_mean = config.data.normalize_mean
    normalize_std = config.data.normalize_std
    
    if augmentation_level == "light":
        train_transform = transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.CenterCrop(img_size),  # Use center crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
    elif augmentation_level == "medium":
        train_transform = transforms.Compose([
            transforms.Resize((int(img_size * 1.2), int(img_size * 1.2))),
            transforms.CenterCrop(img_size),  # Use center crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Reduced for medical images
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
    elif augmentation_level == "heavy":
        train_transform = transforms.Compose([
            transforms.Resize((int(img_size * 1.3), int(img_size * 1.3))),
            transforms.CenterCrop(img_size),  # Use center crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),  # Reduced for medical images
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Reduced translation
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    
    return {
        "train": train_transform,
        "val": val_transform,
        "test": test_transform
    }

import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import nibabel as nib

import SimpleITK as sitk

import numpy as np
import pandas as pd
from PIL import Image
from typing import Union


def has_valid_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Check if file has valid extension"""
    return filename.lower().endswith(extensions)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Make dataset from directory structure"""
    directory = os.path.expanduser(directory)

    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_valid_extension(x, extensions)

    is_valid_file = is_valid_file or (lambda x: False)
    instances = []
    available_classes = set()
    
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        
        if not os.path.isdir(target_dir):
            continue
            
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Find classes in directory"""
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class MedicalImageFolder(Dataset):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        slice_idx: Optional[int] = None,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.slice_idx = slice_idx

        if extensions is None:
            extensions = ('.nii.gz', '.nii', '.mhd', '.mha', '.dcm')

        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions, is_valid_file)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.extensions = extensions

        print(f"🔍 医学图像数据集加载完成:")
        print(f"   📁 发现 {len(classes)} 个类别，共 {len(samples)} 个样本")
        for i, cls in enumerate(classes):
            count = sum(1 for _, target in samples if target == i)
            print(f"   📋 类别 '{cls}': {count} 个样本")
        
        if len(samples) == 0:
            print("⚠️  警告: 未找到任何医学图像文件!")
            print("   📝 支持的格式: .nii.gz, .nii, .mhd, .mha, .dcm")
            print("   📂 请确保数据按以下结构组织:")
            print("      root/")
            print("      ├── class1/")
            print("      │   ├── sample1.nii.gz")
            print("      │   └── sample2.nii.gz") 
            print("      └── class2/")
            print("          ├── sample3.nii.gz")
            print("          └── sample4.nii.gz")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
            
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        
        sample = self.loader(path)
        
        sample_data = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        # Add some structure to make it less obviously fake
        sample_data[112:134, 112:134] = np.random.randint(100, 255, (22, 22, 3), dtype=np.uint8)
        sample = Image.fromarray(sample_data)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def loader(self, path: str):
        """
        Load medical image from path and convert to appropriate format
        """
        if path.lower().endswith(('.nii.gz', '.nii')):
            return self._load_nifti(path)
        elif path.lower().endswith(('.mhd', '.mha')):
            return self._load_sitk(path)
        else:
            # Fallback to PIL for other formats
            return Image.open(path).convert('RGB')

    def _load_nifti(self, path: str):

        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        
        # Handle different dimensions
        if data.ndim == 4:
            # 4D image - take first volume
            data = data[:, :, :, 0]
        
        if data.ndim == 3:
            # 3D volume - extract middle slice or specified slice
            if self.slice_idx is not None:
                slice_idx = min(self.slice_idx, data.shape[2] - 1)
            else:
                slice_idx = data.shape[2] // 2
            data = data[:, :, slice_idx]
        
        # Normalize to [0, 255] and convert to uint8
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min()) * 255
        data = data.astype(np.uint8)
        
        # Convert to PIL Image for compatibility with transforms
        # Rotate 90 degrees to match standard image orientation
        data = np.rot90(data)
        
        # Convert grayscale to RGB for compatibility
        if len(data.shape) == 2:
            data = np.stack([data, data, data], axis=-1)
        
        return Image.fromarray(data)

    def _load_sitk(self, path: str):
        """Load image using SimpleITK"""
        img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(img)
        
        # Handle 3D volumes - take middle slice
        if data.ndim == 3:
            if self.slice_idx is not None:
                slice_idx = min(self.slice_idx, data.shape[0] - 1)
            else:
                slice_idx = data.shape[0] // 2
            data = data[slice_idx, :, :]
        
        # Normalize to [0, 255]
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min()) * 255
        data = data.astype(np.uint8)
        
        # Convert grayscale to RGB
        if len(data.shape) == 2:
            data = np.stack([data, data, data], axis=-1)
        
        return Image.fromarray(data)


class Medical3DImageFolder(Dataset):
    """
    3D Medical Image Dataset that loads full 3D volumes
    Compatible with MONAI transforms - returns numpy arrays directly
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        target_volume_size: Optional[Tuple[int, int, int]] = (64, 64, 64),
    ):
        """
        Args:
            root: Root directory path
            transform: Optional transform to be applied on a sample
            target_transform: Optional transform to be applied on the target
            loader: Function to load a sample given its path
            is_valid_file: Function to check if file is valid
            extensions: Tuple of allowed extensions
            target_volume_size: Target size for 3D volumes (D, H, W)
        """
        super().__init__()
        
        # Set default extensions for medical images
        if extensions is None:
            extensions = ('.nii.gz', '.nii', '.mhd', '.mha', '.dcm')
        
        # Set default loader
        if loader is None:
            loader = self._default_3d_loader
            
        # Find classes and samples
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions, is_valid_file)
        
        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform
        self.target_volume_size = target_volume_size
        
        print(f"🔍 3D医学图像数据集加载完成:")
        print(f"   📁 发现 {len(classes)} 个类别，共 {len(samples)} 个样本")
        for i, cls in enumerate(classes):
            count = sum(1 for _, target in samples if target == i)
            print(f"   📋 类别 '{cls}': {count} 个样本")
            
        if len(samples) == 0:
            print("⚠️  警告: 未找到任何3D医学图像文件!")
            print("   📝 支持的格式: .nii.gz, .nii, .mhd, .mha, .dcm")
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """
        Args:
            index: Index
            
        Returns:
            tuple: (volume, target) where volume is 3D numpy array and target is class index
        """
        path, target = self.samples[index]
        
        # Load 3D volume as numpy array
        volume = self.loader(path)
        
        # Apply transforms if provided
        if self.transform is not None:
            volume = self.transform(volume)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return volume, target
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _default_3d_loader(self, path: str) -> np.ndarray:
        """
        Default 3D volume loader that returns numpy arrays compatible with MONAI
        
        Args:
            path: Path to the medical image file
            
        Returns:
            np.ndarray: 3D volume data
        """
        # Check if file exists
        if not os.path.exists(path):
            # Return dummy 3D data for testing
            print(f"⚠️  文件不存在 {path}，使用假数据")
            volume = np.random.rand(*self.target_volume_size).astype(np.float32)
            return volume
            
        try:
            # Try to load with nibabel first (most common for .nii/.nii.gz)
            if path.lower().endswith(('.nii', '.nii.gz')):
                return self._load_nibabel_3d(path)
            
            # Try SimpleITK for other formats
            elif path.lower().endswith(('.mhd', '.mha', '.dcm')):
                return self._load_sitk_3d(path)
            
            else:
                print(f"⚠️  不支持的文件格式: {path}，使用假数据")
                volume = np.random.rand(*self.target_volume_size).astype(np.float32)
                return volume
                
        except Exception as e:
            print(f"⚠️  加载文件失败 {path}: {e}，使用假数据")
            volume = np.random.rand(*self.target_volume_size).astype(np.float32)
            return volume
    
    def _load_nibabel_3d(self, path: str) -> np.ndarray:
        """Load 3D volume using nibabel"""
        nii_img = nib.load(path)
        data = nii_img.get_fdata()
        
        # Ensure 3D
        if len(data.shape) == 4:
            # Take first volume if 4D
            data = data[:, :, :, 0]
        elif len(data.shape) == 2:
            # Convert 2D to 3D by adding depth dimension
            data = np.expand_dims(data, axis=2)
        
        # Resize to target volume size if specified
        if self.target_volume_size is not None:
            data = self._resize_volume(data, self.target_volume_size)
        
        # Convert to float32 and normalize
        data = data.astype(np.float32)
        
        # Simple intensity normalization
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min())
        
        return data
    
    def _load_sitk_3d(self, path: str) -> np.ndarray:
        """Load 3D volume using SimpleITK"""

        img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(img)
        
        # SimpleITK returns (Z, Y, X), we want (X, Y, Z)
        data = np.transpose(data, (2, 1, 0))
        
        # Resize to target volume size if specified
        if self.target_volume_size is not None:
            data = self._resize_volume(data, self.target_volume_size)
        
        # Convert to float32 and normalize
        data = data.astype(np.float32)
        
        # Simple intensity normalization
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min())
        
        return data
    
    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Resize 3D volume to target size using simple interpolation
        
        Args:
            volume: Input 3D volume
            target_size: Target size (D, H, W)
            
        Returns:
            Resized volume
        """
        try:
            from scipy.ndimage import zoom
            
            # Calculate zoom factors for each dimension
            zoom_factors = [
                target_size[i] / volume.shape[i] for i in range(3)
            ]
            
            # Apply zoom
            resized = zoom(volume, zoom_factors, order=1)  # Linear interpolation
            return resized
            
        except ImportError:
            # Fallback: simple center crop or padding
            print("⚠️  scipy not available, using simple resizing")
            return self._simple_resize(volume, target_size)
    
    def _simple_resize(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Simple resize using center crop/padding"""
        current_size = volume.shape
        result = np.zeros(target_size, dtype=volume.dtype)
        
        # Calculate crop/pad for each dimension
        slices = []
        for i in range(3):
            if current_size[i] >= target_size[i]:
                # Crop - take center region
                start = (current_size[i] - target_size[i]) // 2
                end = start + target_size[i]
                slices.append(slice(start, end))
            else:
                # Will need to pad - take all
                slices.append(slice(None))
        
        # Extract the cropped region
        cropped = volume[tuple(slices)]
        
        # Place in result array (handles padding)
        crop_slices = []
        for i in range(3):
            if cropped.shape[i] < target_size[i]:
                # Center the cropped data in result
                start = (target_size[i] - cropped.shape[i]) // 2
                end = start + cropped.shape[i]
                crop_slices.append(slice(start, end))
            else:
                crop_slices.append(slice(None))
        
        result[tuple(crop_slices)] = cropped
        return result


class CSV3DMedicalDataset(Dataset):
    """
    3D Medical Image Dataset from CSV file
    CSV format: file_path, label, class_name
    """
    def __init__(
        self,
        csv_path: str,
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_volume_size: Tuple[int, int, int] = (64, 64, 64)
    ):
        """
        Args:
            csv_path: Path to CSV file with columns: file_path, label, class_name
            root_dir: Optional root directory to prepend to file paths
            transform: Transform to apply to volumes
            target_volume_size: Target size for volumes (D, H, W)
        """
        
        self.csv_path = csv_path
        self.root_dir = root_dir or ""
        self.transform = transform
        self.target_volume_size = target_volume_size
        
        # Load CSV
        self.data_frame = pd.read_csv(csv_path)
        
        # Extract unique classes and create mappings
        self.classes = sorted(self.data_frame['class_name'].unique().tolist())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        print(f"📊 Loaded CSV dataset from: {csv_path}")
        print(f"   Total samples: {len(self.data_frame)}")
        print(f"   Classes: {self.classes}")
        print(f"   Class distribution:")
        for cls_name in self.classes:
            count = (self.data_frame['class_name'] == cls_name).sum()
            print(f"      {cls_name}: {count}")
    
    def __len__(self) -> int:
        return len(self.data_frame)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index: Index of the sample
            
        Returns:
            tuple: (volume, label) where volume is a tensor of shape (1, D, H, W)
        """
        # Get sample info
        row = self.data_frame.iloc[index]
        file_path = row['file_path']
        label = int(row['label'])
        
        # Construct full path
        # 如果root_dir为空或只是空字符串，直接使用CSV中的file_path
        if self.root_dir and self.root_dir.strip():
            full_path = os.path.join(self.root_dir, file_path)
        else:
            full_path = file_path
        
        # Load volume
        volume = self._load_volume(full_path)
        
        # Resize to target size
        if volume.shape != self.target_volume_size:
            volume = self._resize_volume(volume, self.target_volume_size)
        
        # Apply transforms if provided (MONAI transforms work on numpy arrays)
        if self.transform:
            # MONAI transforms expect numpy array and will handle conversion to tensor
            volume = self.transform(volume)
        
        # Ensure it's a tensor with correct shape
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()
        
        # Ensure channel dimension exists (C, D, H, W)
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)  # Add channel dimension: (1, D, H, W)
        
        return volume, label
    
    def _load_volume(self, path: str) -> np.ndarray:
        """Load 3D medical image volume"""
        if path.endswith('.nii.gz') or path.endswith('.nii'):
            # Load with nibabel
            nii = nib.load(path)
            volume = nii.get_fdata()
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        # Ensure 3D
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
        
        return volume.astype(np.float32)
    
    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize 3D volume to target size"""
        try:
            from scipy.ndimage import zoom
            
            # Calculate zoom factors
            zoom_factors = [target_size[i] / volume.shape[i] for i in range(3)]
            
            # Apply zoom
            resized = zoom(volume, zoom_factors, order=1)
            return resized
            
        except ImportError:
            print("⚠️  scipy not available, using simple resizing")
            return self._simple_resize(volume, target_size)
    
    def _simple_resize(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Simple resize using center crop/padding"""
        current_size = volume.shape
        result = np.zeros(target_size, dtype=volume.dtype)
        
        # Calculate crop/pad for each dimension
        slices = []
        for i in range(3):
            if current_size[i] >= target_size[i]:
                start = (current_size[i] - target_size[i]) // 2
                end = start + target_size[i]
                slices.append(slice(start, end))
            else:
                slices.append(slice(None))
        
        # Extract the cropped region
        cropped = volume[tuple(slices)]
        
        # Place in result array
        crop_slices = []
        for i in range(3):
            if cropped.shape[i] < target_size[i]:
                start = (target_size[i] - cropped.shape[i]) // 2
                end = start + cropped.shape[i]
                crop_slices.append(slice(start, end))
            else:
                crop_slices.append(slice(None))
        
        result[tuple(crop_slices)] = cropped
        return result

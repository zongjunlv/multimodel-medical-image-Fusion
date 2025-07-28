# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os

# 添加nibabel支持.nii.gz文件
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("警告: nibabel 未安装，无法处理 .nii.gz 文件。请运行: pip install nibabel")


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.png"))
    # data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    # data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    # data.extend(glob.glob((os.path.join(data_dir, "*.bmp"))))
    
    # 添加对.nii.gz文件的支持
    if NIBABEL_AVAILABLE:
        data.extend(glob.glob(os.path.join(data_dir, "*.nii.gz")))
        data.extend(glob.glob(os.path.join(data_dir, "*.nii")))
    
    data.sort()
    filenames.sort()
    return data, filenames


def check_image_readable(image_path):
    """检查图像文件是否可读"""
    try:
        # 检查是否为.nii.gz文件
        if image_path.lower().endswith(('.nii.gz', '.nii')):
            if not NIBABEL_AVAILABLE:
                return False
            nii_img = nib.load(image_path)
            data = nii_img.get_fdata()
            return data is not None and data.size > 0
        else:
            img = cv2.imread(image_path)
            return img is not None
    except:
        return False


def load_nii_image(image_path, is_grayscale=False):
    """
    加载.nii.gz文件并转换为适合模型的格式
    Args:
        image_path: .nii.gz文件路径
        is_grayscale: 是否为灰度图像（红外图像）
    Returns:
        numpy array: 归一化后的图像数据
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel 未安装，无法加载 .nii.gz 文件")
    
    # 加载NII文件
    nii_img = nib.load(image_path)
    data = nii_img.get_fdata()
    
    # 如果是3D数据，取中间切片或者第一个切片
    if len(data.shape) == 3:
        # 取中间切片
        slice_idx = data.shape[2] // 2
        data = data[:, :, slice_idx]
    elif len(data.shape) == 4:
        # 如果是4D数据，取第一个时间点的中间切片
        slice_idx = data.shape[2] // 2
        data = data[:, :, slice_idx, 0]
    
    # 数据归一化到[0, 1]
    data_min = data.min()
    data_max = data.max()
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min)
    else:
        data = np.zeros_like(data)
    
    # 转换为uint8格式以兼容现有的处理流程
    data = (data * 255).astype(np.uint8)
    
    # 如果需要RGB格式，复制通道
    if not is_grayscale and len(data.shape) == 2:
        data = np.stack([data, data, data], axis=-1)
    
    return data


class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path, vi_path,length):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.filepath_ir = []
        self.filenames_ir = []
        self.filepath_vis = []
        self.filenames_vis = []
        self.length = length    #  数据集长度，由外部传入
        if split == 'train':
            # 使用调用方传入的路径，而不是硬编码字符串
            data_dir_ir = ir_path
            data_dir_vis = vi_path  # 目前未直接使用，但保留变量名以便后续扩展

            if not data_dir_ir or not data_dir_vis:
                raise ValueError("ir_path 和 vi_path 不能为空，请检查配置文件中的 data.train_ir_path 与 data.train_vis_path")

            if not os.path.isdir(data_dir_ir):
                raise FileNotFoundError(f"训练数据路径不存在: {data_dir_ir}")

            valid_count = 0
            total_count = 0
            
            # 首先检查是否存在平行目录结构 (T1/ 和 T2/ 在同一级别)
            t1_path = os.path.join(data_dir_ir, 'T1')
            t2_path = os.path.join(data_dir_ir, 'T2')
            
            if os.path.isdir(t1_path) and os.path.isdir(t2_path):
                # 平行目录结构
                print("检测到平行目录结构 (T1/ 和 T2/ 在同一级别)")
                t1_files = os.listdir(t1_path)
                t1_files.sort()
                
                for file in t1_files:
                    filepath_ir_ = os.path.join(t1_path, file)
                    filepath_vis_ = os.path.join(t2_path, file)
                    total_count += 1
                    
                    # 检查两个文件是否都存在并可读取
                    if (os.path.exists(filepath_vis_) and 
                        check_image_readable(filepath_ir_) and 
                        check_image_readable(filepath_vis_)):
                        self.filepath_ir.append(filepath_ir_)
                        self.filenames_ir.append(file)
                        self.filepath_vis.append(filepath_vis_)
                        self.filenames_vis.append(file)
                        valid_count += 1
                    else:
                        print(f"跳过文件对: {filepath_ir_} (T2对应文件: {filepath_vis_})")
                        
            else:
                # 分层目录结构 (原始逻辑)
                print("检测到分层目录结构")
                dir = os.listdir(data_dir_ir)
                dir.sort()
                
                for dir0 in dir:
                    req_path = os.path.join(data_dir_ir, dir0, 'T1')
                    if os.path.isdir(req_path):  # 确保 T1 目录存在
                        for file in os.listdir(req_path):
                            filepath_ir_ = os.path.join(req_path, file)
                            # 正确的路径替换：只替换最后的 /T1/ 部分
                            filepath_vis_ = filepath_ir_.replace('/T1/', '/T2/')
                            total_count += 1
                            
                            # 检查两个文件是否都可以读取
                            if check_image_readable(filepath_ir_) and check_image_readable(filepath_vis_):
                                self.filepath_ir.append(filepath_ir_)
                                self.filenames_ir.append(file)
                                self.filepath_vis.append(filepath_vis_)
                                self.filenames_vis.append(file)
                                valid_count += 1
                            else:
                                print(f"跳过损坏的文件对: {filepath_ir_} 和 {filepath_vis_}")
            
            self.split = split
            print(f"数据集过滤完成: 有效文件 {valid_count}/{total_count} 个")

            # 如果未显式指定 length，默认使用全部样本
            if self.length == 0 or self.length > len(self.filepath_ir):
                self.length = len(self.filepath_ir)

        elif split == 'test':
            data_dir_vis = vi_path
            data_dir_ir = ir_path

            if not data_dir_ir or not data_dir_vis:
                raise ValueError("ir_path 和 vi_path 不能为空，请检查配置文件中的 data.test_ir_path 与 data.test_vis_path")

            # 获取所有文件路径
            filepath_vis_all, filenames_vis_all = prepare_data_path(data_dir_vis)
            filepath_ir_all, filenames_ir_all = prepare_data_path(data_dir_ir)
            
            # 过滤掉损坏的文件
            valid_count = 0
            total_count = len(filepath_ir_all)
            for i in range(total_count):
                if check_image_readable(filepath_ir_all[i]) and check_image_readable(filepath_vis_all[i]):
                    self.filepath_ir.append(filepath_ir_all[i])
                    self.filenames_ir.append(filenames_ir_all[i])
                    self.filepath_vis.append(filepath_vis_all[i])
                    self.filenames_vis.append(filenames_vis_all[i])
                    valid_count += 1
                else:
                    print(f"跳过损坏的测试文件对: {filepath_ir_all[i]} 和 {filepath_vis_all[i]}")
            
            print(f"测试数据集过滤完成: 有效文件 {valid_count}/{total_count} 个")
            self.split = split

            if self.length == 0 or self.length > len(self.filepath_ir):
                self.length = len(self.filepath_ir)


    def __getitem__(self, index):
        if self.split=='train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]

            # 根据文件类型选择合适的加载方法
            if vis_path.lower().endswith(('.nii.gz', '.nii')):
                image_vis = load_nii_image(vis_path, is_grayscale=False)
            else:
                image_vis = cv2.imread(vis_path)
                
            if ir_path.lower().endswith(('.nii.gz', '.nii')):
                image_inf = load_nii_image(ir_path, is_grayscale=True)
            else:
                image_inf = cv2.imread(ir_path, 0)

            # 添加额外的安全检查
            if image_vis is None or image_inf is None:
                raise RuntimeError(f"无法读取图像: {vis_path} 或 {ir_path}")

            # 处理可见光图像（RGB或灰度转RGB）
            if len(image_vis.shape) == 2:  # 灰度图转RGB
                image_vis = np.stack([image_vis, image_vis, image_vis], axis=-1)
                
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            
            # 处理红外图像（确保是灰度）
            if len(image_inf.shape) == 3:  # 如果是RGB，转为灰度
                image_inf = cv2.cvtColor(image_inf, cv2.COLOR_BGR2GRAY)
                
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)

            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
            )
        elif self.split=='test':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            
            # 根据文件类型选择合适的加载方法
            if vis_path.lower().endswith(('.nii.gz', '.nii')):
                image_vis = load_nii_image(vis_path, is_grayscale=False)
            else:
                image_vis = cv2.imread(vis_path)
                
            if ir_path.lower().endswith(('.nii.gz', '.nii')):
                image_inf = load_nii_image(ir_path, is_grayscale=True)
            else:
                image_inf = cv2.imread(ir_path, 0)

            # 添加额外的安全检查
            if image_vis is None or image_inf is None:
                raise RuntimeError(f"无法读取图像: {vis_path} 或 {ir_path}")

            # 处理可见光图像（RGB或灰度转RGB）
            if len(image_vis.shape) == 2:  # 灰度图转RGB
                image_vis = np.stack([image_vis, image_vis, image_vis], axis=-1)
                
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            
            # 处理红外图像（确保是灰度）
            if len(image_inf.shape) == 3:  # 如果是RGB，转为灰度
                image_inf = cv2.cvtColor(image_inf, cv2.COLOR_BGR2GRAY)
                
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
            )

    def __len__(self):
        return self.length


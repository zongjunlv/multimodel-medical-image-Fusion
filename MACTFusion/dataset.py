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


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.png"))
    # data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    # data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    # data.extend(glob.glob((os.path.join(data_dir, "*.bmp"))))
    data.sort()
    filenames.sort()
    return data, filenames


def check_image_readable(image_path):
    """检查图像文件是否可读"""
    try:
        img = cv2.imread(image_path)
        return img is not None
    except:
        return False


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
                raise FileNotFoundError(f"训练 CT 路径不存在: {data_dir_ir}")

            dir = os.listdir(data_dir_ir)
            dir.sort()
            '''
            data_dir_ir/
                ├─ CT-MRI/
                   │   ├─ CT/xxx1.png
                   │   └─ MRI/xxx1.png
            代码通过两层 os.listdir 遍历，只读取 CT 文件夹；对应的 MRI 文件路径使用 str.replace('CT','MRI') 得到。
            vis 与 ir 必须同名、并位于平行文件夹    
            '''
            valid_count = 0
            total_count = 0
            for dir0 in dir:
                req_path = os.path.join(data_dir_ir, dir0, 'CT')
                if os.path.isdir(req_path):  # 确保 CT 目录存在
                    for file in os.listdir(req_path):
                        filepath_ir_ = os.path.join(req_path, file)
                        # 正确的路径替换：只替换最后的 /CT/ 部分
                        filepath_vis_ = filepath_ir_.replace('/CT/', '/MRI/')
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

            # 现在所有文件都已经在初始化时验证过可读性，直接读取即可
            image_vis = cv2.imread(vis_path)
            image_inf = cv2.imread(ir_path, 0)

            # 添加额外的安全检查（理论上不应该触发）
            if image_vis is None or image_inf is None:
                raise RuntimeError(f"无法读取图像: {vis_path} 或 {ir_path}")

            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
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
            
            # 现在所有文件都已经在初始化时验证过可读性，直接读取即可
            image_vis = cv2.imread(vis_path)
            image_inf = cv2.imread(ir_path, 0)

            # 添加额外的安全检查（理论上不应该触发）
            if image_vis is None or image_inf is None:
                raise RuntimeError(f"无法读取图像: {vis_path} 或 {ir_path}")

            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
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


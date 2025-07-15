import h5py

with h5py.File('Train_data/train_T1T2_size80.h5', 'r') as file:
    dataset = file['data1'][:]  # 替换为实际的数据集名称
    print(len(dataset))

# import os
#
# file_path = 'Train_data/train_T1T2_size80.h5'
# if not os.path.exists(file_path):
#     print(f"文件不存在: {file_path}")
# else:
#     print("文件存在")

# import SimpleITK as sitk
# import numpy as np
#
# def read_img(img_path):
#     return sitk.GetArrayFromImage(sitk.ReadImage(img_path))
#
# # 读取图像
# img_array = read_img('source_images/T1/4BLC_MR4.nii.gz')
# img_array2 = read_img('Train_data/BraTS2021_00621_t1.nii.gz')
#
# # 检查图像形状
# print("图像形状:", img_array.shape)
# print("图像形状:", img_array2.shape)
#
# # 检查数据类型
# print("数据类型:", img_array.dtype)
#
# # 检查值范围
# print("最小值:", np.min(img_array))
# print("最大值:", np.max(img_array))
#
# # 检查是否有 NaN 值
# print("是否有 NaN 值:", np.isnan(img_array).any())
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import h5py

f = h5py.File('../Dataset/Medical_Dataset.h5', 'r')
# for key in f.keys():
#   print(f[key].name)
a = f['data'][:]
sources = np.transpose(a, (0, 3, 2, 1))

vis = sources[100, :, :, 0]
ir = sources[100, :, :, 1]

ir_ds = scipy.ndimage.zoom(ir, 0.25)
ir_ds_us = scipy.ndimage.zoom(ir_ds, 4, order = 3)

fig = plt.figure()
V = fig.add_subplot(221)
I = fig.add_subplot(222)
I_ds = fig.add_subplot(223)
I_ds_us = fig.add_subplot(224)

V.imshow(vis, cmap = 'gray')
I.imshow(ir, cmap = 'gray')
I_ds.imshow(ir_ds, cmap = 'gray')
I_ds_us.imshow(ir_ds_us, cmap = 'gray')
plt.show()
# 打开HDF5文件：使用 h5py.File 打开名为 Medical_Dataset.h5 的文件，并以只读模式 'r' 打开。
# 读取数据：从文件中读取 data 数据集，并将其存储在变量 a 中。
# 转置数据：将数据 a 的维度进行转置，使其形状变为 (样本数, 通道数, 高度, 宽度)。
# 提取图像：从转置后的数据中提取第100个样本的可见光图像 vis 和红外图像 ir。
# 图像缩放：对红外图像 ir 进行缩小（缩放因子为0.25），得到 ir_ds；然后对 ir_ds 进行放大（缩放因子为4），得到 ir_ds_us。
# 绘制图像：使用 matplotlib 创建一个图形窗口，并在其中绘制原始可见光图像、原始红外图像、缩小后的红外图像和放大后的红外图像
# print
# 'Resampled by a factor of 2 with nearest interpolation:'
# print
# scipy.ndimage.zoom(x, 2, order = 0)
#
# print
# 'Resampled by a factor of 2 with bilinear interpolation:'
# print
# scipy.ndimage.zoom(x, 2, order = 1)
#
# print
# 'Resampled by a factor of 2 with cubic interpolation:'
# print
# scipy.ndimage.zoom(x, 2, order = 3)
#
# print
# 'Downsampled by a factor of 0.5 with default interpolation:'
# print(scipy.ndimage.zoom(x, 0.5))

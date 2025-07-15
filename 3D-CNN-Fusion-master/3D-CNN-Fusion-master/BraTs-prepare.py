import glob
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

t1 = glob.glob('source_images/T1/*.nii.gz')
t2 = glob.glob('source_images/T2/*.nii.gz')
# flair = glob.glob('./MICCAI_BraTS_2019_Data_Training/*/*/*flair.nii.gz')
# t1ce = glob.glob('./MICCAI_BraTS_2019_Data_Training/*/*/*t1ce.nii.gz')
# seg = glob.glob('./MICCAI_BraTS_2019_Data_Training/*/*/*seg.nii.gz')

def read_img(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def make_data(data1, data2):
    save_path = os.path.join('Train_data', 'train_T1T2_size80.h5')  # 目标路径
    # 确保 Train_data 目录存在
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))  # 创建 ./Train_data 目录
    # 写入文件
    with h5py.File(save_path, 'w') as hf:
        hf.create_dataset('data1', data=data1)
        hf.create_dataset('data2', data=data2)


# 将数据进行归一化处理，医学图像数据原始像素通常在0-2047之间
def normalize_2047(image):
    image = image.astype(dtype=np.float32)
    image = image/2047
    image[image > 1.0] = 1.0
    return image

# 对整个3D体积归一化（而非单张切片）
def normalize_3d_volume(volume):
    q_min = np.quantile(volume, 0.02)
    q_max = np.quantile(volume, 0.98)
    return (volume - q_min) / (q_max - q_min)

def main():
    sub_input1_sequence = []
    sub_input2_sequence = []

    count = 0
    image_size = 80

    # train_size = int(len(t1) * 0.8)
    train_size = int(10)
    test_size = int(len(t1) * 0.2)

    for i in range(train_size):
        img1 = (read_img(t1[i])).astype(np.float32)
        img2 = (read_img(t2[i])).astype(np.float32)
        input_1 = img1
        input_2 = img2
        '''
        ma = np.max(np.max(np.max(input_)))
        mi = np.min(np.min(np.min(input_)))
        input_ = (input_ - mi) / (ma - mi)  # [0,1]%%%%%%%%%%%%%%%%%
        '''
        #**********   normalization
        input_1 = normalize_2047(input_1)
        input_2 = normalize_2047(input_2)

        d, h, w = input_1.shape # 20,224,224
        # input_1 = input_1.reshape([d, h, w, 1])
        # sub_input1_sequence.append(input_1)
        # input_2 = input_2.reshape([d, h, w, 1])
        # sub_input2_sequence.append(input_2)
        # 为了增加训练数据的规模，训练集中的多模态体积被裁剪成大小为 20 × 80 × 80 的块。
        # 三个维度数据裁剪的步长分别设置为 20、50 和 50。
        for x in range(20, h - 20 - image_size + 1, image_size):
            for y in range(20, w - 20 - image_size + 1, image_size):
                sub_input1 = input_1[0:d, x:x + image_size, y:y + image_size]
                sub_input1 = sub_input1.reshape([d, image_size, image_size, 1])
                sub_input1_sequence.append(sub_input1)
                # 无法reshape成(80,80,80,1)，原数据集shape是（155，240，240），我的数据集是（20，224，224）

                sub_input2 = input_2[0:d, x:x + image_size, y:y + image_size]
                sub_input2 = sub_input2.reshape([d, image_size, image_size, 1])
                sub_input2_sequence.append(sub_input2)
                count = count + 1
        # 为了增加训练数据的规模，训练集中的多模态体积被裁剪成大小为 80 × 80 × 80 的块。
        # 三个维度数据裁剪的步长分别设置为 30、50 和 50。
        # for z in range(20, d - 20 - image_size, 30):
        #     for x in range(20, h - 20 - image_size + 1, 50):
        #         for y in range(20, w - 20 - image_size + 1, 50):
        #             sub_input1 = input_1[z:z + image_size, x:x + image_size, y:y + image_size]
        #             sub_input1 = sub_input1.reshape([image_size, image_size, image_size, 1])
        #             sub_input1 = sub_input1.reshape([image_size, image_size, image_size, 1])
        #             sub_input1_sequence.append(sub_input1)
        #
        #             # sub_input2 = input_2[z:z + image_size, x:x + image_size, y:y + image_size]
        #             # sub_input2 = sub_input2.reshape([image_size, image_size, image_size, 1])
        #             sub_input2_sequence.append(sub_input2)
        #             count = count + 1


    print("count:", count)  # count：244
    # Make list to numpy array. With this transform
    arrdata1 = np.asarray(sub_input1_sequence, dtype='float32')
    arrdata2 = np.asarray(sub_input2_sequence, dtype='float32')
    print(arrdata1.shape, arrdata2.shape)   # (4880, 20, 50, 50, 1) (4880, 20, 50, 50, 1)
    make_data(arrdata1, arrdata2)

if __name__ == '__main__':
    main()
# Use a trained DenseFuse Net to generate fused images

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave
from datetime import datetime
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
import time


def generate(ir_path, vis_path, model_path, index, output_path=None):
    # 读取图像并归一化
    ir_img = imread(ir_path) / 255.0
    vis_img = imread(vis_path) / 255.0
    # 获取图像维度
    ir_dimension = list(ir_img.shape)
    vis_dimension = list(vis_img.shape)
    # 调整图像维度。在图像维度的最前面插入 1，表示批量大小为 1。在图像维度的最后面插入 1，表示图像的通道数为 1
    ir_dimension.insert(0, 1)
    ir_dimension.append(1)
    vis_dimension.insert(0, 1)
    vis_dimension.append(1)
    # 将图像数据重塑为新的维度，使其符合 [1, height, width, 1] 的格式。
    ir_img = ir_img.reshape(ir_dimension)
    vis_img = vis_img.reshape(vis_dimension)

    # 批量大小：插入 1 表示批量大小为 1，即每次输入模型的图像只有一个样本。这是因为在生成融合图像时，通常一次只处理一张图像。
    # 通道数：插入 1 表示图像的通道数为 1，适用于灰度图像。这对于模型的输入层来说是必要的，因为模型需要知道每个像素的通道信息

    # 创建TensorFlow图和会话：
    # 使用tf.Graph().as_default()创建一个新的TensorFlow计算图，并将其设置为默认图。使用tf.Session()创建一个新的TensorFlow会话。
    with tf.Graph().as_default(), tf.Session() as sess:
        # 定义占位符：
        # 定义两个占位符SOURCE_VIS和SOURCE_ir，分别用于存储可见光图像和红外图像的数据。这些占位符的形状与之前调整后的图像维度一致。
        SOURCE_VIS = tf.placeholder(tf.float32, shape=vis_dimension, name='SOURCE_VIS')
        SOURCE_ir = tf.placeholder(tf.float32, shape=ir_dimension, name='SOURCE_ir')

        # 创建生成器并生成输出图像：
        # 实例化一个生成器G，并调用其transform方法，传入可见光图像和红外图像的占位符，生成输出图像output_image。
        G = Generator('Generator')
        output_image = G.transform(vis=SOURCE_VIS, ir=SOURCE_ir)
        # D1 = Discriminator1('Discriminator1')
        # D2 = Discriminator2('Discriminator2')

        # 恢复预训练模型并运行生成过程：
        # 使用tf.train.Saver()创建一个保存器对象。调用saver.restore(sess, model_path)恢复预训练的模型。
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        # 使用sess.run(output_image, feed_dict={SOURCE_VIS: vis_img, SOURCE_ir: ir_img})运行生成过程，得到输出图像。
        output = sess.run(output_image, feed_dict={SOURCE_VIS: vis_img, SOURCE_ir: ir_img})
        # 提取输出图像的第一个样本，并保存为PNG文件
        output = output[0, :, :, 0]
        imsave(output_path + str(index) + '.png', output)

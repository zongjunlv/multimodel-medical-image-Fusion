import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
from Deconv import deconv_vis, deconv_ir

WEIGHT_INIT_STDDEV = 0.05


class Generator(object):

    def __init__(self, sco):
        self.encoder = Encoder(sco)
        self.decoder = Decoder(sco)

    def transform(self, vis, ir):
        # 反卷积操作（也称为转置卷积）通常用于将特征图的尺寸放大。
        # 步幅为 4 意味着在高度和宽度方向上每一步移动 4 个像素，使得输出特征图高度和宽度扩大4倍。
        IR = deconv_ir(ir, strides=[1, 4, 4, 1], scope_name='deconv_ir')
        VIS = deconv_vis(vis, strides=[1, 1, 1, 1], scope_name='deconv_vis')
        # 将 IR 和 VIS 在通道维度上进行拼接
        img = tf.concat([VIS, IR], 3)
        # 将特征编码到特征向量code里，再解码到图像。
        code = self.encoder.encode(img)
        self.target_features = code
        generated_img = self.decoder.decode(self.target_features)
        return generated_img


class Encoder(object):
    def __init__(self, scope_name):
        self.scope = scope_name
        self.weight_vars = []
        with tf.variable_scope(self.scope):
            with tf.variable_scope('encoder'):
                self.weight_vars.append(self._create_variables(2, 48, 3, scope='conv1_1'))
                self.weight_vars.append(self._create_variables(48, 48, 3, scope='dense_block_conv1'))
                self.weight_vars.append(self._create_variables(96, 48, 3, scope='dense_block_conv2'))
                self.weight_vars.append(self._create_variables(144, 48, 3, scope='dense_block_conv3'))
                self.weight_vars.append(self._create_variables(192, 48, 3, scope='dense_block_conv4'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        # 根据输入通道数 input_filters、输出通道数 output_filters 和卷积核大小 kernel_size 计算权重矩阵的形状 shape
        shape = [kernel_size, kernel_size, input_filters, output_filters]
        # 使用 tf.variable_scope 创建一个变量作用域，确保变量名称唯一。
        with tf.variable_scope(scope):
            # 初始化权重：使用截断正态分布 tf.truncated_normal 初始化权重矩阵 kernel，标准差为 WEIGHT_INIT_STDDEV。
            kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV),
                                 name='kernel')
            bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)

    def encode(self, image):
        dense_indices = [1, 2, 3, 4, 5]
        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]
            if i in dense_indices:
                out = conv2d(out, kernel, bias, dense=True, use_relu=True,
                             Scope=self.scope + '/encoder/b' + str(i))
            else:
                out = conv2d(out, kernel, bias, dense=False, use_relu=True,
                             Scope=self.scope + '/encoder/b' + str(i))
        return out


class Decoder(object):
    def __init__(self, scope_name):
        self.weight_vars = []
        self.scope = scope_name
        with tf.name_scope(scope_name):
            with tf.variable_scope('decoder'):
                self.weight_vars.append(self._create_variables(240, 240, 3, scope='conv2_1'))
                self.weight_vars.append(self._create_variables(240, 128, 3, scope='conv2_1'))
                self.weight_vars.append(self._create_variables(128, 64, 3, scope='conv2_2'))
                self.weight_vars.append(self._create_variables(64, 32, 3, scope='conv2_3'))
                self.weight_vars.append(self._create_variables(32, 1, 3, scope='conv2_4'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.variable_scope(scope):
            shape = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
            bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)

    def decode(self, image):
        final_layer_idx = len(self.weight_vars) - 1

        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]
            if i == 0:
                out = conv2d(out, kernel, bias, dense=False, use_relu=True,
                             Scope=self.scope + '/decoder/b' + str(i), BN=False)
            if i == final_layer_idx:
                out = conv2d(out, kernel, bias, dense=False, use_relu=False,
                             Scope=self.scope + '/decoder/b' + str(i), BN=False)
                out = tf.nn.tanh(out) / 2 + 0.5
            else:
                out = conv2d(out, kernel, bias, dense=False, use_relu=True, BN=True,
                             Scope=self.scope + '/decoder/b' + str(i))
        return out


def conv2d(x, kernel, bias, dense=False, use_relu=True, Scope=None, BN=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)
    if BN:
        with tf.variable_scope(Scope):
            out = tf.layers.batch_normalization(out, training=True)
    if use_relu:
        out = tf.nn.relu(out)
    if dense:
        out = tf.concat([out, x], 3)
    return out

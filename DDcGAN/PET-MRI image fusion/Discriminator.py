import tensorflow as tf

WEIGHT_INIT_STDDEV = 0.1


class Discriminator1(object):
    def __init__(self, scope_name):
        self.weight_vars = []
        self.scope = scope_name
        with tf.variable_scope(scope_name):
            self.weight_vars.append(self._create_variables(1, 16, 3, scope='conv1'))
            self.weight_vars.append(self._create_variables(16, 32, 3, scope='conv2'))
            self.weight_vars.append(self._create_variables(32, 64, 3, scope='conv3'))
            # 初始化weight_vars列表，用于存储权重变量。
            # 设置scope属性为传入的scope_name。
            # 使用tf.variable_scope创建一个变量作用域，确保变量在指定的作用域内创建。
            # 在该作用域内，通过调用_create_variables方法创建三个卷积层的权重变量，并将它们添加到weight_vars列表中。

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        shape = [kernel_size, kernel_size, input_filters, output_filters]
        with tf.variable_scope(scope):
            kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
            bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return kernel, bias

    # 这段代码定义了一个名为`_create_variables`的方法，用于创建卷积层的权重和偏置变量。具体功能如下：
    # 1. ** 输入参数 **：
    # - `input_filters`: 输入通道数
    # - `output_filters`: 输出通道数
    # - `kernel_size`: 卷积核的大小
    # - `scope`: 变量作用域名称
    # 2. ** 主要步骤 **：
    # - 计算权重矩阵的形状`shape`，其维度为[kernel_size, kernel_size, input_filters, output_filters]`。
    # - 使用tf.variable_scope`创建一个变量作用域。
    # - 在该作用域内，使用`tf.truncated_normal`初始化权重矩阵。`kernel`，标准差为`WEIGHT_INIT_STDDEV`。
    # - 使用`tf.zeros`初始化偏置向量`bias`。
    # - 返回权重矩阵`kernel`和偏置向量`bias`。

    def discrim(self, img, reuse):
        conv_num = len(self.weight_vars)
        # 图片处理
        if len(img.shape) != 4:
            img = tf.expand_dims(img, -1)
        out = img
        # 对于每一层使用Conv2d_1函数进行卷积操作，第一层不用BN，其余层均使用。
        for i in range(conv_num):
            kernel, bias = self.weight_vars[i]
            if i == 0:
                out = conv2d_1(out, kernel, bias, [1, 2, 2, 1], use_relu=True, use_BN=False,
                               Scope=self.scope + '/b' + str(i), Reuse=reuse)
            else:
                out = conv2d_1(out, kernel, bias, [1, 2, 2, 1], use_relu=True, use_BN=True,
                               Scope=self.scope + '/b' + str(i), Reuse=reuse)
        out = tf.reshape(out, [-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])
        with tf.variable_scope(self.scope):
            with tf.variable_scope('flatten1'):
                out = tf.layers.dense(out, 1, activation=tf.nn.tanh, use_bias=True, trainable=True,
                                      reuse=reuse)
        out = out / 2 + 0.5
        # out是在[-1,1]区间内，/2 + 0.5调整到[0,1]区间内
        return out
    # 输入处理：首先检查输入图像 img 的形状，如果其维度不是4，则通过 tf.expand_dims 将其扩展为4维。
    # 卷积层循环：根据 self.weight_vars 的长度确定卷积层数量。对于每一层，使用 conv2d_1 函数进行卷积操作。
    #           第一层不使用批量归一化（Batch Normalization），其余层均使用。
    # 展平操作：将卷积后的特征图展平为一维向量。
    # 全连接层：通过一个全连接层将展平后的向量映射为一个标量，并使用 tanh 激活函数。
    # 输出调整：将输出值调整到 [0, 1] 区间内


def conv2d_1(x, kernel, bias, strides, use_relu=True, use_BN=True, Scope=None, Reuse=None):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides, padding='VALID')
    out = tf.nn.bias_add(out, bias)
    if use_BN:
        with tf.variable_scope(Scope):
            out = tf.layers.batch_normalization(out, training=True, reuse=Reuse)
    if use_relu:
        out = tf.nn.relu(out)
    return out
    # 图像填充：使用反射模式对输入图像进行填充，以确保卷积操作后图像尺寸的变化符合预期。
    # 卷积操作：对填充后的图像执行卷积操作，并添加偏置项。
    # 批归一化：如果 use_BN 参数为 True，则在当前作用域内应用批归一化。
    # ReLU激活：如果 use_relu 参数为 True，则应用ReLU激活函数。
    # 返回结果：返回最终的输出。


class Discriminator2(object):
    def __init__(self, scope_name):
        self.weight_vars = []
        self.scope = scope_name
        with tf.variable_scope(scope_name):
            self.weight_vars.append(self._create_variables(1, 16, 3, scope='conv1'))
            self.weight_vars.append(self._create_variables(16, 32, 3, scope='conv2'))
            self.weight_vars.append(self._create_variables(32, 64, 3, scope='conv3'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        shape = [kernel_size, kernel_size, input_filters, output_filters]
        with tf.variable_scope(scope):
            kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
            bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)

    def discrim(self, img, reuse):
        conv_num = len(self.weight_vars)
        if len(img.shape) != 4:
            img = tf.expand_dims(img, -1)
        out = img
        for i in range(conv_num):
            kernel, bias = self.weight_vars[i]
            if i == 0:
                out = conv2d_2(out, kernel, bias, [1, 2, 2, 1], use_relu=True, use_BN=False,
                               Scope=self.scope + '/b' + str(i), Reuse=reuse)
            else:
                out = conv2d_2(out, kernel, bias, [1, 2, 2, 1], use_relu=True, use_BN=True,
                               Scope=self.scope + '/b' + str(i), Reuse=reuse)
        out = tf.reshape(out, [-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])
        with tf.variable_scope(self.scope):
            with tf.variable_scope('flatten1'):
                out = tf.layers.dense(out, 1, activation=tf.nn.tanh, use_bias=True, trainable=True,
                                      reuse=reuse)
        out = out / 2 + 0.5
        return out


def conv2d_2(x, kernel, bias, strides, use_relu=True, use_BN=True, Scope=None, Reuse=None):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides, padding='VALID')
    out = tf.nn.bias_add(out, bias)
    if use_BN:
        with tf.variable_scope(Scope):
            out = tf.layers.batch_normalization(out, training=True, reuse=Reuse)
    if use_relu:
        out = tf.nn.relu(out)
    return out

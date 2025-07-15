import tensorflow as tf

WEIGHT_INIT_STDDEV = 0.1


def deconv_ir(input, strides, scope_name):
    weight_vars = []
    scope = ['deconv1']
    with tf.variable_scope('Generator'):
        with tf.variable_scope(scope_name):
            weight_vars.append(_create_variables(1, 1, 3, scope=scope[0]))
        # weight_vars.append(_create_variables(1, 1, 3, scope = scope[1]))
    deconv_num = len(weight_vars)
    out = input
    for i in range(deconv_num):
        input_shape = out.shape
        kernel = weight_vars[i]
        out = tf.nn.conv2d_transpose(out, filter=kernel, output_shape=[int(input_shape[0]), int(input_shape[1]) * 4,
                                                                       int(input_shape[2]) * 4,
                                                                       int(input_shape[3])],
                                     # output_shape的四个参数：批次大小、高度、宽度、通道数。
                                     strides=strides, padding='SAME')
    return out


# 初始化变量：创建一个空列表 weight_vars 用于存储权重变量，定义一个包含单个字符串 'deconv1' 的列表 scope。
# 创建权重变量：在 tf.variable_scope('Generator') 和 tf.variable_scope(scope_name) 的作用域下，
# 调用 _create_variables 函数创建一个权重变量，并将其添加到 weight_vars 列表中。
# 执行反卷积操作：遍历 weight_vars列表，对输入 input 执行反卷积操作。
# 每次迭代中，计算输出形状并使用 tf.nn.conv2d_transpose 函数进行反卷积操作，
# 最终返回处理后的输出 out。

def deconv_vis(input, strides, scope_name):
    weight_vars = []
    scope = ['deconv1']
    with tf.variable_scope('Generator'):
        with tf.variable_scope(scope_name):
            weight_vars.append(_create_variables(1, 1, 3, scope=scope[0]))
        # weight_vars.append(_create_variables(1, 1, 3, scope = scope[1]))
    deconv_num = len(weight_vars)
    out = input
    for i in range(deconv_num):
        input_shape = out.shape
        kernel = weight_vars[i]
        out = tf.nn.conv2d_transpose(out, filter=kernel, output_shape=[int(input_shape[0]), int(input_shape[1]),
                                                                       int(input_shape[2]), int(input_shape[3])],
                                     strides=strides, padding='SAME')
    return out


def _create_variables(input_filters, output_filters, kernel_size, scope):
    shape = [kernel_size, kernel_size, output_filters, input_filters]
    with tf.variable_scope(scope):
        kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
    # bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
    return kernel  # , bias)

    # 计算权重变量的形状 shape，其格式为 [kernel_size, kernel_size, output_filters, input_filters]。
    # 使用 tf.variable_scope 创建一个变量作用域，确保变量名称的唯一性。
    # 在该作用域内，使用 tf.truncated_normal 初始化一个权重变量 kernel，标准差为 WEIGHT_INIT_STDDEV。
    # 返回创建的权重变量 kernel。

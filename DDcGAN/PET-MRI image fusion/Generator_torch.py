import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderWeights(nn.Module):
    def __init__(self):
        super(EncoderWeights, self).__init__()
        self.weight_vars = nn.ParameterList()  # 用于存储所有可训练的参数
        self._create_variables()

    def _create_variables(self):
        # 创建权重和偏置，并添加到参数列表中
        self.weight_vars.append(self._add_conv_params(2, 48, 3))  # conv1_1
        self.weight_vars.append(self._add_conv_params(48, 48, 3))  # dense_block_conv1
        self.weight_vars.append(self._add_conv_params(96, 48, 3))  # dense_block_conv2
        self.weight_vars.append(self._add_conv_params(144, 48, 3))  # dense_block_conv3
        self.weight_vars.append(self._add_conv_params(192, 48, 3))  # dense_block_conv4

    def _add_conv_params(self, in_channels, out_channels, kernel_size):
        # 创建卷积层权重和偏置
        kernel_shape = (out_channels, in_channels, kernel_size, kernel_size)
        kernel = nn.Parameter(torch.empty(kernel_shape))
        bias = nn.Parameter(torch.zeros(out_channels))

        # 初始化权重 (等效于 truncated_normal 初始化)
        nn.init.kaiming_normal_(kernel, mode='fan_out', nonlinearity='relu')

        return (kernel, bias)

    def conv2d(self, x, kernel, bias, dense=False, use_relu=True):
        # Perform 2D convolution
        padding = kernel.size(2) // 2  # Assuming same padding
        out = F.conv2d(x, kernel, bias, stride=1, padding=padding)

        # Add dense connection if `dense=True`
        if dense:
            out = torch.cat([x, out], dim=1)

        # Apply ReLU if `use_relu=True`
        if use_relu:
            out = F.relu(out)
        return out

    def encode(self, image):
        dense_indices = [1, 2, 3, 4, 5]  # Indices for dense layers
        out = image

        for i, (kernel, bias) in enumerate(self.weight_vars):
            dense = (i in dense_indices)
            out = self.conv2d(out, kernel, bias, dense=dense, use_relu=True)

        return out


class VariableCreator:
    def __init__(self, weight_init_stddev):
        self.weight_init_stddev = weight_init_stddev

    def create_variables(self, input_filters, output_filters, kernel_size):
        # Define the shape of the kernel and bias
        kernel_shape = (output_filters, input_filters, kernel_size,
                        kernel_size)  # PyTorch uses (out_channels, in_channels, kernel_height, kernel_width)
        bias_shape = (output_filters,)

        # Initialize kernel and bias
        kernel = nn.Parameter(torch.empty(kernel_shape))  # Create trainable tensor
        bias = nn.Parameter(torch.zeros(bias_shape))  # Initialize bias to zeros

        # Initialize kernel with truncated normal distribution
        nn.init.normal_(kernel, mean=0.0, std=self.weight_init_stddev)
        return kernel, bias

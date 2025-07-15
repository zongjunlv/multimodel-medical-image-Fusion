import torch
import torch.nn.functional as F


def gaussian_window(size, sigma):
    """生成高斯核"""
    coords = torch.arange(size).float() - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]  # 生成二维高斯核
    return window


def ssim_loss(img1, img2, size=11, sigma=1.5):
    """
    计算 SSIM 损失
    :param img1: 输入图像 1 (B, C, H, W)
    :param img2: 输入图像 2 (B, C, H, W)
    :param size: 高斯窗口大小
    :param sigma: 高斯窗口标准差
    :return: SSIM 值（越接近 1，图像越相似）
    """
    # 初始化高斯核
    window = gaussian_window(size, sigma).to(img1.device)  # 确保在相同设备
    window = window.view(1, 1, size, size)  # 将窗口扩展为 (1, 1, size, size)

    # 参数设置
    K1, K2, L = 0.01, 0.03, 1.0  # 假定像素范围 [0, 1]
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # 计算均值
    mu1 = F.conv2d(img1, window, stride=1, padding=0, groups=1)
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=1)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, window, stride=1, padding=0, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, stride=1, padding=0, groups=1) - mu1_mu2

    # SSIM 公式
    value = ((2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)).mean()
    return value


# 示例测试
if __name__ == "__main__":
    # 假设输入图像的形状为 (batch_size, channels, height, width)
    img1 = torch.rand(1, 1, 256, 256)  # 范围 [0, 1]
    img2 = torch.rand(1, 1, 256, 256)  # 范围 [0, 1]

    ssim = ssim_loss(img1, img2, size=11, sigma=1.5)
    print(f"SSIM Loss: {ssim.item()}")

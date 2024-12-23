import torch
import torch.nn as nn
from .snn_layers import *
from .fsvae_prior import *
from .fsvae_posterior import *
import torch.nn.functional as F

import global_v as glv

# 12.13
from .snn_layers import MembraneOutputLayer

# 12.12
# 新增损失函数
from boundary_loss import boundary_loss
import math


class SAE(nn.Module):
    def __init__(self, device, boundary_weight):  # 新增lambda_l2参数来控制L2正则化权重
        super().__init__()
        super().__init__()
        # out_channels = in_channels
        # in_channels = in_channels

        in_channels = glv.network_config['in_channels']  # 确保从配置中获取正确的输入通道数
        latent_dim = glv.network_config['latent_dim']

        self.latent_dim = latent_dim
        # self.n_steps = n_steps
        self.n_steps = glv.network_config['n_steps']
        self.device = device
        self.boundary_weight = boundary_weight  # 新增 Boundary Loss 权重

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        """
        新增代码段如下一行
        """
        # 添加边缘提取模块
        #  self.edge_extractor = SobelEdgeExtractionModule(device=device, in_channels=glv.network_config['in_channels'])

        # ... 构建编码器和解码器 ...

        self.membrane_output_layer = MembraneOutputLayer(self.n_steps)  # 使用 self.n_steps

        self.psp = PSP()

        # Build Encoder
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                       out_channels=h_dim,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       bias=True,
                       bn=tdBatchNorm(h_dim),
                       spike=LIFSpike(),
                       is_first_conv=is_first_conv)
            )
            in_channels = h_dim  # 每次卷积后，in_channels 被更新为当前层的 h_dim
            is_first_conv = False

        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1] * 4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        # Build Decoder
        modules = []

        self.decoder_input = tdLinear(latent_dim,
                                      hidden_dims[-1] * 4,
                                      bias=True,
                                      bn=tdBatchNorm(hidden_dims[-1] * 4),
                                      spike=LIFSpike())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                tdConvTranspose(hidden_dims[i],
                                hidden_dims[i + 1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                                bias=True,
                                bn=tdBatchNorm(hidden_dims[i + 1]),
                                spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            tdConvTranspose(hidden_dims[-1],
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                            bias=True,
                            bn=tdBatchNorm(hidden_dims[-1]),
                            spike=LIFSpike()),
            tdConvTranspose(hidden_dims[-1],
                            out_channels=1,  # 这里最后一层的输出应该改为1通道
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            bn=None,
                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer(self.n_steps)

        self.psp = PSP()

    def forward(self, x, scheduled=False):

        latent = self.encode(x)
        x_recon = self.decode(latent)  # [N, 1, H, W]
        return x_recon, latent

    def encode(self, x):
        x = self.encoder(x)  # (N,C,H,W,T)
        x = torch.flatten(x, start_dim=1, end_dim=3)  # (N,C*H*W,T)
        latent_x = self.before_latent_layer(x)  # (N,latent_dim,T)
        return latent_x

    def decode(self, z):
        result = self.decoder_input(z)  # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps)  # (N,C,H,W,T)
        result = self.decoder(result)  # (N,C,H,W,T)
        result = self.final_layer(result)  # (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))
        return out

    # # 添加L2正则化项限制模型复杂度，防止过拟合
    # def l2_regularization(self):
    #     l2_loss = 0
    #     # 遍历所有参数，计算L2范数
    #     for param in self.parameters():
    #         if param.requires_grad:
    #             l2_loss += torch.sum(param ** 2)
    #     return l2_loss

    def loss_function(self, edge_img, recons_img):
        # 计算边缘重建损失（均方误差）
        # 假设 edge_img 和 recons_img 的通道数一致，或者需要调整
        edge_loss = F.mse_loss(recons_img, edge_img)

        # 计算 SSIM 损失
        # ssim_loss = 1 - self.ssim(recons_img, edge_img)

        # 计算 Boundary Loss
        b_loss = boundary_loss(recons_img, edge_img)

        # # 计算L2正则化损失
        # l2_loss = self.l2_regularization()

        # 检查 edge_loss 是否包含 NaN
        if torch.isnan(edge_loss):
            print("Edge reconstruction loss contains NaN")

        # 原总损失：loss = edge_loss + self.distance_lambda * mmd_loss

        # 组合新总损失
        loss = edge_loss + self.boundary_weight * b_loss
        # + self.lambda_l2 * l2_loss

        # 检查总损失是否包含 NaN
        if torch.isnan(loss):
            print("Total loss contains NaN")

        # return {"loss": loss, "EdgeReconstruction_Loss": edge_loss}
        # return {'loss': loss, 'EdgeReconstruction_Loss': edge_loss, 'Distance_Loss': mmd_loss}
        # 返回多个损失项以供记录
        return {
            'loss': loss,
            'EdgeReconstruction_Loss': edge_loss,
            'Boundary_Loss': b_loss,
        }

    # def l2_regularization(self):
    #     l2_loss = 0
    #     # 遍历所有参数，计算L2范数
    #     for param in self.parameters():
    #         if param.requires_grad:
    #             l2_loss += torch.sum(param ** 2)
    #     return l2_loss

    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4, 4)

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p - init_p) * epoch / max_epoch + init_p


class SobelEdgeExtractionModule(nn.Module):
    def __init__(self, device, in_channels=3):
        super(SobelEdgeExtractionModule, self).__init__()
        self.device = device

        # 定义 Sobel 核
        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]], dtype=torch.float32)

        # 扩展维度并重复以匹配输入通道数
        self.sobel_kernel_x = sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1).to(device)
        self.sobel_kernel_y = sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1).to(device)

    def forward(self, x):
        # x 的形状应为 [N, C, H, W]
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")

        # 计算 Sobel 卷积
        edge_x = F.conv2d(x, self.sobel_kernel_x, padding=1, groups=x.size(1))
        edge_y = F.conv2d(x, self.sobel_kernel_y, padding=1, groups=x.size(1))

        # 计算梯度幅值
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)  # 添加小常数以避免除以零

        # 跨通道平均，得到单通道边缘图
        edges = torch.mean(edges, dim=1, keepdim=True)

        return edges


# 使用Canny算法进行边缘提取
class CannyEdgeDetectionModule(nn.Module):
    def __init__(self, device, in_channels=3, low_threshold=0.4, high_threshold=0.6):
        super(CannyEdgeDetectionModule, self).__init__()
        self.device = device
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.in_channels = in_channels

        # Sobel Kernels for gradient computation
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]], dtype=torch.float32).to(device)

        self.sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                            [0, 0, 0],
                                            [1, 2, 1]], dtype=torch.float32).to(device)

        # Gaussian kernel for smoothing (example 5x5 kernel)
        self.gaussian_kernel = self.get_gaussian_kernel(5, 1.0).to(device)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")

        # Step 1: Convert RGB to grayscale (for simplicity, use weighted average)
        x_gray = self.rgb_to_grayscale(x)

        # Step 2: Apply Gaussian Blur
        x_blurred = self.apply_gaussian_blur(x_gray)

        # Step 3: Compute gradients using Sobel kernels
        grad_x = F.conv2d(x_blurred, self.sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1), padding=1)
        grad_y = F.conv2d(x_blurred, self.sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1), padding=1)

        # Step 4: Compute edge magnitude and direction
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_direction = torch.atan2(grad_y, grad_x)

        # Step 5: Non-Maximum Suppression (NMS)
        edges = self.non_maximum_suppression(edge_magnitude, edge_direction)

        # Step 6: Edge Tracking by Hysteresis
        edges = self.edge_tracking_by_hysteresis(edges)

        return edges

    def rgb_to_grayscale(self, x):
        """Convert RGB image to grayscale using weighted sum, or pass through if already grayscale."""
        if x.size(1) == 3:
            # Use the formula: Gray = 0.299 * R + 0.587 * G + 0.114 * B
            r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray.unsqueeze(1)  # Add channel dimension (1 channel)
        elif x.size(1) == 1:
            return x  # Already grayscale
        else:
            raise ValueError(f"Expected input with 1 or 3 channels, but got {x.size(1)} channels.")

    def get_gaussian_kernel(self, size, sigma):
        """Generate a Gaussian kernel."""
        kernel = torch.zeros((size, size), dtype=torch.float32, device=self.device)
        mean = size // 2
        sum_val = 0
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device=self.device)  # Make sure sigma is a tensor

        for i in range(size):
            for j in range(size):
                # Ensure dist_sq is a tensor
                dist_sq = torch.tensor((i - mean) ** 2 + (j - mean) ** 2, dtype=torch.float32, device=self.device)
                kernel[i, j] = torch.exp(-dist_sq / (2 * sigma_tensor ** 2))  # Use tensor for the calculation
                sum_val += kernel[i, j]

        kernel /= sum_val  # Normalize the kernel
        return kernel

    def apply_gaussian_blur(self, x):
        """Apply Gaussian Blur to the image."""
        # Using the 5x5 Gaussian kernel (for simplicity)
        return F.conv2d(x, self.gaussian_kernel.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1), padding=2)

    def non_maximum_suppression(self, magnitude, direction):
        """Perform Non-Maximum Suppression (NMS)."""
        # Note: The implementation here is quite simplified; NMS requires more complex handling for edge directions.
        # This function would suppress non-maximum gradients.
        return magnitude

    def edge_tracking_by_hysteresis(self, edges):
        """Perform edge tracking by hysteresis."""
        # Simple thresholding example (for real-world application, this would be more sophisticated)
        edges = torch.where(edges > self.high_threshold, torch.ones_like(edges), torch.zeros_like(edges))
        return edges

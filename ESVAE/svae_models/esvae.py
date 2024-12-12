import torch
import torch.nn as nn
from .snn_layers import *
from .fsvae_prior import *
from .fsvae_posterior import *
import torch.nn.functional as F

import global_v as glv
# 12.11日新增
# import pytorch_ssim  # 确保已安装 pytorch-ssim 库
from boundary_loss import boundary_loss

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class SampledSpikeAct(torch.autograd.Function):
    """
        Implementation of the spiking activation function with an approximation of gradient.
    """

    aa = 0.5  # 定义一个合适的值，根据需要调整

    @staticmethod
    def forward(ctx, input):
        random_sign = torch.rand_like(input, dtype=input.dtype).to(input.device)
        ctx.save_for_backward(input, random_sign)
        # if input = u > Vth then output = 1
        output = torch.gt(input, random_sign)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, random_sign = ctx.saved_tensors
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        hu = abs(input - random_sign) < aa
        hu = hu.float() / (2 * aa)
        return grad_input * hu


class ESVAE(nn.Module):
    def __init__(self, device, distance_lambda, mmd_type, boundary_weight):    # 此处增添了一个参数
        super().__init__()

        in_channels = glv.network_config['in_channels']  # 确保从配置中获取正确的输入通道数
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.device = device
        self.distance_lambda = distance_lambda
        self.mmd_type = mmd_type

        self.boundary_weight = boundary_weight  # 新增 Boundary Loss 权重

        # 12.11日新增模块
        # self.ssim_weight = ssim_weight  # 新增 SSIM 损失权重
        # self.ssim = pytorch_ssim.SSIM().to(device)  # 初始化 SSIM 模块

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        """
        新增代码段如下一行
        """
        # 添加边缘提取模块
        self.edge_extractor = SobelEdgeExtractionModule(device=device, in_channels=glv.network_config['in_channels'])

        # Build Encoder
        # 构建一个包含卷积层、全连接层和变分推断的编码器部分
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:  # 遍历隐藏层维度每个元素，逐层添加卷积层
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

        self.encoder = nn.Sequential(*modules)  # 通过 nn.Sequential 将所有的卷积层模块组合成一个序列，组成编码器 encoder
        """
        全连接层（tdLinear），接收来自卷积网络的输出，并将其映射到潜在空间（latent space）。
        """
        self.before_latent_layer = tdLinear(hidden_dims[-1] * 4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k)  # 定义先验分布

        self.posterior = PosteriorBernoulliSTBP(self.k)  # 定义后验分布

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
                            out_channels=1,  # 这里最后一层的输出应该改为1通道,源代码：glv.network_config['in_channels']
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            bn=None,
                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()

        self.sample_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )  # 瓶颈层

        self.mmd_loss = MMD_loss(kernel_type=self.mmd_type)

    def forward(self, x, scheduled=False):
        sampled_z_q, r_q, r_p = self.encode(x, scheduled)
        x_recon = self.decode(sampled_z_q)

        # # 新增模块
        # # 我们需要移除时间步维度以进行边缘提取
        # original_x = x.mean(dim=-1)  # 聚合时间步维度，将 [N, C, H, W, T] 转换为 [N, C, H, W]
        # # 进行边缘提取，只考虑原始输入图像
        # edge = self.edge_extractor(original_x)  # 提取边缘图像

        # 检查 x_recon 是否包含 NaNs
        if torch.isnan(x_recon).any():
            print("Reconstructed image contains NaN")

        return x_recon, r_q, r_p, sampled_z_q

    """
    此处有新增代码模块，第200和第201行
    """

    def encode(self, x, scheduled=False, return_firing_rate=False):
        x = self.encoder(x)  # (N,C,H,W,T)
        x = torch.flatten(x, start_dim=1, end_dim=3)  # (N,C*H*W,T)
        latent_x = self.before_latent_layer(x)  # (N,latent_dim,T)    # 在这里通过编码得到了x_e

        sampled_z_q, r_q, r_p = self.gaussian_sample(latent_x, latent_x.shape[0])
        return sampled_z_q, r_q, r_p  # 这里进行了点火率的计算

    def decode(self, z):
        result = self.decoder_input(z)  # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps)  # (N,C,H,W,T)
        result = self.decoder(result)  # (N,C,H,W,T)
        result = self.final_layer(result)  # (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))
        return out

    def sample(self, batch_size=64):
        sampled_z_p, _, _ = self.gaussian_sample(batch_size=batch_size)
        sampled_x = self.decode(sampled_z_p)
        return sampled_x, sampled_z_p  # 返回生成样本和潜在变量

    def gaussian_sample(self, latent_x=None, batch_size=None, mu=None, var=None):
        """
        泊松尖峰采样的核心部分
        """
        if latent_x is not None:
            sampled_z_n = torch.randn((batch_size, self.latent_dim)).to(self.device)  # (N, latent_dim)
            r_p = self.sample_layer(sampled_z_n)  # 经过瓶颈层生成点火率

            r_q = latent_x.mean(-1, keepdim=True).repeat((1, 1, self.n_steps))
            sampled_z_q = SampledSpikeAct.apply(r_q)
            # 泊松尖峰采样，SampledSpikeAct，自定义操作，用于生成[0,1]序列，能否找到其他更好的生成[0,1]序列的方法
            # 这里可以进行优化

            r_q = latent_x.mean(-1)  # (N, latent_dim)    # 这一行就是直接计算 latent_x 的平均点火率的地方

            return sampled_z_q, r_q, r_p  # 在这里会涉及到隐变量的输出
        else:
            sampled_z_n = torch.randn((batch_size, self.latent_dim)).to(self.device)
            # if mu is None and var is None:
            #     mu = self.mu
            #     var = self.var
            # var = var * torch.ones_like(sampled_p).to(self.device)  # (N, latent_dim)
            # mu = mu * torch.ones_like(sampled_p).to(self.device)  # (N, latent_dim)
            # sampled_p = mu + sampled_z_n * var
            r_p = self.sample_layer(sampled_z_n)
            r_p = r_p.unsqueeze(dim=-1).repeat(
                (1, 1, self.n_steps))  # 求T时间步点火率的均值，并拓展回时间步维度，如（0.2， 0.5）————>(0.35, 0.35)
            sampled_z_q = SampledSpikeAct.apply(r_p)
            return sampled_z_q, None, None  # z_q解码得到重构图像

    """
    此处有代码段的修改
    """

    def loss_function_mmd(self, edge_img,
                          recons_img, r_q,
                          r_p):  # 参数列表由 input_image ---> edge_img （self, edge_img, recons_img, r_q, r_p）原参数
        """
        r_q is q(z|x): (N,latent_dim)
        r_p is p(z): (N,latent_dim)
        """
        # 此处先把重建损失注释掉，只计算边缘重建损失
        # recons_loss = F.mse_loss(recons_img, input_img)
        print(r_p.shape, r_p.max(), r_p.min(), r_p.mean())
        print(r_q.shape, r_q.max(), r_q.min(), r_q.mean())
        # 距离损失，如下
        mmd_loss = self.mmd_loss(r_q, r_p)

        # 计算边缘重建损失（均方误差）
        # 假设 edge_img 和 recons_img 的通道数一致，或者需要调整
        edge_loss = F.mse_loss(recons_img, edge_img)

        # 计算 SSIM 损失
        # ssim_loss = 1 - self.ssim(recons_img, edge_img)
        # 计算 Boundary Loss
        b_loss = boundary_loss(recons_img, edge_img)

        # 检查 edge_loss 是否包含 NaN
        if torch.isnan(edge_loss):
            print("Edge reconstruction loss contains NaN")

        # 原总损失：loss = edge_loss + self.distance_lambda * mmd_loss

        # 组合新总损失
        loss = edge_loss + self.distance_lambda * mmd_loss + self.boundary_weight * b_loss

        # 检查总损失是否包含 NaN
        if torch.isnan(loss):
            print("Total loss contains NaN")

        # return {"loss": loss, "EdgeReconstruction_Loss": edge_loss}
        # return {'loss': loss, 'EdgeReconstruction_Loss': edge_loss, 'Distance_Loss': mmd_loss}
        # 返回多个损失项以供记录
        return {
            'loss': loss,
            'EdgeReconstruction_Loss': edge_loss,
            'Distance_Loss': mmd_loss,
            'Boundary_Loss': b_loss
        }

    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4, 4)

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p - init_p) * epoch / max_epoch + init_p


"""
ESVAELarge 类实现了一个深度变分自编码器（VAE）模型，结合了卷积神经网络（CNN）进行编码和解码，
以及突触整合（PSP）和 MMD 损失用于生成任务。模型的目标是生成从潜在空间采样的图像，
同时保持潜在空间分布与先验分布的一致性。
通过多层卷积和反卷积（转置卷积）网络结构，模型能够处理较为复杂的生成任务。
"""


class ESVAELarge(ESVAE):
    def __init__(self, device, distance_lambda, mmd_type):
        super(ESVAELarge, self).__init__(device, distance_lambda, mmd_type)
        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.device = device
        self.distance_lambda = distance_lambda
        self.mmd_type = mmd_type

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                       out_channels=h_dim,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       bias=True,
                       bn=tdBatchNorm(h_dim),
                       spike=LIFSpike())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1] * 4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k)

        self.posterior = PosteriorBernoulliSTBP(self.k)

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
                            out_channels=1,  # 这里最后一层的输出应该改为1通道，原代码glv.network_config['in_channels']
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            bn=None,
                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()

        self.sample_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )

        self.gaussian_mmd_loss = MMD_loss(kernel_type=self.mmd_type)

        print(self)


# 新增一个边缘提取模块
import torch
import torch.nn as nn
import torch.nn.functional as F


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



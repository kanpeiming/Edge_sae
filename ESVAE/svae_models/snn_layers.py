import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import global_v as glv

dt = 5
a = 0.25
aa = 0.5  
Vth = 0.2
tau = 0.25

class SpikeAct(torch.autograd.Function):
    """ 
        Implementation of the spiking activation function with an approximation of gradient.
        替代梯度
        这是一个自定义的尖峰激活函数类，用于神经网络。它使用梯度的一个近似来进行反向传播。forward 方法会在输入大于某个阈值 Vth 时输出 1，否则输出 0。backward 方法使用一个近似的函数来计算梯度。
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, Vth) 
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors 
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        hu = abs(input) < aa
        hu = hu.float() / (2 * aa)
        return grad_input * hu

class LIFSpike(nn.Module):
    """
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU.
        The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
        基于LIF模块生成尖峰。它可以被视为一种激活函数，并且其使用方式类似于ReLU。输入张量需要有一个额外的时间维度，在这种情况下，它位于数据的最后一个维度上。
        这个类实现了漏积分点火（Leaky Integrate-and-Fire, LIF）模型，用于生成尖峰信号。它可以作为神经网络中的一个激活函数。该类在输入张量的每个时间步更新神经元的状态（膜电位和尖峰输出）。
    """
    def __init__(self):
        super(LIFSpike, self).__init__()

    def forward(self, x):
        nsteps = x.shape[-1]
        u   = torch.zeros(x.shape[:-1] , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(nsteps):
            u, out[..., step] = self.state_update(u, out[..., max(step-1, 0)], x[..., step])
        return out
    
    def state_update(self, u_t_n1, o_t_n1, W_mul_o_t1_n, tau=tau):
        u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
        o_t1_n1 = SpikeAct.apply(u_t1_n1)
        return u_t1_n1, o_t1_n1

class MySpikeAct(SpikeAct):
    """
        Implementation of the spiking activation function with an approximation of gradient.
        实现带有梯度近似的尖峰激活函数。
        SpikeAct 的子类，看起来是一个更具体的尖峰激活函数版本，它在前向传播时接受一个额外的参数 Vth
    """
    @staticmethod
    def forward(ctx, input):
        data, Vth = input
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(data, Vth)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        data, Vth, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        hu = abs(data) < aa
        hu = hu.float() / (2 * aa)
        return grad_input * hu


class MyLIFSpike(LIFSpike):
    """
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU.
        The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
        LIFSpike 的子类，重写了 forward 和 state_update 方法，以包含阈值 Vth 作为生成尖峰的参数，基于膜电位进行判断。
    """

    def __init__(self):
        super(MyLIFSpike, self).__init__()

    def forward(self, input):
        x, Vth = input
        nsteps = x.shape[-1]
        u   = torch.zeros(x.shape[:-1] , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(nsteps):
            u, out[..., step] = self.state_update(u, out[..., max(step-1, 0)], x[..., step], Vth)
        return out

    def state_update(self, u_t_n1, o_t_n1, W_mul_o_t1_n, Vth, tau=tau):
        u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
        o_t1_n1 = MySpikeAct.apply([u_t1_n1, Vth])
        return u_t1_n1, o_t1_n1



class tdLinear(nn.Linear):
    """
    torch.nn.Linear 的子类，扩展了典型的线性层以包括可选的批量归一化（bn）和尖峰激活函数（spike）。
    为了处理带有时序维度的数据而设计.
    """
    def __init__(self, 
                in_features,
                out_features,
                bias=True,
                bn=None,
                spike=None):
        assert type(in_features) == int, 'inFeatures should not be more than 1 dimesnion. It was: {}'.format(in_features.shape)
        assert type(out_features) == int, 'outFeatures should not be more than 1 dimesnion. It was: {}'.format(out_features.shape)

        super(tdLinear, self).__init__(in_features, out_features, bias=bias)

        self.bn = bn
        self.spike = spike
        

    def forward(self, x):
        """
        x : (N,C,T)
        """        
        x = x.transpose(1, 2) # (N, T, C)
        y = F.linear(x, self.weight, self.bias)
        y = y.transpose(1, 2)# (N, C, T)
        
        if self.bn is not None:
            y = y[:,:,None,None,:]
            y = self.bn(y)
            y = y[:,:,0,0,:]
        if self.spike is not None:
            y = self.spike(y)
        return y

class tdConv(nn.Conv3d):
    """
    torch.nn.Conv3d 的子类，同样包括可选的批量归一化和尖峰激活函数。与 tdLinear 类似，它也适用于带有时序维度的数据。
    """
    def __init__(self, 
                in_channels, 
                out_channels,  
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                bn=None,
                spike=None,
                is_first_conv=False):

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        super(tdConv, self).__init__(in_channels, out_channels, kernel, stride, padding, dilation, groups,
                                        bias=bias)
        self.bn = bn
        self.spike = spike
        self.is_first_conv = is_first_conv

    def forward(self, x):
        x = F.conv3d(x, self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
        if self.bn is not None:
            x = self.bn(x)
        if self.spike is not None:
            x = self.spike(x)
        return x
        

class tdConvTranspose(nn.ConvTranspose3d):
    """
    torch.nn.ConvTranspose3d 的子类，提供了与 tdConv 类似的可选批量归一化和尖峰激活函数
    """
    def __init__(self, 
                in_channels, 
                out_channels,  
                kernel_size,
                stride=1,
                padding=0,
                output_padding=0,
                dilation=1,
                groups=1,
                bias=True,
                bn=None,
                spike=None):

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))


        # output padding
        if type(output_padding) == int:
            output_padding = (output_padding, output_padding, 0)
        elif len(output_padding) == 2:
            output_padding = (output_padding[0], output_padding[1], 0)
        else:
            raise Exception('output_padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        super().__init__(in_channels, out_channels, kernel, stride, padding, output_padding, groups,
                                        bias=bias, dilation=dilation)

        self.bn = bn
        self.spike = spike

    def forward(self, x):
        x = F.conv_transpose3d(x, self.weight, self.bias,
                        self.stride, self.padding, 
                        self.output_padding, self.groups, self.dilation)

        if self.bn is not None:
            x = self.bn(x)
        if self.spike is not None:
            x = self.spike(x)
        return x

class tdBatchNorm(nn.BatchNorm2d):
    """
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
        实现时间域批量归一化（tdBN）。相关论文链接：https://arxiv.org/pdf/2011.05280。简而言之，在进行批量归一化时，它也会在时间域上进行平均。
        参数：
        - num_features (int)：与 nn.BatchNorm2d 相同
        - eps (float)：与 nn.BatchNorm2d 相同
        - momentum (float)：与 nn.BatchNorm2d 相同
        - alpha (float)：一个额外的参数，可能在残差块中发生变化。
        - affine (bool)：与 nn.BatchNorm2d 相同
        - track_running_stats (bool)：与 nn.BatchNorm2d 相同
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * Vth * (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]
        
        return input


class PSP(torch.nn.Module):
    """
    这个类实现了一种突触整合的形式，可能是突触短期可塑性的某种形式，其中突触响应随时间以时间常数 tau_s 进行积分
    这个 PSP 类模拟了一种突触短期可塑性（STP）现象，突触响应随着时间逐步改变，每次输入信号到来时都会根据当前突触响应调整，且调整速度由时间常数 tau_s 控制。
    """
    def __init__(self):
        super().__init__()
        self.tau_s = 2

    def forward(self, inputs):
        """
        inputs: (N, C, T)
        """
        syns = None
        syn = 0
        n_steps = inputs.shape[-1]
        for t in range(n_steps):
            syn = syn + (inputs[...,t] - syn) / self.tau_s
            if syns is None:
                syns = syn.unsqueeze(-1)
            else:
                syns = torch.cat([syns, syn.unsqueeze(-1)], dim=-1)

        return syns

class MembraneOutputLayer(nn.Module):
    """
    outputs the last time membrane potential of the LIF neuron with V_th=infty
    这个类输出LIF神经元的最后一个时间步的膜电位，模拟无限阈值（V_th=infinity）。它通过对时间维度上的输入张量进行加权求和来实现。
    """
    def __init__(self, n_steps=None) -> None:
        super().__init__()
        if n_steps is None:
            n_steps = glv.n_steps

        arr = torch.arange(n_steps-1,-1,-1)
        self.register_buffer("coef", torch.pow(0.8, arr)[None,None,None,None,:]) # (1,1,1,1,T)

    def forward(self, x):
        """
        x : (N,C,H,W,T)
        """
        out = torch.sum(x*self.coef, dim=-1)
        return out
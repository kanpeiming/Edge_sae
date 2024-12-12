import os
import os.path
import random
import numpy as np
import logging
import argparse
import pycuda.driver as cuda

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

import global_v as glv
from network_parser import parse
from datasets import load_dataset_snn
from utils import aboutCudaDevices
from utils import AverageMeter
from utils import CountMulAddSNN
# 21行有更改
import svae_models.edge_sae as edge_sae
from svae_models.snn_layers import LIFSpike

# 三个评价指标不再需要
# import metrics.inception_score as inception_score
# import metrics.clean_fid as clean_fid
# import metrics.autoencoder_fid as autoencoder_fid
from svae_models.edge_sae import SobelEdgeExtractionModule
# 新增
from torch.cuda.amp import autocast, GradScaler
# 12.12修改，ESVAE模型不再需要,导入SAE模型
# from svae_models.esvae import ESVAE
from svae_models.edge_sae import SAE
from boundary_loss import boundary_loss

max_accuracy = 0
min_loss = 1000


def add_hook(net):
    count_mul_add = CountMulAddSNN()
    hook_handles = []
    for m in net.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear) or isinstance(m,
                                                                                          torch.nn.ConvTranspose3d) or isinstance(
            m, LIFSpike):
            handle = m.register_forward_hook(count_mul_add)
            hook_handles.append(handle)
    return count_mul_add, hook_handles


def write_weight_hist(net, index):
    for n, m in net.named_parameters():
        root, name = os.path.splitext(n)
        writer.add_histogram(root + '/' + name, m, index)


def train(network, trainloader, opti, epoch):
    n_steps = glv.network_config['n_steps']
    max_epoch = glv.network_config['epochs']

    loss_meter = AverageMeter()
    recons_meter = AverageMeter()
    # dist_meter = AverageMeter()

    boundary_meter = AverageMeter()  # 新增 Boundary Loss 记录器

    # mean_r_q = 0
    # mean_r_p = 0
    # mean_sampled_z_q = 0

    # 初始化边缘提取模块
    edge_extractor = SobelEdgeExtractionModule(device=network.module.device,
                                               in_channels=glv.network_config['in_channels'])
    # edge_extractor = EdgeExtractionModule(device=network.device, in_channels=glv.network_config['in_channels'])

    scaler = GradScaler()  # 初始化 GradScaler

    network = network.train()

    for batch_idx, (real_img, labels) in enumerate(trainloader):

        # 检查 real_img 是否包含 NaN
        if torch.isnan(real_img).any():
            print(f"Batch {batch_idx}: real_img contains NaN")
            continue  # 跳过这个批次

        opti.zero_grad()
        real_img = real_img.to(init_device, non_blocking=True)
        labels = labels.to(init_device, non_blocking=True)

        # 提取边缘图像
        edge_img = edge_extractor(real_img)  # [N, 1, H, W]

        # 修改后：
        spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps).to(init_device)  # 确保输入张量在正确的设备上
        # spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # (N, C, H, W, T)

        with torch.amp.autocast('cuda'):  # 使用 autocast 进行混合精度
            # 只保留x_recon，去掉r_q, r_p的计算
            # x_recon, _, _, sampled_z_q = network.module(spike_input, scheduled=glv.network_config['scheduled'])
            # losses = network.module.loss_function_mmd(edge_img, x_recon)  # 不需要r_q和r_p了
            # loss = losses['loss']
            x_recon, _, _, _ = network.module(spike_input, scheduled=glv.network_config['scheduled'])
            losses = network.module.loss_function(edge_img, x_recon)
            loss = losses['loss']

        # 检查 loss 是否包含 NaN
        if torch.isnan(loss):
            print(f"Batch {batch_idx}: Loss is NaN")
            continue  # 跳过这个批次

        scaler.scale(loss).backward()

        # 梯度裁剪
        scaler.unscale_(opti)
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

        scaler.step(opti)
        scaler.update()

        loss_meter.update(losses['loss'].detach().cpu().item())
        recons_meter.update(
            losses['EdgeReconstruction_Loss'].detach().cpu().item())  # Reconstruction_Loss --> Edge reconstruction loss
        # dist_meter.update(losses['Distance_Loss'].detach().cpu().item())
        boundary_meter.update(losses['Boundary_Loss'].detach().cpu().item())  # 更新 Boundary Loss

        # mean_r_q = (r_q.mean(0).detach().cpu() + batch_idx * mean_r_q) / (batch_idx + 1)  # (latent_dim)
        # mean_r_p = (r_p.mean(0).detach().cpu() + batch_idx * mean_r_p) / (batch_idx + 1)  # (latent_dim)
        # mean_sampled_z_q = (sampled_z_q.mean(0).detach().cpu() + batch_idx * mean_sampled_z_q) / (
        #         batch_idx + 1)  # (C,T)

        print(
            f'Train[{epoch}/{max_epoch}] [{batch_idx}/{len(trainloader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, BOUNDARY: {boundary_meter.avg}')  # 新增, SSIM: {ssim_meter.avg:.7f}

        # 在第一轮时保存重构图像和边缘图像
        # 在第一轮和最后一轮，且是第一个批次时保存图像
        if (epoch == 0 or epoch == max_epoch - 1 or epoch == 20) and batch_idx == 0:  #
            os.makedirs(f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/imgs/train/', exist_ok=True)

            # 保存原始输入图像
            torchvision.utils.save_image((real_img + 1) / 2,
                                         f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/imgs/train/epoch{epoch}_input.png')

            # 保存解码后的重构图像
            torchvision.utils.save_image((x_recon + 1) / 2,
                                         f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/imgs/train/epoch{epoch}_recons.png')

            # 保存边缘提取图像
            torchvision.utils.save_image((edge_img + 1) / 2,
                                         f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/imgs/train/epoch{epoch}_edge.png')

            # 将这些图像记录到 TensorBoard
            writer.add_images('Train/input_img', (real_img + 1) / 2, epoch)
            writer.add_images('Train/recons_img', (x_recon + 1) / 2, epoch)
            writer.add_images('Train/edge_img', (edge_img + 1) / 2, epoch)

        # if batch_idx == len(trainloader) - 1:
        #     os.makedirs(f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/imgs/train/', exist_ok=True)
        #     torchvision.utils.save_image((real_img + 1) / 2,
        #                                  f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/imgs/train/epoch{epoch}_input.png')
        #     torchvision.utils.save_image((x_recon + 1) / 2,
        #                                  f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/imgs/train/epoch{epoch}_recons.png')
        #     writer.add_images('Train/input_img', (real_img + 1) / 2, epoch)
        #     writer.add_images('Train/recons_img', (x_recon + 1) / 2, epoch)

        # 清理缓存
        torch.cuda.empty_cache()

        # break

    logging.info(
        f"Train [{epoch}] Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} BOUNDARY: {boundary_meter.avg}")  # 删除DISTANCE: {dist_meter.avg}
    writer.add_scalar('Train/loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/recons_loss', recons_meter.avg, epoch)
    # writer.add_scalar('Train/distance', dist_meter.avg, epoch)
    writer.add_scalar('Train/BOUNDARY', boundary_meter.avg, epoch)
    # writer.add_scalar('Train/mean_r_q', mean_r_q.mean().item(), epoch)
    # writer.add_scalar('Train/mean_r_p', mean_r_p.mean().item(), epoch)

    # writer.add_image('Train/mean_sampled_z_q', mean_sampled_z_q.unsqueeze(0), epoch)
    # writer.add_histogram(f'Train/mean_sampled_z_q_distribution', mean_sampled_z_q.sum(-1), epoch)

    return loss_meter.avg


def test(network, testloader, epoch):
    n_steps = glv.network_config['n_steps']
    max_epoch = glv.network_config['epochs']

    loss_meter = AverageMeter()
    recons_meter = AverageMeter()
    # dist_meter = AverageMeter()

    boundary_meter = AverageMeter()  # 新增 Boundary Loss 记录器

    mean_r_q = 0
    mean_r_p = 0
    mean_sampled_z_q = 0

    count_mul_add, hook_handles = add_hook(net)

    network = network.eval()
    with torch.no_grad():

        # 初始化边缘提取模块
        edge_extractor = SobelEdgeExtractionModule(device=network.module.device,
                                                   in_channels=glv.network_config['in_channels'])
        # edge_extractor = EdgeExtractionModule(device=network.device, in_channels=glv.network_config['in_channels'])
        for batch_idx, (real_img, labels) in enumerate(testloader):
            real_img = real_img.to(init_device, non_blocking=True)
            labels = labels.to(init_device, non_blocking=True)

            # 提取边缘图像
            edge_img = edge_extractor(real_img)  # [N, 1, H, W]

            # direct spike input
            # 修改后：
            spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps).to(init_device)  # 确保输入张量在正确的设备上
            # spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # (N,C,H,W,T)

            with torch.amp.autocast('cuda'):  # 使用混合精度
                # 只保留x_recon，去掉r_q, r_p的计算
                # x_recon, _, _, sampled_z_q = network.module(spike_input, scheduled=glv.network_config['scheduled'])
                # losses = network.module.loss_function_mmd(edge_img, x_recon)  # 不需要r_q和r_p了
                x_recon, _, _, _ = network.module(spike_input, scheduled=glv.network_config['scheduled'])
                losses = network.module.loss_function(edge_img, x_recon)

            # 检查 loss 是否包含 NaN
            if torch.isnan(losses['loss']):
                print(f"Batch {batch_idx}: Loss is NaN")
                continue  # 跳过这个批次

            # mean_r_q = (r_q.mean(0).detach().cpu() + batch_idx * mean_r_q) / (batch_idx + 1)  # (latent_dim)
            # mean_r_p = (r_p.mean(0).detach().cpu() + batch_idx * mean_r_p) / (batch_idx + 1)  # (latent_dim)
            # mean_sampled_z_q = (sampled_z_q.mean(0).detach().cpu() + batch_idx * mean_sampled_z_q) / (
            #         batch_idx + 1)  # (C,T)

            loss_meter.update(losses['loss'].detach().cpu().item())
            recons_meter.update(losses['EdgeReconstruction_Loss'].detach().cpu().item())  # Edge reconstruction loss
            # dist_meter.update(losses['Distance_Loss'].detach().cpu().item())
            boundary_meter.update(losses['Boundary_Loss'].detach().cpu().item())  # 更新 Boundary Loss

            print(
                f'Test[{epoch}/{max_epoch}] [{batch_idx}/{len(testloader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, BOUNDARY: {boundary_meter.avg}')  # 删除, DISTANCE: {dist_meter.avg}

            # 在第一轮和最后一轮，且是第一个批次时保存图像
            if (epoch == 0 or epoch == max_epoch - 1 or epoch == 20) and batch_idx == 0:  # max_epoch - 1
                os.makedirs(f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/imgs/test/', exist_ok=True)

                # 保存原始输入图像
                torchvision.utils.save_image((real_img + 1) / 2,
                                             f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/imgs/test/epoch{epoch}_input.png')

                # 保存解码后的重构图像
                torchvision.utils.save_image((x_recon + 1) / 2,
                                             f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/imgs/test/epoch{epoch}_recons.png')

                # 保存边缘提取图像
                torchvision.utils.save_image((edge_img + 1) / 2,
                                             f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/imgs/test/epoch{epoch}_edge.png')

                # 将这些图像记录到 TensorBoard
                writer.add_images('Test/input_img', (real_img + 1) / 2, epoch)
                writer.add_images('Test/recons_img', (x_recon + 1) / 2, epoch)
                writer.add_images('Test/edge_img', (edge_img + 1) / 2, epoch)

                # 清理缓存
            torch.cuda.empty_cache()

            # break

    logging.info(
        f"Test [{epoch}] Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} BOUNDARY: {boundary_meter.avg}")  # 删除 DISTANCE: {dist_meter.avg}
    writer.add_scalar('Test/loss', loss_meter.avg, epoch)
    writer.add_scalar('Test/recons_loss', recons_meter.avg, epoch)
    # writer.add_scalar('Test/distance', dist_meter.avg, epoch)
    writer.add_scalar('Test/BOUNDARY', boundary_meter.avg, epoch)
    # writer.add_scalar('Test/mean_r_q', mean_r_q.mean().item(), epoch)
    # writer.add_scalar('Test/mean_r_p', mean_r_p.mean().item(), epoch)
    writer.add_scalar('Test/mul', count_mul_add.mul_sum.item() / len(testloader), epoch)
    writer.add_scalar('Test/add', count_mul_add.add_sum.item() / len(testloader), epoch)

    for handle in hook_handles:
        handle.remove()

    # writer.add_image('Test/mean_sampled_z_q', mean_sampled_z_q.unsqueeze(0), epoch)
    # writer.add_histogram('Test/mean_sampled_z_q_distribution', mean_sampled_z_q.sum(-1), epoch)

    return loss_meter.avg


def seed_all(seed=42):
    """
    set random seed.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seed_all()

    # 初始化设备并选择多卡
    init_device = torch.device("cuda:2")  # 默认使用 cuda:0
    device_ids = [2, 3]  # 使用 GPU 2 和 GPU 3

    parser = argparse.ArgumentParser()  # 解析命令行参数
    parser.add_argument('-name', default='tmp', type=str)
    parser.add_argument('-config', action='store', dest='config', help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint',
                        help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-device', type=int)
    parser.add_argument('-project_save_path', default='C:/coding/ESVAE/results', type=str)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')

    if args.device is None:
        init_device = torch.device("cuda:2")
    else:
        init_device = torch.device(f"cuda:{args.device}")

    logging.info("start parsing settings")

    params = parse(args.config)
    network_config = params['Network']

    logging.info("finish parsing settings")
    logging.info(network_config)
    print(network_config)

    glv.init(network_config, devs=[2, 3])  # glv.init(network_config, [args.device])

    # distance_lambda = glv.network_config['distance_lambda']  # 已经在配置文件中定义了这个值
    # mmd_type = glv.network_config['mmd_type']  # 已经在配置文件中定义了这个值

    boundary_weight = glv.network_config['boundary_weight']  # 已经在配置文件中定义了这个值

    dataset_name = glv.network_config['dataset']
    data_path = glv.network_config['data_path']

    # 动态设置 in_channels
    if dataset_name in ["MNIST", "FashionMNIST"]:
        glv.network_config['in_channels'] = 1
    elif dataset_name in ["CIFAR10", "CelebA"]:
        glv.network_config['in_channels'] = 3
    else:
        raise Exception('Unrecognized dataset name.')

    lr = glv.network_config['lr']
    # sample_layer_lr_times = glv.network_config['sample_layer_lr_times']
    # distance_lambda = glv.network_config['distance_lambda']
    loss_func = glv.network_config['loss_func']
    # mmd_type = glv.network_config['mmd_type']
    latent_dim = glv.network_config['latent_dim']
    try:
        add_name = glv.network_config['add_name']
    except:
        add_name = None

    args.name = f'edge_sae_lr-{lr}_loss_func-{loss_func}-latent_dim-{latent_dim}'

    if add_name is not None:
        args.name = add_name + '-' + args.name

    os.makedirs(f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}', exist_ok=True)
    writer = SummaryWriter(log_dir=f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/tb')
    logging.basicConfig(filename=f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}.log',
                        level=logging.INFO)

    # Check whether a GPU is available
    if torch.cuda.is_available():
        cuda.init()
        c_device = aboutCudaDevices()
        print(c_device.info())
        print("selected device: ", args.device)
    else:
        raise Exception("only support gpu")

    logging.info("dataset loading...")
    if dataset_name == "MNIST":
        data_path = os.path.expanduser(data_path)
        train_loader, test_loader = load_dataset_snn.load_mnist(data_path)
    elif dataset_name == "FashionMNIST":
        data_path = os.path.expanduser(data_path)
        train_loader, test_loader = load_dataset_snn.load_fashionmnist(data_path)
    elif dataset_name == "CIFAR10":
        data_path = os.path.expanduser(data_path)
        train_loader, test_loader = load_dataset_snn.load_cifar10(data_path)
    elif dataset_name == "CelebA":
        data_path = os.path.expanduser(data_path)
        train_loader, test_loader = load_dataset_snn.load_celebA(data_path)
    else:
        raise Exception('Unrecognized dataset name.')
    logging.info("dataset loaded")

    if network_config['model'] == 'SAE':
        net = edge_sae.SAE(device=init_device,
                           boundary_weight=boundary_weight)  # 设置 Boundary 权重
    elif network_config['model'] == 'ESVAE_large':
        net = esvae.ESVAELarge(device=init_device, distance_lambda=distance_lambda, mmd_type=mmd_type)
    else:
        raise Exception('not defined model')

    # 使用 DataParallel 来支持多卡训练
    net = torch.nn.DataParallel(net, device_ids=[2, 3])  # 使用 GPU 2 和 GPU 3   新增的一行
    net = net.to(init_device)  # 将模型移动到默认设备（cuda:0）

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)

    params = list(net.named_parameters())
    param_group = [
        {'params': [p for n, p in params if 'sample_layer' in n], 'weight_decay': 0.001,
         'lr': lr * sample_layer_lr_times},
        {'params': [p for n, p in params if 'sample_layer' not in n], 'weight_decay': 0.001, 'lr': lr},
    ]

    optimizer = torch.optim.AdamW(param_group,
                                  lr=lr,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.001)  # 优化器

    best_loss = 1e8
    best_inception_score = 1e-8
    best_autoencoder_dist = 1e8
    best_fid = 1e8
    for e in range(glv.network_config['epochs']):

        write_weight_hist(net, e)
        if network_config['scheduled']:
            net.module.update_p(e, glv.network_config[
                'epochs'])  # 使用 net.module 来调用方法    # net.update_p(e, glv.network_config['epochs'])
            logging.info("update p")
        train_loss = train(net, train_loader, optimizer, e)
        test_loss = test(net, test_loader, e)

        torch.save(net.state_dict(), f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/checkpoint.pth')
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(net.state_dict(), f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/best.pth')

        # sample(net, e, batch_size=64)
        # in_score = calc_inception_score(net, e, batch_size=glv.network_config['sample_batch_size'])
        # autoencoder_dist = calc_autoencoder_frechet_distance(net, e)
        # fid = calc_clean_fid(net, e)

        # if in_score > best_inception_score:
        #     best_inception_score = in_score
        #     print(best_inception_score)
        #     torch.save({'net': net.state_dict(), 'epoch': e},
        #                f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/best_inception_score.pth')

        # if autoencoder_dist < best_autoencoder_dist:
        #     best_autoencoder_dist = autoencoder_dist
        #     torch.save({'net': net.state_dict(), 'epoch': e},
        #                f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/best_autoencoder_dist.pth')
        #
        # if fid < best_fid:
        #     best_fid = fid
        #     torch.save({'net': net.state_dict(), 'epoch': e},
        #                f'{args.project_save_path}/checkpoint/{dataset_name}/{args.name}/best_fid.pth')

    writer.close()

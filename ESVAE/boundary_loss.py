# boundary_loss.py

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2


def compute_distance_map(mask):
    """
    计算二值掩码的距离变换图。
    mask: Tensor, shape (N, 1, H, W), 值为0或1
    返回: Tensor, shape (N, 1, H, W)
    """
    mask_np = mask.cpu().numpy()
    distance_maps = []
    for i in range(mask_np.shape[0]):
        # 将掩码转换为uint8
        mask_uint8 = (mask_np[i, 0] > 0.5).astype(np.uint8) * 255
        # 计算距离变换
        distance = distance_transform_edt(1 - mask_uint8 / 255.0)
        # 归一化到[0,1]
        distance = distance / distance.max() if distance.max() > 0 else distance
        distance_maps.append(distance)
    distance_maps = np.stack(distance_maps, axis=0)
    distance_maps = torch.tensor(distance_maps, dtype=torch.float32).unsqueeze(1).to(mask.device)
    return distance_maps


def boundary_loss(pred, target, epsilon=1e-6):
    """
    计算 Boundary Loss。
    pred: Tensor, shape (N, 1, H, W), 预测的图像，范围[0,1]
    target: Tensor, shape (N, 1, H, W), 真实的图像，范围[0,1]
    epsilon: float, 防止除零错误
    返回: Tensor, 标量损失
    """
    # 将预测和目标转换为二值图
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()

    # 计算距离变换图
    pred_distance = compute_distance_map(pred_binary)
    target_distance = compute_distance_map(target_binary)

    # 计算损失
    loss = F.mse_loss(pred_distance, target_distance)

    return loss

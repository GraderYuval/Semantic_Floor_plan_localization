from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import *
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
from modules.semantic.semantic_mapper import ObjectType

def custom_loss_with_normalization(desdf, rays):
    """
    Computes a penalty-driven loss for mismatches, normalized by the size of predictions.
    Input:
        desdf: (H, W, O) actual data
        rays: (H, W, O) predicted data
    Output:
        loss: (H, W) loss per spatial location
    """
    # Boolean tensors for presence
    pred_window = rays[..., ObjectType.WINDOW.value] > 0
    pred_door = rays[..., ObjectType.DOOR.value] > 0
    pred_wall = rays[..., ObjectType.WALL.value] > 0
    
    actual_window = desdf[..., ObjectType.WINDOW.value] > 0
    actual_door = desdf[..., ObjectType.DOOR.value] > 0
    actual_wall = desdf[..., ObjectType.WALL.value] > 0
    
    # Penalties for false positives
    window_mismatch = pred_window & ~actual_window  # Predict window but no window in actual
    door_mismatch = pred_door & ~actual_door       # Predict door but no door in actual
    wall_mismatch = pred_wall & ~actual_wall       # Predict wall but no wall in actual
    
    # Count total predictions for normalization
    total_window_preds = pred_window.sum().float()
    total_door_preds = pred_door.sum().float()
    total_wall_preds = pred_wall.sum().float()
    
    # Avoid division by zero
    total_window_preds = torch.clamp(total_window_preds, min=1.0)
    total_door_preds = torch.clamp(total_door_preds, min=1.0)
    total_wall_preds = torch.clamp(total_wall_preds, min=1.0)
    
    # Assign penalties normalized by the total predictions
    window_penalty = (5.0 * total_window_preds) * window_mismatch.float()
    door_penalty = (3.0 * total_door_preds) * door_mismatch.float()
    wall_penalty = (1.0 / total_wall_preds) * wall_mismatch.float()
    
    # Combine penalties
    total_penalty = window_penalty + door_penalty + wall_penalty
    
    # Return negative exponential for probabilistic compatibility
    return -torch.exp(-total_penalty)


def localize(
    desdf: torch.tensor, rays: torch.tensor, orn_slice=36, return_np=True, lambd=40, localize_type = "depth"
) -> Tuple[torch.tensor]:
    """
    Localize in the desdf according to the rays
    Input:
        desdf: (H, W, O), counter clockwise
        rays: (V,) from left to right (clockwise)
        orn_slice: number of orientations
        return_np: return as ndarray instead of torch.tensor
        lambd: parameter for likelihood
    Output:
        prob_vol: probability volume (H, W, O), ndarray
        prob_dist: probability distribution, (H, W) maxpool the prob_vol along orientation, ndarray
        orientations: orientation with max likelihood at each position, (H, W), ndarray
        pred: (3, ) predicted state [x,y,theta], ndarray
    """

    # flip the ray, to make rotation direction mathematically positive
    rays = torch.flip(rays, [0])
    O = desdf.shape[2]
    V = rays.shape[0]
    # expand rays to have the same dimension as desdf
    rays = rays.reshape((1, 1, -1))

    # circular pad the desdf
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

    if localize_type == "depth":
        # probablility is -l1norm
        prob_vol = torch.stack(
            [
                -torch.norm(pad_desdf[:, :, i : i + V] - rays, p=1.0, dim=2)
                for i in range(O)
            ],
            dim=2,
        )  # (H,W,O)
        prob_vol = torch.exp(prob_vol / lambd)  # NOTE: here make prob positive
    else:
        prob_vol = torch.stack(
            [
                custom_loss_with_normalization(pad_desdf[:, :, i : i + V], rays)
                for i in range(O)
            ],
            dim=2,
        )
        prob_vol = torch.exp(prob_vol / lambd)   
        
    # maxpooling
    prob_dist, orientations = torch.max(prob_vol, dim=2)
    
    # get the prediction
    pred_y_in_pixel, pred_x_in_pixel = torch.where(prob_dist == prob_dist.max())
    sampled_index = torch.randint(0, pred_y_in_pixel.shape[0], (1,))
    
    pred_y = pred_y_in_pixel[sampled_index]
    pred_x = pred_x_in_pixel[sampled_index]
    
    orn = orientations[pred_y, pred_x]
    # from orientation indices to radians
    orn = orn / orn_slice * 2 * torch.pi
    pred = torch.cat((pred_x, pred_y, orn))
    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )

def finalize_localization(prob_vol: torch.IntTensor) -> Tuple[torch.tensor]:
    """
    Finalize localization using the combined probability volume.
    Input:
        prob_vol: combined probability volume (H, W, O)
    Output:
        prob_dist: probability distribution, (H, W) maxpool the prob_vol along orientation, ndarray
        orientations: orientation with max likelihood at each position, (H, W), ndarray
        pred: (3, ) predicted state [x,y,theta], ndarray
    """
    prob_dist, orientations = torch.max(prob_vol, dim=2)

    # get the prediction
    pred_y_in_pixel, pred_x_in_pixel = torch.where(prob_dist == prob_dist.max())
    sampled_index = torch.randint(0, pred_y_in_pixel.shape[0], (1,))
    
    pred_y = pred_y_in_pixel[sampled_index]
    pred_x = pred_x_in_pixel[sampled_index]
    orn = orientations[pred_y, pred_x]
    # from orientation indices to radians
    orn = orn / 36 * 2 * torch.pi
    pred = torch.cat((pred_x, pred_y, orn))
    return (
        prob_vol.detach().cpu().numpy(),
        prob_dist.detach().cpu().numpy(),
        orientations.detach().cpu().numpy(),
        pred.detach().cpu().numpy(),
    )
    
def finalize_localization_acc_only(prob_vol: torch.IntTensor) -> Tuple[torch.tensor]:
    """
    Finalize localization using the combined probability volume without orientation.
    Input:
        prob_vol: combined probability volume (H, W)
    Output:
        prob_dist: probability distribution (same as prob_vol in this case), ndarray
        pred: (2, ) predicted state [x, y], ndarray
    """
    # prob_dist is simply the prob_vol in this case as there is no orientation dimension
    prob_dist = prob_vol

    # get the prediction
    pred_y_in_pixel, pred_x_in_pixel = torch.where(prob_dist == prob_dist.max())
    sampled_index = torch.randint(0, pred_y_in_pixel.shape[0], (1,))
    
    pred_y = pred_y_in_pixel[sampled_index]
    pred_x = pred_x_in_pixel[sampled_index]
    pred = torch.cat((pred_x, pred_y))
    
    return (
        prob_vol.detach().cpu().numpy(),
        prob_dist.detach().cpu().numpy(),
        pred.detach().cpu().numpy(),
    )

def get_ray_from_depth(d, V=7, dv=10, a0=None, F_W=1/np.tan(0.698132)/2):
    """
    Shoot the rays to the depths, from left to right
    Input:
        d: 1d depths from image
        V: number of rays
        dv: angle between two neighboring rays
        a0: camera intrisic
        F/W: focal length / image width
    Output:
        rays: interpolated rays
    """
    W = d.shape[0]
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi

    if a0 is None:
        # assume a0 is in the middle of the image
        w = np.tan(angles) * W * F_W + (W - 1) / 2  # desired width, left to right
    else:
        w = np.tan(angles) * W * F_W + a0  # left to right
    # w=np.linspace(0, 39, 9)
    interp_d = griddata(np.arange(W).reshape(-1, 1), d, w, method="linear")
    rays = interp_d / np.cos(angles)

    return rays

def get_ray_from_semantics(semantics, V=7, dv=10, a0=None, F_W=1/np.tan(0.698132)/2):
    """
    Shoot the rays to the semantics, from left to right
    Input:
        semantics: 1D array of semantics from the image (e.g., [0, 1, 2] for wall, window, door)
        V: number of rays
        dv: angle between two neighboring rays (in degrees)
        a0: camera intrinsic (center of the image by default)
        F/W: focal length / image width ratio
    Output:
        rays: interpolated rays for semantics
    """
    W = semantics.shape[0]
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi

    if a0 is None:
        # assume a0 is in the middle of the image
        w = np.tan(angles) * W * F_W + (W - 1) / 2  # desired width, left to right
    else:
        w = np.tan(angles) * W * F_W + a0  # left to right
    # w=np.linspace(0, 39, 21)
    # w = np.clip(w, 0, W-1)
    # Interpolating semantics across the desired angles
    interp_semantics = griddata(np.arange(W).reshape(-1, 1), semantics, w, method="linear", fill_value=0)
    rays = np.round(interp_semantics).astype(int)  # Convert interpolated values to nearest integer to get the semantics

    return rays

from collections import Counter

def get_ray_from_semantics_v2(original_rays, angle_between_rays=80/39, desired_ray_count=9, window_size=0):

    desired_angle_step = 10  
    
    # Placeholder for the resulting representative rays
    representative_rays = []
    
    # Iterate over each desired ray
    for i in range(desired_ray_count):
        # Calculate the angle of the desired ray
        desired_angle = i * desired_angle_step
        
        # Find the closest index in the original rays
        idx_float = desired_angle / angle_between_rays
        idx = round(idx_float)
        
        # Collect neighbors for majority vote
        neighbors = []
        for j in range(max(0,idx - window_size), idx + window_size + 1):
            if 0 <= j < len(original_rays):  # Ensure we stay within bounds
                neighbors.append(original_rays[j])
        
        # Perform majority vote
        count = Counter(neighbors)
        majority_class = count.most_common(1)[0][0]
        
        # Append the majority class to the representative rays
        representative_rays.append(majority_class)
    
    return np.array(representative_rays)
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
    # Define weights according to the enum: WALL=0, WINDOW=1, DOOR=2, UNKNOWN=3
    weights = torch.tensor([1.0, 5.0, 3.0, 0.0], device=rays.device)
    
    # Create a mask that is 1 when predicted and ground truth differ, else 0.
    mismatches = (rays != desdf).float()  # Shape: (H, W, V)
    
    # Look up the weight for each predicted ray.
    ray_weights = weights[rays.long()]  # Shape: (H, W, V)
    
    # Compute the weighted error per ray.
    weighted_errors = mismatches * ray_weights  # Shape: (H, W, V)
    
    # Sum the weighted errors over the ray dimension.
    total_penalty_per_pixel = weighted_errors.sum(dim=2)  # Shape: (H, W)
    
    # Return the negative total penalty so that a perfect match yields 0.
    loss = -total_penalty_per_pixel
    return loss

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

def get_ray_from_semantics_v2(original_rays, angle_between_rays=80/40, desired_ray_count=9, window_size=1):
    desired_angle_step = 10  

    representative_rays = []
    
    # Assume the center of the FOV corresponds to the middle index.
    center_index = len(original_rays) // 2

    # For an odd number of desired rays, the middle one (index desired_ray_count//2) is 0°.
    # Thus, the desired angles (in degrees) are computed relative to 0.
    for i in range(desired_ray_count):
        # Compute desired angle relative to the center.
        desired_angle = (i - desired_ray_count // 2) * desired_angle_step
        
        # Compute the corresponding index offset (how many original rays away from the center).
        idx_offset = desired_angle / angle_between_rays
        
        # The target index is the center index plus the offset.
        idx = round(center_index + idx_offset)
        
        # Clamp the index so it remains within the valid range.
        idx = max(0, min(idx, len(original_rays) - 1))
        
        # If a window is provided, collect neighbors around the target index.
        neighbors = []
        for j in range(max(0, idx - window_size), min(len(original_rays), idx + window_size + 1)):
            neighbors.append(original_rays[j])
        
        # Use majority vote from the neighbors if there is a window; otherwise, just take the ray.
        count = Counter(neighbors)
        majority_class = count.most_common(1)[0][0]
        
        representative_rays.append(majority_class)
    
    return np.array(representative_rays)
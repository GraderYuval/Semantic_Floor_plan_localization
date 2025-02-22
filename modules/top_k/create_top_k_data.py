import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json
from tqdm import tqdm  # For progress bar

# === 1) Hard-coded config from your config.yaml ===
CONFIG = {
    "dataset_dir": "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full",
    "prob_vol_dir": "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full",
    "desdf_path": "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/desdf",
    "log_dir_depth": "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/modules/Final_wights/depth/final_depth_model_checkpoint.ckpt",
    "log_dir_semantic": "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/modules/Final_wights/semantic/final_semantic_model_checkpoint.ckpt",
    "combined_prob_vols_small_net": "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/logs/combined/Final_test/combined_prob_vols_net_type-large_dataset_size-medium_epochs-30_loss-nll_acc_only-True/final_combined_model_checkpoint.ckpt",
    "split_file": "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full/split.yaml",
    "results_dir": "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/modules/top_k/results_dist_1_top_10_vizualizations",

    # Dataset parameters
    "L": 0,

    # Depth network parameters
    "D": 128,
    "d_min": 0.1,
    "d_max": 15.0,
    "d_hyp": -0.2,

    # Localization
    "F_W": 0.59587643422,
    "V": 7,

    # Semantic network parameters
    "num_classes": 4,

    # Evaluation parameters
    "prediction_type": "combined",  # 'depth', 'semantic', 'combined', or 'all'
    "use_ground_truth_depth": False,
    "use_ground_truth_semantic": False,

    # Weight combinations
    "weight_combinations": [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [0.6, 0.4],
        [0.7, 0.3],
        [0.8, 0.2],
        [0.9, 0.1]
    ]
}

# === 2) References to your local modules ===
from data_utils.prob_vol_data_utils import ProbVolDataset
from utils.data_loader_helper import load_scene_data
from utils.raycast_utils import ray_cast   # <-- ensure your ray_cast function is available

# === 3) Probability volume helpers ===
def combine_prob_volumes(prob_vol_depth: torch.Tensor,
                         prob_vol_semantic: torch.Tensor,
                         depth_weight: float = 0.5,
                         semantic_weight: float = 0.5) -> torch.Tensor:
    """
    Combine two probability volumes [H, W, O] with given weights.
    """
    H = min(prob_vol_depth.shape[0], prob_vol_semantic.shape[0])
    W = min(prob_vol_depth.shape[1], prob_vol_semantic.shape[1])
    O = min(prob_vol_depth.shape[2], prob_vol_semantic.shape[2])
    
    depth_sliced = prob_vol_depth[:H, :W, :O]
    semantic_sliced = prob_vol_semantic[:H, :W, :O]
    return depth_weight * depth_sliced + semantic_weight * semantic_sliced

def get_max_over_orientation(prob_vol: torch.Tensor, num_orientations: int = 36):
    """
    Convert [H, W, O] -> (prob_dist: [H, W], orientation_map: [H, W])
    by taking max across orientation dimension.
    """
    prob_dist, orientation_map = torch.max(prob_vol, dim=2)
    return prob_dist, orientation_map

def indices_to_radians(orientation_idx: int, num_orientations: int = 36) -> float:
    """
    Convert orientation index (0..num_orientations-1) to radians [0..2π).
    """
    return orientation_idx / num_orientations * 2.0 * np.pi

def extract_top_k_locations(
    prob_dist: torch.Tensor,
    orientation_map: torch.Tensor,
    K: int = 1,
    min_dist_m: float = 1.0,
    resolution_m_per_pixel: float = 0.1,
    num_orientations: int = 36
):
    """
    From a 2D probability map (H, W), pick the top-K (x, y, orientation, prob_value)
    ensuring no two picks are within 'min_dist_m' in real-world space.
    """
    H, W = prob_dist.shape
    flat_prob = prob_dist.view(-1)          # shape [H*W]
    flat_orient = orientation_map.view(-1)  # shape [H*W]

    sorted_indices = torch.argsort(flat_prob, descending=True)

    picks = []
    min_dist_pixels = min_dist_m / resolution_m_per_pixel
    excluded_mask = torch.zeros(H, W, dtype=torch.bool)

    for idx in sorted_indices:
        if len(picks) >= K:
            break
        y = idx // W
        x = idx % W
        if excluded_mask[y, x]:
            continue

        pick_orientation_idx = flat_orient[idx].item()
        pick_prob_value = flat_prob[idx].item()
        pick_orientation_rad = indices_to_radians(pick_orientation_idx, num_orientations)

        picks.append({
            'x': float(x.item()),
            'y': float(y.item()),
            'orientation_radians': pick_orientation_rad,
            'prob_value': pick_prob_value
        })

        # Exclude neighbors
        y_min = max(0, int(y.item() - min_dist_pixels))
        y_max = min(H - 1, int(y.item() + min_dist_pixels))
        x_min = max(0, int(x.item() - min_dist_pixels))
        x_max = min(W - 1, int(x.item() + min_dist_pixels))
        for yy in range(y_min, y_max + 1):
            for xx in range(x_min, x_max + 1):
                dist = np.sqrt((yy - y.item())**2 + (xx - x.item())**2)
                if dist <= min_dist_pixels:
                    excluded_mask[yy, xx] = True

    return picks

# === 4) Plotting top-K on the semantic map with rays (FOV) ===
def plot_prob_dist_with_top_k(
    prob_dist: torch.Tensor,
    top_k_candidates: list,
    candidate_rays: dict,
    occ: np.ndarray = None,
    resolution: float = 0.1,
    save_path: str = None
):
    """
    Resizes 'prob_dist' to match 'occ' if available, flips it vertically
    for correct orientation, overlays as a heatmap on occ, draws circles for top-K
    candidates, and draws the rays (FOV) for each candidate. Now each ray is drawn
    in a color corresponding to its semantic prediction.
    
    candidate_rays is a dict mapping candidate index (1-indexed) to a list of tuples 
    (x, y, semantic_prediction) where (x, y) are the ray endpoint coordinates in the
    original probability map coordinate system.
    """
    # Define a mapping from semantic class to color. Adjust as needed.
    SEMANTIC_COLORS = {
        0: 'black',
        1: 'blue',
        2: 'red',
        3: 'orange'
    }

    if isinstance(prob_dist, torch.Tensor):
        prob_dist_np = prob_dist.detach().cpu().numpy()
    else:
        prob_dist_np = prob_dist

    if occ is not None:
        H_occ, W_occ = occ.shape[:2]
        prob_dist_resized = cv2.resize(prob_dist_np, (W_occ, H_occ), interpolation=cv2.INTER_LINEAR)
    else:
        prob_dist_resized = prob_dist_np

    # Flip vertically for correct orientation
    prob_dist_resized = np.flipud(prob_dist_resized)

    fig, ax = plt.subplots(figsize=(8, 6))
    if occ is not None:
        occ_flipped = np.flipud(occ)
        if occ_flipped.ndim == 2:
            ax.imshow(occ_flipped, cmap='gray', alpha=0.6)
        else:
            ax.imshow(occ_flipped, alpha=0.6)
    ax.imshow(prob_dist_resized, cmap='viridis', alpha=0.5)

    H_orig, W_orig = prob_dist_np.shape
    if occ is not None:
        H_occ, W_occ = occ.shape[:2]
    else:
        H_occ, W_occ = H_orig, W_orig

    # Plot each candidate
    for i, cand in enumerate(top_k_candidates, start=1):
        x_pixel = cand['x']
        y_pixel = cand['y']
        # Scale candidate position to occ image size
        x_scaled = (x_pixel / W_orig) * W_occ
        y_scaled = (y_pixel / H_orig) * H_occ
        y_flipped = (H_occ - 1) - y_scaled

        # Draw candidate circle
        circ = Circle((x_scaled, y_flipped), radius=100, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(circ)
        ax.plot(x_scaled, y_flipped, 'ro', markersize=3)
        label_str = f"K{i}\n{cand['prob_value']:.3f}"
        ax.text(x_scaled, y_flipped - 12, label_str, fontsize=10, color='blue', ha='center', va='top')

        # Plot the rays (if available)
        if i in candidate_rays:
            for (rx, ry, semantic_class) in candidate_rays[i]:
                # Scale the ray endpoint similarly.
                rx_scaled = rx
                ry_scaled = ry
                # Adjust for flipping
                ry_scaled = (H_occ - 1) - ry_scaled
                ray_color = SEMANTIC_COLORS.get(semantic_class, 'cyan')
                ax.plot([x_scaled, rx_scaled], [y_flipped, ry_scaled], color=ray_color, linewidth=1)

    ax.axis('off')
    ax.set_title("Combined Probability Distribution + Top-K Picks with Rays (FOV)")
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


# --- Global variables for workers (set via initializer) ---
global_test_set = None
global_scene_data = None
global_config = None

# === 5) Worker function for processing one image ===
def process_data_idx(idx):
    """
    Process a single dataset image. Uses global_test_set and global_scene_data.
    Returns a tuple (idx, scene, floor_image_idx, log_info)
    """
    log_info = []
    test_set = global_test_set
    scene_data = global_scene_data
    dataset_dir = global_config["dataset_dir"]
    desdf_path = global_config["desdf_path"]
    results_dir = global_config["results_dir"]
    prob_vol_dir = global_config["prob_vol_dir"]
    L = global_config["L"]
    resolution_m_per_pixel = 0.1  # hard-coded for now

    # Build derived path for probability volumes
    _ = os.path.join(prob_vol_dir, "prob_vols")  # Not used further in this function

    # Get the data item
    data = test_set[idx]
    # Compute the scene index using the stored scene_start_idx array.
    scene_idx = np.sum(idx >= np.array(test_set.scene_start_idx)) - 1
    scene = test_set.scene_names[scene_idx]
    if 'floor' not in scene:
        scene_number = int(scene.split('_')[1])
        scene = f"scene_{scene_number}"

    # Compute the image index for the floor (starting from 0)
    floor_image_idx = idx - test_set.scene_start_idx[scene_idx]

    valid_scene_names = scene_data["valid_scene_names"]
    if scene not in valid_scene_names:
        log_info.append(f"Scene {scene} not in valid_scene_names. Skipping...")
        return idx, scene, floor_image_idx, log_info

    # Prepare output folder paths
    scene_folder = os.path.join(results_dir, scene)
    image_folder = os.path.join(scene_folder, f"image_{floor_image_idx}")
    os.makedirs(image_folder, exist_ok=True)
    metadata_save_path = os.path.join(image_folder, "metadata.json")

    # Check if scene result already exists.
    # if os.path.exists(metadata_save_path):
    #     log_info.append(f"Results for scene {scene}, image index {floor_image_idx} already exist. Skipping processing.")
    #     return idx, scene, floor_image_idx, log_info

    # Use pre-loaded scene data (maps, walls, etc.)
    maps = scene_data["maps"]
    walls = scene_data["walls"]
    semantics = scene_data["semantics"]

    prob_vol_depth = data["prob_vol_depth"]    # [H, W, O]
    prob_vol_semantic = data["prob_vol_semantic"]  # [H, W, O]

    # 1) Combine probability volumes.
    combined_prob_vol = combine_prob_volumes(prob_vol_depth, prob_vol_semantic,
                                             depth_weight=0.5, semantic_weight=0.5)
    prob_dist_2d, orientation_map_2d = get_max_over_orientation(combined_prob_vol, num_orientations=36)

    # 2) Extract top-K locations.
    top_k_candidates = extract_top_k_locations(prob_dist_2d,
                                               orientation_map_2d,
                                               K=10,
                                               min_dist_m=1,
                                               resolution_m_per_pixel=resolution_m_per_pixel,
                                               num_orientations=36)

    semantic_map = maps[scene]
    if semantic_map.dtype != np.uint8:
        semantic_map = semantic_map.astype(np.float32)
        max_val = semantic_map.max()
        if max_val > 0:
            semantic_map = (semantic_map / max_val) * 255
        semantic_map = semantic_map.astype(np.uint8)

    walls_map = walls[scene]

    # 3) Compute candidate rays (for plotting the FOV).
    ray_n = 40  # number of rays per candidate
    F_W = 1 / np.tan(0.698132) / 2
    depth_max = 15  # meters

    candidate_rays = {}
    json_data = {}
    depth_rays_all_candidates = []    # List [K x ray_n]
    semantic_rays_all_candidates = []  # List [K x ray_n]

    for i, cand in enumerate(top_k_candidates, start=1):
        cand_x_m = cand['x'] * resolution_m_per_pixel  
        cand_y_m = cand['y'] * resolution_m_per_pixel  
        cand_orientation = cand['orientation_radians']
        cand_score = cand['prob_value']

        # Candidate center in original probability map pixel coordinates.
        center_x = cand['x'] * 10
        center_y = cand['y'] * 10
        candidate_pos_pixels = np.array([center_x, center_y])

        center_angs = np.flip(np.arctan2((np.arange(ray_n) - np.arange(ray_n).mean()),
                                         ray_n * F_W))
        ray_angles = center_angs + cand_orientation

        depth_rays = []
        semantic_rays = []
        candidate_ray_endpoints = []  # To store endpoints (x, y, semantic_prediction) in pixels

        for ang in ray_angles:
            # Cast ray on walls map for depth and hit coordinates.
            dist_depth, _, hit_coords_walls, _ = ray_cast(
                walls_map, candidate_pos_pixels, ang, dist_max=depth_max*100
            )
            # Cast ray on semantic map for semantic prediction.
            _, prediction_class, _, _ = ray_cast(
                semantic_map, candidate_pos_pixels, ang, dist_max=depth_max*100, min_dist=80
            )
            depth_val_m = dist_depth / 100.0  
            depth_rays.append(depth_val_m)
            semantic_rays.append(prediction_class)
            candidate_ray_endpoints.append((hit_coords_walls[0], hit_coords_walls[1], prediction_class))

        # Flip the ray endpoints order (if needed)
        endpoints_tensor = torch.tensor(candidate_ray_endpoints)
        endpoints_tensor = torch.flip(endpoints_tensor, [0])
        candidate_ray_endpoints = endpoints_tensor.tolist()

        depth_rays_all_candidates.append(depth_rays)
        semantic_rays_all_candidates.append(semantic_rays)
        candidate_rays[i] = candidate_ray_endpoints

        json_data[f"K{i}"] = {
            "x": cand_x_m,
            "y": cand_y_m,
            "o": cand_orientation,
            "score": cand_score,
            "angle_count": ray_n,
            "depth_rays": [
                {
                    'angle_degrees': np.rad2deg(ang),
                    'distance_m': dist
                }
                for ang, dist in zip(ray_angles, depth_rays)
            ],
            "semantic_rays": [
                {
                    'angle_degrees': np.rad2deg(ang),
                    'prediction_class': pred
                }
                for ang, pred in zip(ray_angles, semantic_rays)
            ]
        }

    # 4) Save the probability distribution plot (scene .png) WITH rays overlaid.
    plot_filename = os.path.join(image_folder, f"{scene}_{floor_image_idx}.png")
    plot_prob_dist_with_top_k(prob_dist=prob_dist_2d,
                              top_k_candidates=top_k_candidates,
                              candidate_rays=candidate_rays,
                              occ=semantic_map,
                              resolution=resolution_m_per_pixel,
                              save_path=plot_filename)

    # 5) Save metadata JSON and tensors for the candidate rays.
    with open(metadata_save_path, "w") as mf:
        json.dump(json_data, mf, indent=4)
    log_info.append(f"Saved metadata JSON to {metadata_save_path}")

    depth_tensor = torch.tensor(depth_rays_all_candidates)    # shape: [K, ray_n]
    semantic_tensor = torch.tensor(semantic_rays_all_candidates)  # shape: [K, ray_n]

    depth_save_path = os.path.join(image_folder, "depth.pt")
    semantic_save_path = os.path.join(image_folder, "semantic.pt")
    torch.save(depth_tensor, depth_save_path)
    torch.save(semantic_tensor, semantic_save_path)
    log_info.append(f"Saved depth tensor to {depth_save_path}")
    log_info.append(f"Saved semantic tensor to {semantic_save_path}")

    log_info.append(f"Scene: {scene}, Image (floor index): {floor_image_idx}")
    log_info.append(f"Depth tensor [K, ray_n]:\n{depth_tensor}")
    log_info.append(f"Semantic tensor [K, ray_n]:\n{semantic_tensor}")

    return idx, scene, floor_image_idx, log_info

# === 6) Batch worker function (processes a chunk of image indices) ===
def process_batch(indices):
    results = []
    for idx in indices:
        try:
            result = process_data_idx(idx)
            results.append(result)
        except Exception as e:
            results.append((idx, None, None, [f"Exception: {e}"]))
    return results

# === 7) Initializer for worker processes ===
def worker_init(test_set, scene_data, config):
    global global_test_set, global_scene_data, global_config
    global_test_set = test_set
    global_scene_data = scene_data
    global_config = config

# === 8) Main pipeline with multiprocessing and progress bar ===
def main():
    import yaml
    from attrdict import AttrDict

    # Load the split file to determine which scenes to process.
    with open(CONFIG["split_file"], "r") as f:
        split_data = AttrDict(yaml.safe_load(f))
    # scene_names = split_data.train + split_data.val + split_data.test
    scene_names = split_data.test[:10]

    dataset_dir = CONFIG["dataset_dir"]
    prob_vol_dir = CONFIG["prob_vol_dir"]
    L = CONFIG["L"]
    desdf_path = CONFIG["desdf_path"]
    results_dir = CONFIG["results_dir"]
    resolution_m_per_pixel = 0.1  # or use a config parameter if available

    # Build the probability volume path
    _ = os.path.join(prob_vol_dir, "prob_vols")

    # Create a test set instance to determine the number of items.
    test_set = ProbVolDataset(
        dataset_dir=dataset_dir,
        scene_names=scene_names,
        L=L,
        prob_vol_path=os.path.join(prob_vol_dir, "prob_vols"),
        acc_only=False
    )
    num_items = len(test_set)
    print(f"Total items to process: {num_items}")

    # --- Load scene data once in main ---
    desdfs, semantics, maps, gt_poses, valid_scene_names, walls = load_scene_data(
        test_set, dataset_dir, desdf_path
    )
    scene_data = {
        "desdfs": desdfs,
        "semantics": semantics,
        "maps": maps,
        "gt_poses": gt_poses,
        "valid_scene_names": valid_scene_names,
        "walls": walls
    }

    # --- Create batches/chunks ---
    chunk_size = 50  # Adjust batch size as needed
    indices = list(range(num_items))
    batches = [indices[i:i+chunk_size] for i in range(0, len(indices), chunk_size)]

    # --- Multiprocessing using ProcessPoolExecutor ---
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Limit the number of workers to, say, 4 (adjust this value as needed)
    max_workers = 4

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=worker_init,
                             initargs=(test_set, scene_data, CONFIG)) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                print(f"Batch exception: {e}")

if __name__ == "__main__":
    main()

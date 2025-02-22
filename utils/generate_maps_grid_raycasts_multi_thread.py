import os
import cv2
import numpy as np
import tqdm
import yaml
from raycast_utils import ray_cast, get_color_name
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import os
import cv2
import numpy as np
import tqdm
import yaml
from raycast_utils import ray_cast, get_color_name
from modules.semantic.semantic_mapper import ObjectType, object_to_color
import matplotlib.pyplot as plt


def raycast_depth(
    occ_walls, orn_slice=36, max_dist=1500, original_resolution=0.01, output_resolution=0.1
):
    """
    Get desdf from walls-only occupancy grid and color from semantic grid through brute force raycast.
    Input:
        occ_walls: the walls-only map as occupancy.
        orn_slice: number of equiangular orientations.
        max_dist: maximum raycast distance, [m].
        original_resolution: the resolution of occ input [m/pixel].
        resolution: output resolution of the desdf [m/pixel].
    Output:
        desdf: the directional esdf of the occ input in meters.
    """
    ratio = output_resolution / original_resolution
    desdf = np.zeros(list((np.array(occ_walls.shape[:2]) // ratio).astype(int)) + [orn_slice])

    # Perform raycasting for each orientation slice
    for o in tqdm.tqdm(range(orn_slice)):
        theta = o / orn_slice * np.pi * 2
        for y in range(desdf.shape[0]):
            for x in range(desdf.shape[1]):
                pos = np.array([x, y]) * ratio
                dist, _, _ , _ = ray_cast(occ_walls, pos, theta)
                desdf[y, x, o] = dist / 100  # ray_cast returns in mm/10 --> Store the distance in M    
    
    return desdf

def raycast_semantic(
    occ_semantic, orn_slice=36, max_dist=1500, original_resolution=0.01, output_resolution=0.1, min_dist=5
):
    ratio = output_resolution / original_resolution
    colors = np.zeros(list((np.array(occ_semantic.shape[:2]) // ratio).astype(int)) + [orn_slice])

    # Perform raycasting for each orientation slice
    for o in tqdm.tqdm(range(orn_slice)):
        theta = o / orn_slice * np.pi * 2
        for col in range(colors.shape[0]):
            for row in range(colors.shape[1]):
                pos = np.array([row, col]) * ratio
                _, color_val , _ , _ = ray_cast(occ_semantic, pos, theta, min_dist=min_dist)
                colors[col, row, o] = color_val                                      
    return colors

def load_yaml(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def process_scene(scene, base_dir, desdf_dir):
    is_zind= False
    if 'floor' in scene:  # Zind dataset
        is_zind = True
        pass
    else:
        scene_number = int(scene.split('_')[1])
        scene = f"scene_{scene_number}"
    try:
        semantic_map_path = os.path.join(base_dir, scene, 'floorplan_semantic.png')
        walls_only_map_path = os.path.join(base_dir, scene, 'floorplan_walls_only.png')
        print(f"Processing scene: {scene}")
        
        # Load the maps
        occ_walls = plt.imread(walls_only_map_path)
        occ_semantic = plt.imread(semantic_map_path)
        
        desdf = {}
        color = {}
        
        # Compute DESDF and color
        desdf["desdf"] = raycast_depth(occ_walls)
        min_dist = 80 if is_zind else 5
        color["desdf"] = raycast_semantic(occ_semantic, min_dist=min_dist)
        
        # Save the results
        scene_dir = os.path.join(desdf_dir, scene)
        if not os.path.exists(scene_dir):
            os.mkdir(scene_dir)
        
        np.save(os.path.join(scene_dir, "desdf.npy"), desdf)
        np.save(os.path.join(scene_dir, "color.npy"), color)
    except:
        print(f"Failed Processing scene: {scene}")


def process_test_scenes(yaml_path, base_dir, desdf_dir):
    # Load the YAML file
    split_data = load_yaml(yaml_path)
    scenes = split_data.get('test', [])
    # scenes = ['scene_03261','scene_03279','scene_03280','scene_03452']
    # scenes.sort(key=lambda x: int(x.split('_')[-1]))
    
    total_scenes = len(scenes)
    print(f"Total scenes to process: {total_scenes}\n")

    # Define the number of processes (e.g., number of CPU cores)
    number_of_processes = 8

    # Use partial to fix base_dir and desdf_dir arguments
    process_scene_partial = partial(process_scene, base_dir=base_dir, desdf_dir=desdf_dir)

    # Use Pool to process scenes in parallel
    with Pool(processes=number_of_processes) as pool:
        list(tqdm.tqdm(pool.imap(process_scene_partial, scenes), total=total_scenes))

if __name__ == "__main__":
    # Update these paths to match your environment
    # #Zind
    yaml_path = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/zind_data_set_80_fov/split.yaml'
    base_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/zind_data_set_80_fov'
    desdf_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/desdf_80_fov'

    # #S3D
    # yaml_path = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full/split.yaml'
    # base_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full'
    # desdf_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/desdf'

    process_test_scenes(yaml_path, base_dir, desdf_dir)

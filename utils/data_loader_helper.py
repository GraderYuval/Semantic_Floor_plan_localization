# utils/data_loader_helper.py

import os
import cv2
import numpy as np
import tqdm

def load_scene_data(test_set, dataset_dir, desdf_path):
    desdfs = {}
    semantics = {}
    maps = {}
    walls = {}
    gt_poses = {}
    valid_scene_names = []  # To keep track of valid scenes

    for scene in tqdm.tqdm(test_set.scene_names):
        try:
            if 'floor' in scene: # zind
                pass
            else:
                scene_number = int(scene.split('_')[1])
                scene = f"scene_{scene_number}"
            
            desdf = np.load(os.path.join(desdf_path, scene, "desdf.npy"), allow_pickle=True)
            semantic = np.load(os.path.join(desdf_path, scene, "color.npy"), allow_pickle=True)
            occ_sem = cv2.imread(os.path.join(dataset_dir, scene, "floorplan_semantic.png"))
            occ_walls = cv2.imread(os.path.join(dataset_dir, scene, "floorplan_walls_only.png"))
            
            desdfs[scene] = desdf.item()
            semantics[scene] = semantic.item()
            maps[scene] = occ_sem
            walls[scene] = occ_walls


            with open(os.path.join(dataset_dir, scene, "poses.txt"), "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]
                traj_len = len(poses_txt)
                poses = np.zeros([traj_len, 3], dtype=np.float32)
                for state_id in range(traj_len):
                    pose = poses_txt[state_id].split(" ")
                    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
                    poses[state_id, :] = np.array((x, y, th), dtype=np.float32)
                gt_poses[scene] = poses
            
            valid_scene_names.append(scene)
        except:
            print(f"Error in loading desdf of: {scene}")
            continue

    # print(f"number of valid scenes for evaluation: {len(valid_scene_names)} out of: {len(test_set.scene_names)} --> {(len(valid_scene_names)/len(test_set.scene_names))*100}%")
    return desdfs, semantics, maps, gt_poses, valid_scene_names, walls

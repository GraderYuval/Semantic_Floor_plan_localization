import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from utils.utils import gravity_align

K = np.array([[320/np.tan(0.698132), 0, 320],
            [0, 180/np.tan(0.440992), 180],
            [0, 0, 1]], dtype=np.float32) # hardcoded intrinsics for structured3D cameras

class GridSeqDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        data_dir=None,
        roll=0,
        pitch=0,
        start_scene=None,
        end_scene=None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.L = L
        self.data_dir = data_dir or dataset_dir
        self.roll = roll
        self.pitch = pitch
        self.start_scene = start_scene
        self.end_scene = end_scene
        self.scene_start_idx = []
        self.gt_values = []
        self.gt_pose = []
        self.load_scene_start_idx_and_values_and_poses()
    
        self.total_len=0
        for poses in self.gt_pose:
            self.total_len+= len(poses)
    
    def __len__(self):        
        return self.total_len
    
    def load_scene_start_idx_and_values_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        valid_scene_names = []  # To keep track of valid scenes

        for scene_idx, scene in enumerate(self.scene_names):
            # Remove leading zeros from the scene number part
            if 'floor' in scene: #Zind dataset
                pass
            else: #S3D
                scene_number = int(scene.split('_')[1])
                scene = f"scene_{scene_number}"
            
            scene_folder = os.path.join(self.data_dir, scene)
            
            depth_file = os.path.join(scene_folder, "depth.txt")
            semantic_file = os.path.join(scene_folder, "semantic.txt")
            pitch_file = os.path.join(scene_folder, "pitch.txt")
            roll_file = os.path.join(scene_folder, "roll.txt")
            
            try:
                # Read depth
                with open(depth_file, "r") as f:
                    depth_txt = [line.strip() for line in f.readlines()]

                # Read semantics
                with open(semantic_file, "r") as f:
                    semantic_txt = [line.strip() for line in f.readlines()]

                # Read pose
                pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
                with open(pose_file, "r") as f:
                    poses_txt = [line.strip() for line in f.readlines()]

                # Read pitch and roll
                with open(pitch_file, "r") as f:
                    pitch_txt = [float(line.strip()) for line in f.readlines()]

                with open(roll_file, "r") as f:
                    roll_txt = [float(line.strip()) for line in f.readlines()]
            
            except Exception as e:
                # If any of the files are missing or corrupted, continue to the next scene
                print(f"Error: missing depth/semantic/pitch/roll files - skipping: {scene}")
                continue

            traj_len = len(poses_txt)
            scene_depths = []
            scene_semantics = []
            scene_poses = []
            scene_pitch = []
            scene_roll = []
            
            for state_id in range(traj_len):
                # Get depth
                depth = depth_txt[state_id].split(" ")
                depth = np.array([float(d) for d in depth]).astype(np.float32)
                scene_depths.append(depth)

                # Get semantic
                semantic = semantic_txt[state_id].split(" ")
                semantic = np.array([float(s) for s in semantic]).astype(np.float32)
                scene_semantics.append(semantic)

                # Get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)

                # Get pitch and roll
                scene_pitch.append(pitch_txt[state_id])
                scene_roll.append(roll_txt[state_id])

            # If we reach here, the scene is valid, so we add it to the valid scenes list
            valid_scene_names.append(self.scene_names[scene_idx])  # Track valid scenes only
            start_idx += traj_len
            self.scene_start_idx.append(start_idx)
            
            # Store ground truth values
            self.gt_values.append({
                "depth": scene_depths,
                "semantic": scene_semantics,
                "pitch": scene_pitch,
                "roll": scene_roll
            })
            self.gt_pose.append(scene_poses)

        # After loop, update scene_names to valid scenes only
        self.scene_names = valid_scene_names


    def __getitem__(self, idx):
        if self.start_scene is not None:
            idx += self.scene_start_idx[self.start_scene]

        # Get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        is_zind = False
        # Remove leading zeros from the scene number part
        if 'floor' in scene_name: #Zind dataset
            is_zind = True
            pass        
        else: #S3D
            scene_number = int(scene_name.split('_')[1])
            scene_name = f"scene_{scene_number}"

        # Get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        # Get reference depth
        ref_depth = self.gt_values[scene_idx]["depth"][idx_within_scene]
        data_dict = {"ref_depth": ref_depth}

        # Get reference semantic
        ref_semantics = self.gt_values[scene_idx]["semantic"][idx_within_scene]
        data_dict["ref_semantics"] = ref_semantics

        # Get reference pose
        ref_pose = self.gt_pose[scene_idx][idx_within_scene]
        data_dict["ref_noise"] = 0
        data_dict["ref_pose"] = ref_pose

        # Get pitch and roll
        ref_pitch = self.gt_values[scene_idx]["pitch"][idx_within_scene]
        ref_roll = self.gt_values[scene_idx]["roll"][idx_within_scene]
        data_dict["ref_pitch"] = ref_pitch
        data_dict["ref_roll"] = ref_roll

        # Get reference image
        image_path = os.path.join(
            self.dataset_dir,
            scene_name,
            "rgb",
            str(idx_within_scene) + ".png",
        )
        ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if ref_img is None:
            raise FileNotFoundError(f"Image not found or could not be read at path: {image_path}")

        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0            
        r = ref_roll
        p = ref_pitch
        
        if not is_zind:
        # if True:
            ref_img = gravity_align(ref_img, r=r, p=p, K=K)                    
            mask = np.ones(list(ref_img.shape[:2]))
            mask = gravity_align(mask, r=r, p=p, K=K)            
            mask[mask < 1] = 0
            ref_mask = mask.astype(np.uint8)
            data_dict["ref_mask"] = ref_mask
        else: # no gravity_align for Zind
            mask = np.ones(list(ref_img.shape[:2])).astype(np.uint8)
            data_dict["ref_mask"] = mask

        # From H,W,C to C,H,W
        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
        data_dict["ref_img"] = ref_img

        return data_dict


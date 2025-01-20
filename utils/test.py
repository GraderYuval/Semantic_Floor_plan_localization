import os
import cv2
import yaml

def load_yaml(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def check_scene_dimensions(yaml_path, base_dir):
    # Load the YAML file
    split_data = load_yaml(yaml_path)
    test_scenes = split_data.get('test', [])

    # Loop through each test scene and check dimensions
    mismatched_scenes = []
    for scene in test_scenes:
        scene_number = int(scene.split('_')[1])
        scene = f"scene_{scene_number}"
        print(f"Checking scene: {scene}")
        
        semantic_map_path = os.path.join(base_dir, scene, 'floorplan_semantic.png')
        walls_only_map_path = os.path.join(base_dir, scene, 'floorplan_walls_only.png')
        
        # Load the maps
        occ_walls = cv2.imread(walls_only_map_path)[:, :, :3]
        occ_semantic = cv2.imread(semantic_map_path)[:, :, :3]
        
        # Check if the dimensions of the images are different
        if occ_walls.shape != occ_semantic.shape and (abs(occ_walls.shape[0] - occ_semantic.shape[0])>2 or abs(occ_walls.shape[1] - occ_semantic.shape[1])>2):
            mismatched_scenes.append(scene)
            print(f"Dimension mismatch found in scene {scene}:")
            print(f"    Walls map shape: {occ_walls.shape}")
            print(f"    Semantic map shape: {occ_semantic.shape}")

    # Print summary of mismatched scenes
    if mismatched_scenes:
        print("\nScenes with dimension mismatches:")
        for scene in mismatched_scenes:
            print(f"  - {scene}")
    else:
        print("\nNo dimension mismatches found.")

if __name__ == "__main__":
    yaml_path = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_Plan_Localization/data/test_data_set/structured3d_perspective_empty/split.yaml'
    base_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_Plan_Localization/data/test_data_set/structured3d_perspective_empty'
    
    check_scene_dimensions(yaml_path, base_dir)

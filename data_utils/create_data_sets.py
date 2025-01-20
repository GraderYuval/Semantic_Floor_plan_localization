import os
from create_casting_files import process_scene as process_casting_scene

def process_all_scenes(base_path, output_base_path, resolution=0.01, dpi=100, fov_segments=40):
    scenes_to_process = ['scene_03261','scene_03279','scene_03280','scene_03296','scene_03452']
    scenes = [d for d in scenes_to_process if os.path.isdir(os.path.join(base_path, d))]

    # scenes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith('scene_')] 
    # scenes.sort(key=lambda x: int(x.split('_')[-1]))
    # scenes = scenes[:1]
    # scenes = scenes[0+489:500] #1
    # scenes = scenes[500+195:1000]#2
    # scenes = scenes[1000+234:1500]#3
    # scenes = scenes[1500+139:2000]#4
    # scenes = scenes[2000+169+204:2500]#5
    # scenes = scenes[2500+281:3000]#6
    # scenes = scenes[3000:3500]#7
    total_scenes = len(scenes)
    print(f"Total scenes to process: {total_scenes}\n")
    failed_scenes = []
    
    for idx, scene_dir in enumerate(scenes, start=1):
        scene_id = str(int(scene_dir.split('_')[-1]))
        try:
            print(f"Processing Scene {scene_id} ({idx}/{total_scenes})...")
            process_casting_scene(scene_id, base_path, output_base_path, resolution, dpi, fov_segments)
            print(f"Finished processing Scene {scene_id}.\n")
        except Exception as e:
            print(f"Failed to process Scene {scene_id}. Error: {e}\n")
            failed_scenes.append(scene_id)
    
    if failed_scenes:
        print("The following scenes failed to process:")
        for scene in failed_scenes:
            print(f"Scene {scene}")


def main():
    # base_path = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_Plan_Localization/data/structured3d_perspective_empty/Structured3D"
    base_path = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_Plan_Localization/data/structured3d_perspective_full/Structured3D"
    # base_path = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_Plan_Localization/data/temp"
    output_base_path = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_Plan_Localization/data/test_data_set_full/structured3d_perspective_full"
    # output_base_path = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_Plan_Localization/data/depth_from_semantic_data_set/structured3d_perspective_empty"
    # output_base_path = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_Plan_Localization/data/temp_data_set"
    resolution = 0.01  # Resolution in meters per pixel
    dpi = 100          # DPI for output images
    fov_segments = 40  # Number of segments to divide the FOV into

    process_all_scenes(base_path, output_base_path, resolution, dpi, fov_segments)

if __name__ == "__main__":
    main()

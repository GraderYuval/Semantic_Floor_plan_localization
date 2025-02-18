import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Import these for color mapping. Ensure that your PYTHONPATH is set appropriately.
from modules.semantic.semantic_mapper import ObjectType, object_to_color

def plot_camera_positions_and_rays(camera_positions, img, ray_data, output_path, resolution=0.01, dpi=100, position_key='semantic'):
    """
    Plots camera positions and their corresponding rays on the provided image and saves the plot.
    """
    img_height, img_width = img.shape[:2]
    fig, ax = plt.subplots(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
    ax.imshow(img)

    # Plot camera positions as blue dots.
    for camera_info in camera_positions:
        if position_key == 'semantic':
            camera_x = camera_info['vx_semantic']
            camera_y = camera_info['vy_semantic']
        elif position_key == 'walls':
            camera_x = camera_info['vx_walls']
            camera_y = camera_info['vy_walls']
        else:
            raise ValueError(f"Invalid position_key: {position_key}. Expected 'semantic' or 'walls'.")
        ax.plot(camera_x, camera_y, 'bo', markersize=5)

    # Plot rays using the stored start and end positions.
    for camera_data in ray_data['cameras']:
        for ray in camera_data['rays']:
            if position_key == 'semantic':
                start_x = ray['start_position_semantic']['x']
                start_y = ray['start_position_semantic']['y']
            elif position_key == 'walls':
                start_x = ray['start_position_walls']['x']
                start_y = ray['start_position_walls']['y']
            else:
                raise ValueError(f"Invalid position_key: {position_key}. Expected 'semantic' or 'walls'.")

            end_x = ray['end_position']['x']
            end_y = ray['end_position']['y']

            # Map the ray color using the object_to_color dictionary.
            object_type = ObjectType(ray['prediction_class'])
            color = object_to_color.get(object_type, 'black')  # default to black if type not found

            ax.plot([start_x, end_x], [start_y, end_y], color=color, lw=0.5)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('equal')
    ax.axis('off')

    # Ensure the output folder exists.
    os.makedirs(output_path, exist_ok=True)
    output_image_path = os.path.join(output_path, f'camera_positions_with_rays_{position_key}.png')
    plt.savefig(output_image_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    print(f"Plot saved to: {output_image_path}")

if __name__ == "__main__":
    # --- Hardcoded configuration ---
    # Folder containing the input data (semantic floorplan image and camera rays JSON).
    input_folder = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full/scene_0"
    
    # Folder where the visualization (plot) will be saved.
    output_folder = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/results/final_results/visualtizations/visualize_rays"

    # --- Load the semantic floorplan image ---
    semantic_img_path = os.path.join(input_folder, 'floorplan_semantic.png')
    if not os.path.exists(semantic_img_path):
        print(f"Semantic floorplan image not found at {semantic_img_path}")
        exit(1)
    img_semantic = plt.imread(semantic_img_path)

    # --- Load the raycast JSON data ---
    ray_data_path = os.path.join(input_folder, 'camera_rays.json')
    if not os.path.exists(ray_data_path):
        print(f"Camera rays JSON not found at {ray_data_path}")
        exit(1)
    with open(ray_data_path, 'r') as f:
        ray_data = json.load(f)

    # --- Extract camera positions ---
    # The visualization function expects a list of dictionaries with keys:
    # 'vx_semantic', 'vy_semantic', 'vx_walls', and 'vy_walls'.
    camera_positions = []
    for cam in ray_data['cameras']:
        camera_positions.append({
            'vx_semantic': cam['camera_position_pixel_semantic']['x'],
            'vy_semantic': cam['camera_position_pixel_semantic']['y'],
            'vx_walls': cam['camera_position_pixel_walls']['x'],
            'vy_walls': cam['camera_position_pixel_walls']['y']
        })

    # --- Plot and save the camera positions with rays ---
    plot_camera_positions_and_rays(camera_positions, img_semantic, ray_data, output_folder, position_key='semantic')

    # Optionally display the plot window.
    plt.show()


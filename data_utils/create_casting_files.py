import os
import json
import numpy as np
import matplotlib.pyplot as plt
import shutil
import argparse
from utils.raycast_utils import ray_cast
from modules.semantic.semantic_mapper import ObjectType, object_to_color
from create_floorplans import visualize_floorplan
import math
from scipy.spatial.transform import Rotation


def plot_camera_positions_and_rays(camera_positions, img, ray_data, output_path, resolution=0.01, dpi=100, position_key='semantic'):
    img_height, img_width = img.shape[:2]
    fig, ax = plt.subplots(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
    ax.imshow(img)

    # Plot camera positions
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

    # Plot rays using the stored start and end positions from ray_data
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

            # Map the ray color using the object_to_color dictionary
            object_type = ObjectType(ray['prediction_class'])  # Convert numeric value to ObjectType
            color = object_to_color.get(object_type, 'black')  # Default to 'black' if type is not found

            ax.plot([start_x, end_x], [start_y, end_y], color=color, lw=0.5)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('equal')
    ax.axis('off')

    output_image_path = os.path.join(output_path, f'camera_positions_with_rays_{position_key}.png')
    plt.savefig(output_image_path, dpi=dpi, bbox_inches='tight', pad_inches=0)


def copy_and_rename_images(scene_id, base_path, output_base_path):
    # Use the integer representation of scene_id for the output paths
    int_scene_id = str(int(scene_id))

    # Define the path to the specific scene's 2D rendering directory
    scene_base_path = os.path.join(base_path, f'scene_{scene_id.zfill(5)}', '2D_rendering')

    # Create the output directory for RGB images
    rgb_output_path = os.path.join(output_base_path, f'scene_{int_scene_id}', 'rgb')
    os.makedirs(rgb_output_path, exist_ok=True)

    image_counter = 0

    # Traverse the trajectory and sub-trajectory folders in sorted order
    for traj in sorted(os.listdir(scene_base_path)):
        traj_dir = os.path.join(scene_base_path, traj)

        # Traverse the sub-trajectory folders in sorted order
        for sub_traj in sorted(os.listdir(os.path.join(traj_dir, 'perspective', 'full'))):
            sub_traj_dir = os.path.join(traj_dir, 'perspective', 'full', sub_traj)

            # Find the image with 'rgb_rawlight' in the file name
            for file in sorted(os.listdir(sub_traj_dir)):
                if file.endswith('.png') and 'rgb_rawlight' in file:
                    source_file = os.path.join(sub_traj_dir, file)
                    dest_file = os.path.join(rgb_output_path, f"{image_counter}.png")
                    shutil.copyfile(source_file, dest_file)
                    image_counter += 1

def create_raycast_file(camera_positions, img_walls, img_semantic, output_path, resolution=0.01, fov_segments=40, epsilon=0.01):
    resolution_mm_per_pixel = resolution * 1000  # Convert resolution to mm per pixel
    scene_data = {'cameras': []}
    
    ray_n = fov_segments  # Number of rays
    F_W = 1 / np.tan(0.698132) / 2  # Adjust F_W based on the provided formula

    for i, camera_info in enumerate(camera_positions):
        camera_x_walls = camera_info['vx_walls']
        camera_y_walls = camera_info['vy_walls']
        camera_x_semantic = camera_info['vx_semantic']
        camera_y_semantic = camera_info['vy_semantic']

        # Camera orientation vectors
        view_dir = np.array([camera_info['tx'], camera_info['ty'], camera_info['tz']])
        up_dir = np.array([camera_info['ux'], camera_info['uy'], camera_info['uz']])
        right_dir = np.cross(up_dir, view_dir)  # Right direction

        # Calculate the angle th on the 2D (xy) plane
        th = np.arctan2(camera_info['ty'], camera_info['tx'])

        # Normalize vectors
        view_dir = view_dir / np.linalg.norm(view_dir)
        up_dir = up_dir / np.linalg.norm(up_dir)
        right_dir = right_dir / np.linalg.norm(right_dir)

        # Convert NumPy arrays to lists for JSON serialization
        view_dir_list = view_dir.tolist()
        up_dir_list = up_dir.tolist()
        right_dir_list = right_dir.tolist()

        # Calculate equidistant angles using the new formula
        center_angs = np.flip(np.arctan2((np.arange(ray_n) - np.arange(ray_n).mean()), ray_n * F_W))

        # Adjust the angles by the camera's orientation
        angs = center_angs + th
        
        camera_data = {
            'room_id': camera_info['room_id'],
            'camera_number': camera_info['camera_number'],
            'camera_position_m_semantic': {'x': camera_x_semantic / 100, 'y': camera_y_semantic / 100},
            'camera_position_pixel_semantic': {'x': camera_x_semantic, 'y': camera_y_semantic},
            'camera_position_m_walls': {'x': camera_x_walls / 100, 'y': camera_y_walls / 100},
            'camera_position_pixel_walls': {'x': camera_x_walls, 'y': camera_y_walls},
            'normalize_dir': {'view_dir': view_dir_list, "up_dir": up_dir_list, "right_dir": right_dir_list},
            'camera_full_info': camera_info,
            'th': np.rad2deg(th),  # store th in degrees for easier interpretation
            'rays': []
        }

        for i, ang in enumerate(angs):
            # First raycast: Calculate the distance using the walls-only image
            # dist, _, hit_coords_walls, _ = ray_cast(img_walls, np.array([camera_x_walls, camera_y_walls]), ang, dist_max=50000)
            dist, _, hit_coords_walls, _ = ray_cast(img_walls, np.array([camera_x_walls, camera_y_walls]), ang)

            # Second raycast: Get the prediction class using the semantic image
            _, prediction_class, _, normal = ray_cast(img_semantic, np.array([camera_x_semantic, camera_y_semantic]), ang)

            # Cosine adjustment to account for the angle
            distance_adjusted = dist * np.cos(center_angs[i])

            # Use the hit coordinates from the walls-only raycast as the end position of the ray
            end_x, end_y = hit_coords_walls

            ray_data = {
                'angle': np.rad2deg(ang),
                'distance_m': distance_adjusted * resolution_mm_per_pixel / 1000,  # from MM to M
                'prediction_class': prediction_class,
                'start_position_semantic': {'x': camera_x_semantic, 'y': camera_y_semantic},
                'start_position_walls': {'x': camera_x_walls, 'y': camera_y_walls},
                'end_position': {'x': end_x, 'y': end_y},
                'normal': normal  # Store the normal as an angle in degrees
            }
            camera_data['rays'].append(ray_data)

        scene_data['cameras'].append(camera_data)

    output_file = os.path.join(output_path, 'camera_rays.json')
    with open(output_file, 'w') as f:
        json.dump(scene_data, f, indent=4)        

    return scene_data  # Return the ray data for further processing



def create_additional_files(ray_data, img, output_path):
    depth_file = os.path.join(output_path, 'depth.txt')
    poses_file = os.path.join(output_path, 'poses.txt')
    colors_file = os.path.join(output_path, 'semantic.txt')

    with open(depth_file, 'w') as df, open(poses_file, 'w') as pf, open(colors_file, 'w') as cf:
        for camera in ray_data['cameras']:
            # Write depth and semantic information
            for ray_index, ray in enumerate(camera['rays']):
                df.write(f"{ray['distance_m']} ")
                cf.write(f"{ray['prediction_class']} ")

            df.write('\n')
            cf.write('\n')

            yaw_rad = np.deg2rad(camera['th'])
            pf.write(f"{camera['camera_position_m_semantic']['x']} {camera['camera_position_m_semantic']['y']} {yaw_rad}\n")


def create_pitch_roll_files_from_json(output_path):
    # Path to the camera_rays.json file
    camera_rays_path = os.path.join(output_path, 'camera_rays.json')

    # Load the ray data from camera_rays.json
    with open(camera_rays_path, 'r') as f:
        ray_data = json.load(f)

    # Define output files for pitch, roll, and theta
    pitch_file = os.path.join(output_path, 'pitch.txt')
    roll_file = os.path.join(output_path, 'roll.txt')

    with open(pitch_file, 'w') as pf, open(roll_file, 'w') as rf:
        for camera in ray_data['cameras']:
            # Get the directional vectors directly from the JSON data
            tx = float(camera['camera_full_info']['tx'])
            ty = float(camera['camera_full_info']['ty'])
            tz = float(camera['camera_full_info']['tz'])

            ux = float(camera['camera_full_info']['ux'])
            uy = float(camera['camera_full_info']['uy'])
            uz = float(camera['camera_full_info']['uz'])

            # Create the direction vectors
            # Forward-facing (view direction)
            t = np.array([tx, ty, tz])
            t = t / np.linalg.norm(t)  # Normalize the vector

            # Upwards (up direction)

            u = np.array([ux, uy, uz])
            u = u / np.linalg.norm(u)  # Normalize the vector #TODO make sure they have mistake

            # Get the y-axis (right direction) using cross product
            w = np.cross(u, t)

            # Recompute the up vector to ensure orthogonality
            u = np.cross(t, w)

            # Build the rotation matrix
            R = np.stack([t, w, u], axis=1)
            
            # Convert rotation matrix to a Rotation object
            r = Rotation.from_matrix(R)

            # Get Euler angles in radians
            theta, pitch, roll = r.as_euler('ZYX')

            # Write pitch, roll, and theta in degrees to the files
            pf.write(f"{pitch}\n")
            rf.write(f"{roll}\n")




def process_scene(scene_id, base_path, output_base_path, resolution=0.01, dpi=100, fov_segments=40):
    # Use the integer representation of scene_id for the output paths
    int_scene_id = str(int(scene_id))
    
    # Visualize floorplans and get the bounding box coordinates
    annotation_file = os.path.join(base_path, f'scene_{scene_id.zfill(5)}', 'annotation_3d.json')
    with open(annotation_file, 'r') as file:
        annos = json.load(file)

    semantic_bbox, walls_only_bbox = visualize_floorplan(annos, int_scene_id, output_base_path, resolution, dpi)

    camera_positions = []

    # Define the path to the specific scene's 2D rendering directory
    scene_base_path = os.path.join(base_path, f'scene_{scene_id.zfill(5)}', '2D_rendering')

    for traj in sorted(os.listdir(scene_base_path)):
        traj_dir = os.path.join(scene_base_path, traj)

        # Traverse sub-trajectory folders (also sorted for consistency)
        for sub_traj in sorted(os.listdir(os.path.join(traj_dir, 'perspective', 'full'))):
            sub_traj_dir = os.path.join(traj_dir, 'perspective', 'full', sub_traj)

            # Get the camera pose file and read it
            camera_pose_file = os.path.join(sub_traj_dir, 'camera_pose.txt')
            if not os.path.exists(camera_pose_file):
                continue  # Skip if no pose file is found

            # Correctly identify room_id and position_id from the path structure
            room_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(traj_dir))))
            position_id = os.path.basename(sub_traj_dir)

            resolution_mm_per_pixel = resolution * 1000  # Convert resolution to mm per pixel

            with open(camera_pose_file, 'r') as file:
                for line in file:
                    try:
                        vx_raw, vy_raw, vz, tx, ty, tz, ux, uy, uz, xfov, yfov, _ = map(float, line.strip().split())

                        # Corrected positions for semantic_bbox
                        vx_semantic = vx_raw / resolution_mm_per_pixel - semantic_bbox[0]
                        vy_semantic = -vy_raw / resolution_mm_per_pixel - semantic_bbox[1]

                        # Corrected positions for walls_only_bbox
                        vx_walls = vx_raw / resolution_mm_per_pixel - walls_only_bbox[0]
                        vy_walls = -vy_raw / resolution_mm_per_pixel - walls_only_bbox[1]

                        # Append camera position in the same order as image paths are traversed
                        camera_positions.append({
                            'room_id': room_id,
                            'camera_number': position_id,
                            'vx_semantic': vx_semantic, 'vy_semantic': vy_semantic,
                            'vx_walls': vx_walls, 'vy_walls': vy_walls,
                            'vz': vz,
                            'tx': tx, 'ty': ty, 'tz': tz,
                            'ux': ux, 'uy': uy, 'uz': uz,
                            'xfov': xfov
                        })

                    except ValueError as e:
                        print(f"Error parsing line: {line}")
                        print(f"Error details: {e}")


    if not camera_positions:
        print(f"No camera positions were found.")
        return

    # Read the corresponding cropped floorplan images (walls and semantic) from the output_base_path
    floorplan_walls_image_path = os.path.join(output_base_path, f'scene_{int_scene_id}', f'floorplan_walls_only.png')
    floorplan_semantic_image_path = os.path.join(output_base_path, f'scene_{int_scene_id}', f'floorplan_semantic.png')

    if not os.path.exists(floorplan_walls_image_path) or not os.path.exists(floorplan_semantic_image_path):
        return

    img_walls = plt.imread(floorplan_walls_image_path)
    img_semantic = plt.imread(floorplan_semantic_image_path)

    # Create raycast file and retrieve the ray data
    ray_data = create_raycast_file(camera_positions, img_walls, img_semantic,
                                   os.path.join(output_base_path, f'scene_{int_scene_id}'), resolution, fov_segments)

    # Create additional files from the ray data
    create_additional_files(ray_data, img_semantic, os.path.join(output_base_path, f'scene_{int_scene_id}'))

    # Create pitch and roll files
    create_pitch_roll_files_from_json(os.path.join(output_base_path, f'scene_{int_scene_id}'))
    
    # Plot and save camera positions with rays on semantic image
    plot_camera_positions_and_rays(camera_positions, img_semantic, ray_data,
                                   os.path.join(output_base_path, f'scene_{int_scene_id}'), resolution, dpi,
                                   position_key='semantic')

    # Copy and rename the images
    copy_and_rename_images(scene_id, base_path, output_base_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process scene data.')
    parser.add_argument('--scene_id', type=str, required=True, help='Scene ID to process.')
    parser.add_argument('--base_path', type=str, required=True, help='Base path to the scene data.')
    parser.add_argument('--output_base_path', type=str, required=True, help='Output base path for processed data.')
    parser.add_argument('--resolution', type=float, default=0.01, help='Resolution in meters per pixel.')
    parser.add_argument('--dpi', type=int, default=100, help='DPI for image processing.')
    parser.add_argument('--fov_segments', type=int, default=40, help='Number of segments for FOV rays.')

    args = parser.parse_args()

    process_scene(args.scene_id, args.base_path, args.output_base_path, args.resolution, args.dpi, args.fov_segments)

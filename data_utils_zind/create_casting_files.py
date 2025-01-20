import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.raycast_utils import ray_cast
from modules.semantic.semantic_mapper import ObjectType, object_to_color
import argparse
import os
from zind_utils import Polygon, PolygonType, pano2persp, rot_verts
from typing import List, Tuple
import cv2
import re
from render_floorplans_zind import (
    render_jpg_image,
)

import numpy as np
import matplotlib.pyplot as plt
import os

DEFAULT_RENDER_RESOLUTION = 2048

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_camera_positions_and_rays(camera_positions, img, ray_data, output_path, resolution=0.01, dpi=100, position_key='semantic'):
    img_height, img_width = img.shape[:2]
    
    # -----------------------
    # Figure 1: Camera positions and rays
    # -----------------------
    fig1, ax1 = plt.subplots(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
    ax1.imshow(img)
    
    for i, camera_data in enumerate(ray_data['cameras']):
        x = camera_data['camera_position_pixel_semantic']['x']
        y = camera_data['camera_position_pixel_semantic']['y']
        
        # Plot the camera position as a blue dot
        ax1.plot(x, y, 'bo', markersize=5)
        
        # Plot the image number (index) below each camera point
        ax1.text(x, y - 0.04, f"{str(i)}", color='green', fontsize=30, fontweight='bold', ha='center', va='top')
        
        # Plot rays from the camera position to each endpoint
        for ray in camera_data['rays']:
            end_x = ray['end_position']['x']
            end_y = ray['end_position']['y']
            
            # Map the ray color using the object_to_color dictionary
            object_type = ObjectType(ray['prediction_class'])  # Convert numeric value to ObjectType
            color = object_to_color.get(object_type, 'black')    # Default to 'black' if type is not found
            
            ax1.plot([x, end_x], [y, end_y], color=color, lw=0.5)
    
    # Adjust layout and save Figure 1
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax1.axis('equal')
    ax1.axis('off')
    output_image_path1 = os.path.join(output_path, f'camera_positions_with_rays_{position_key}.png')
    fig1.savefig(output_image_path1, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig1)
    
    # -----------------------
    # Figure 2: Camera positions with circles (no rays)
    # -----------------------
    fig2, ax2 = plt.subplots(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
    ax2.imshow(img)
    
    for i, camera_data in enumerate(ray_data['cameras']):
        x = camera_data['camera_position_pixel_semantic']['x']
        y = camera_data['camera_position_pixel_semantic']['y']
        
        # Plot the camera position as a blue dot
        ax2.plot(x, y, 'bo', markersize=5)
        
        # Plot the image number (index) below each camera point
        ax2.text(x, y - 0.04, f"{str(i)}", color='green', fontsize=30, fontweight='bold', ha='center', va='top')
        
        # Draw a circle with radius 100 around the camera position
        circle = Circle((x, y), radius=80, edgecolor='green', facecolor='none', lw=1)
        ax2.add_patch(circle)
        
    # Adjust layout and save Figure 2
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax2.axis('equal')
    ax2.axis('off')
    output_image_path2 = os.path.join(output_path, f'camera_positions_with_circles_{position_key}.png')
    fig2.savefig(output_image_path2, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig2)


def dataset_to_ray_cast(angle_rad):
    """
    Transforms dataset angle to ray_cast angle by rotating 270 degrees (3π/2 radians) and inverting direction.

    Parameters:
        angle_rad (float or np.ndarray): Angle(s) in radians from the dataset.

    Returns:
        float or np.ndarray: Transformed angle(s) in radians for ray_cast.
    """
    # transformed_angle = (1 * np.pi / 2 - angle_rad) % (2 * np.pi)
    transformed_angle = (np.radians(90) - angle_rad) % (2 * np.pi)

    return transformed_angle

def create_raycast_file(camera_positions, img_walls, img_semantic, output_path, resolution=0.01, fov_segments=40, epsilon=0.01, depth =15, rot_rad = None):
    resolution_mm_per_pixel = resolution * 1000  # Convert resolution to mm per pixel
    scene_data = {'cameras': []}
    
    ray_n = fov_segments  # Number of rays
    F_W = 1 / np.tan(0.698132) / 2  # Adjust F_W based on the provided formula
    # F_W = 0.5

    with open(camera_positions, "r") as f:
        poses_txt = [line.strip() for line in f.readlines()]
    
                    
    for i, camera_info in enumerate(poses_txt):
        pose = camera_info.split(" ")
        pose = np.array([float(s) for s in pose]).astype(float)
        x = pose[0]
        y = pose[1]
        th = pose [2]
        center_angs = np.flip(np.arctan2((np.arange(ray_n) - np.arange(ray_n).mean()), ray_n * F_W))

        # Adjust the angles by the camera's orientation
        angs = center_angs + rot_rad[i] 
        angs = dataset_to_ray_cast(angs)[::-1]
        
        camera_data = {
            'camera_number': i,
            'camera_position_m_semantic': {'x': x , 'y': y},
            'camera_position_pixel_semantic': {'x': x*100, 'y': y*100},
            'camera_position_m_walls': {'x': x , 'y': y },
            'camera_position_pixel_walls': {'x': x*100, 'y': y*100},
            'th': th, 
            'rays': []
        }

        for i, ang in enumerate(angs):
            # First raycast: Calculate the distance using the walls-only image
            dist, _, hit_coords_walls, _ = ray_cast(img_walls, np.array([x*100, y*100]), ang, dist_max= depth*100)

            # Second raycast: Get the prediction class using the semantic image
            _, prediction_class, _, normal = ray_cast(img_semantic, np.array([x*100, y*100]), ang, dist_max= depth*100,min_dist=80)

            # Cosine adjustment to account for the angle
            # distance_adjusted = dist * np.cos(center_angs[i])
            distance_adjusted = dist

            # Use the hit coordinates from the walls-only raycast as the end position of the ray
            end_x, end_y = hit_coords_walls

            ray_data = {
                'angle': np.rad2deg(ang),
                'distance_m': distance_adjusted * resolution_mm_per_pixel / 1000,  # from MM to M
                'prediction_class': prediction_class,
                'start_position_semantic': {'x': x, 'y': y},
                'start_position_walls': {'x': x, 'y': y},
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
    # poses_file = os.path.join(output_path, 'poses.txt')
    colors_file = os.path.join(output_path, 'semantic.txt')
    pitch_file = os.path.join(output_path, 'pitch.txt')
    roll_file = os.path.join(output_path, 'roll.txt')

    with open(depth_file, 'w') as df, open(colors_file, 'w') as cf, open(pitch_file, 'w') as pitch_f, open(roll_file, 'w') as rf:
        for camera in ray_data['cameras']:
            # Write depth and semantic information
            for ray_index, ray in enumerate(camera['rays']):
                df.write(f"{ray['distance_m']} ")
                cf.write(f"{ray['prediction_class']} ")

            df.write('\n')
            cf.write('\n')

            pitch_f.write(f"0\n")
            rf.write(f"0\n")

def extract_floor_room_pano_from_name(name: str):
    # Extract the floor, room, and pano numbers from the file name.
    match = re.search(r'floor_(-?\d+)_partial_room_(\d+)_pano_(\d+)\.jpg', name)
    if match:
        floor = int(match.group(1))
        room = int(match.group(2))
        pano = int(match.group(3))
        return floor, room, pano
    return None, None, None                    

def extract_polygons_from_zind(data: dict, floor_name: str, global_rot: float = 0.0) -> List[Polygon]:
    """
    Extract a list of polygons for 'room', 'door', and 'window' from the 
    first pano in each room on the specified floor. Scales & rotates 
    to match the ZInD pipeline logic (pano transform + floor scale + global_rot).

    Parameters
    ----------
    data : dict
        The loaded zind_data.json dictionary.
    floor_name : str
        E.g. "floor_0" or "floor_-1", etc.
    global_rot : float
        Optional extra rotation in degrees after everything else.

    Returns
    -------
    polygon_list : List[Polygon]
        Each polygon has (points, type, name).
    """
    polygon_list = []

    # Floor scale in meters/coordinate
    floor_scale = data["scale_meters_per_coordinate"][floor_name]
    if floor_scale is None:
        return polygon_list  # no geometry

    # For each room in this floor
    floor_data = data["merger"].get(floor_name, {})
    for room_name, partial_rooms in floor_data.items():
        # pick first partial-room
        first_partial_room_key = list(partial_rooms.keys())[0]
        proom_node = partial_rooms[first_partial_room_key]
        # pick first pano inside that partial-room
        first_pano_key = list(proom_node.keys())[0]
        pano_node = proom_node[first_pano_key]

        pano_transform = pano_node["floor_plan_transformation"]
        # layout_complete has e.g. "vertices", "doors", "windows", ...
        layout = pano_node.get("layout_complete", {})

        # ============ Handle ROOM (vertices) ============
        if "vertices" in layout:
            room_verts = np.array(layout["vertices"], dtype=float)
            if len(room_verts) >= 3:
                # 1) rotate by pano_transform["rotation"]
                room_verts = rot_verts(room_verts, pano_transform["rotation"])
                # 2) scale by pano_transform["scale"]
                room_verts *= pano_transform["scale"]
                # 3) translate
                t = np.array(pano_transform["translation"], dtype=float)
                room_verts += t
                # 4) multiply by floor_scale
                room_verts *= floor_scale
                # 5) optional global rotation
                if abs(global_rot) > 1e-6:
                    room_verts = rot_verts(room_verts, global_rot)

                # Construct a Polygon
                polygon_list.append(
                    Polygon(points=[(pt[0], pt[1]) for pt in room_verts],
                            type=PolygonType.ROOM,
                            name=f"room_{room_name}")
                )

        # ============ Handle DOORS ============
        if "doors" in layout and len(layout["doors"]) > 0:
            door_verts = np.array(layout["doors"], dtype=float)
            # same transform pipeline
            door_verts = rot_verts(door_verts, pano_transform["rotation"])
            door_verts *= pano_transform["scale"]
            door_verts += np.array(pano_transform["translation"], dtype=float)
            door_verts *= floor_scale
            if abs(global_rot) > 1e-6:
                door_verts = rot_verts(door_verts, global_rot)

            polygon_list.append(
                Polygon(points=[(pt[0], pt[1]) for pt in door_verts],
                        type=PolygonType.DOOR,
                        name=f"doors_{room_name}")
            )

        # ============ Handle WINDOWS ============
        if "windows" in layout and len(layout["windows"]) > 0:
            window_verts = np.array(layout["windows"], dtype=float)
            # same transform pipeline
            window_verts = rot_verts(window_verts, pano_transform["rotation"])
            window_verts *= pano_transform["scale"]
            window_verts += np.array(pano_transform["translation"], dtype=float)
            window_verts *= floor_scale
            if abs(global_rot) > 1e-6:
                window_verts = rot_verts(window_verts, global_rot)

            polygon_list.append(
                Polygon(points=[(pt[0], pt[1]) for pt in window_verts],
                        type=PolygonType.WINDOW,
                        name=f"windows_{room_name}")
            )

    return polygon_list

def create_posses_and_copy_images(
    scene_path: str,
    output_path: str,
    posses_file_name: str = "",
    fov: int = 80,
    camera_poses_shifted: List[tuple] = None,
):
    if camera_poses_shifted is None:
        camera_poses_shifted = []
        
    gt_rots = []
    gt_fovs = []

    # Copy images, store fovs & rots
    for idx in range(len(camera_poses_shifted)):
        (x_shifted, y_shifted, rot_deg, img_path) = camera_poses_shifted[idx]
        rot_deg =(rot_deg % 360 + 360) % 360 
        pano_image_path = os.path.join(scene_path, img_path)
        if not os.path.isfile(pano_image_path):
            print(f"[create_posses_and_copy_images] Missing pano image: {pano_image_path}")
            continue

        pano_image = cv2.imread(pano_image_path, cv2.IMREAD_COLOR)
        if pano_image is None:
            print(f"[create_posses_and_copy_images] Failed to load: {pano_image_path}")
            continue

        query_image = pano2persp(pano_image, fov, 0, 0, 0, (360, 640))

        # Save the perspective in 'rgb' folder
        image_output_path = os.path.join(output_path, "rgb")
        os.makedirs(image_output_path, exist_ok=True)
        final_path = os.path.join(image_output_path, f"{idx}.png")
        cv2.imwrite(final_path, query_image)

        gt_rots.append(float(rot_deg))
        gt_fovs.append(float(fov))

    # Now write those SHIFTED camera poses to 'poses.txt'
    posses_file = os.path.join(output_path, posses_file_name)
    with open(posses_file, "w") as f_pose:
        for i in range(len(camera_poses_shifted)):
            x_shifted, y_shifted, rot_deg, _ = camera_poses_shifted[i]
            rot_deg =(rot_deg % 360 + 360) % 360 
            rot_rad = np.deg2rad(rot_deg)
            rot_rad = dataset_to_ray_cast(rot_rad)
            f_pose.write(f"{x_shifted} {y_shifted} {rot_rad}\n")

    # Write metadata
    ret_img = {
        "gt_rot": gt_rots, 
        "gt_fov": gt_fovs,
    }
    json_output_path = os.path.join(output_path, "metadata.json")
    with open(json_output_path, "w") as json_file:
        json.dump(ret_img, json_file, indent=4)

    return [np.deg2rad(r_deg) for r_deg in gt_rots]
        
def get_polygon_list_points(polygon_list: List[Polygon]) -> List[List[tuple]]:
    """
    Shifts all polygon coordinates so the top-left becomes (0,0),
    and optionally resizes if the bounding box is bigger than 2048 in either dimension.
    Returns a list of the same shape as polygon_list, i.e. each element is a list of (x,y).
    """
    # Gather all x,y into one big list
    all_x = []
    all_y = []
    for poly in polygon_list:
        for (xx, yy) in poly.points:
            all_x.append(xx)
            all_y.append(yy)
    if len(all_x) == 0:
        return [[] for _ in polygon_list]

    min_x, min_y = min(all_x), min(all_y)
    max_x, max_y = max(all_x), max(all_y)

    # Shift everything so (min_x, min_y) is (0,0)
    shift_polygons = []
    for poly in polygon_list:
        shifted = [(p[0] - min_x, p[1] - min_y) for p in poly.points]
        shift_polygons.append(shifted)

    # Possibly scale down if bounding box > 2048
    width, height = (max_x - min_x), (max_y - min_y)
    largest_dim = max(width, height)
    if largest_dim < 1e-8:
        scale_factor = 1.0
    else:
        scale_factor = min(2048.0 / largest_dim, 1.0)  # <= 1.0 if you don't want to upscale

    final_polygons = []
    for shifted_points in shift_polygons:
        scaled = [(pt[0] * scale_factor, pt[1] * scale_factor) for pt in shifted_points]
        final_polygons.append(scaled)

    return final_polygons

def shift_polygons_and_cameras(
    zind_poly_list: List[Polygon],
    camera_info: List[Tuple[float, float, float]],
) -> Tuple[List[Polygon], List[Tuple[float, float, float]]]:
    if not zind_poly_list and not camera_info:
        return zind_poly_list, camera_info

    all_x = []
    all_y = []

    # Gather polygon coords
    for poly in zind_poly_list:
        for (px, py) in poly.points:
            all_x.append(px)
            all_y.append(py)

    # Gather camera coords
    for (cx, cy, _, _ ) in camera_info:
        all_x.append(cx)
        all_y.append(cy)

    if not all_x:
        return zind_poly_list, camera_info  # no geometry at all

    min_x = min(all_x)
    min_y = min(all_y)

    # Shift polygons
    shifted_polygons = []
    for poly in zind_poly_list:
        shifted_pts = []
        for (px, py) in poly.points:
            sx = px - min_x
            sy = py - min_y
            shifted_pts.append((sx, sy))
        # Rebuild the polygon
        new_poly = Polygon(
            points=shifted_pts,
            type=poly.type,
            name=poly.name
        )
        shifted_polygons.append(new_poly)

    # Shift cameras
    shifted_cameras = []
    for (cx, cy, rot_deg, img_path) in camera_info:
        sx = cx - min_x
        sy = cy - min_y
        shifted_cameras.append((sx, sy, rot_deg, img_path))

    return shifted_polygons, shifted_cameras


def compute_camera_poses_zind(
    zind_data: dict,
    floor_name: str,
    global_rot_deg: float = 0.0
) -> List[Tuple[float, float, float]]:
    """
    For each pano in data["merger"][floor_name], compute (x,y,rot_deg)
    just like the snippet does.
    """
    camera_info = []
    floor_scale = zind_data["scale_meters_per_coordinate"][floor_name]
    floor_dict = zind_data["merger"].get(floor_name, {})

    for room_name in floor_dict:
        partial_dict = floor_dict[room_name]
        for partial_room_name in partial_dict:
            pano_dict = partial_dict[partial_room_name]
            for pano_node_key, pano_node in pano_dict.items():
                img_path = pano_node["image_path"]
                fp_trans = pano_node["floor_plan_transformation"]
                pano_loc = np.array(fp_trans["translation"]) * floor_scale
                pano_loc = rot_verts(pano_loc, global_rot_deg)
                pano_rot_deg = fp_trans["rotation"] + global_rot_deg
                # pano_rot_deg = fp_trans["rotation"]

                x, y = pano_loc  # shape (2,)
                camera_info.append((x, y, pano_rot_deg, img_path))

    return camera_info


def process_scene(scene_id, base_path, output_base_path, resolution=0.01, dpi=100, fov_segments=40, depth = 15):
    scene_folder = os.path.join(base_path, str(scene_id))
    zind_json = os.path.join(scene_folder, "zind_data.json")
    if not os.path.isfile(zind_json):
        print(f"[process_scene_zind_style] Missing {zind_json}; skipping.")
        return

    with open(zind_json, "r") as f:
        data = json.load(f)
    
    # Which floors are valid?
    scaled_floors = [
        floor_name
        for floor_name, sc in data["scale_meters_per_coordinate"].items()
        if sc is not None
    ]
    if len(scaled_floors) == 0:
        print(f"No valid floors found for scene {scene_id}")
        return
    
    # For each valid floor, extract polygons and render
    for floor_name in scaled_floors:
        print(f"processing floor: {floor_name}")
        # Optionally unify orientation:
        # e.g., global_rot_deg = - data["floorplan_to_redraw_transformation"][floor_name]["rotation"]
        global_rot_deg = 0.0
        if "floorplan_to_redraw_transformation" in data:
            if floor_name in data["floorplan_to_redraw_transformation"]:
                global_rot_deg = - data["floorplan_to_redraw_transformation"][floor_name]["rotation"]
        
        # Extract polygons
        zind_poly_list = extract_polygons_from_zind(data, floor_name, global_rot=global_rot_deg)
        if len(zind_poly_list) == 0:
            print(f"  Floor '{floor_name}' found no polygons. Skipping.")
            continue
        
        camera_info = compute_camera_poses_zind(data, floor_name, global_rot_deg)
        
        shifted_polygons, shifted_cameras = shift_polygons_and_cameras(zind_poly_list, camera_info)
        
        polygon_list_points = [p.points for p in shifted_polygons]
        
        out_dir = os.path.join(output_base_path, f"scene_{str(int(scene_id))}_{floor_name}")
        os.makedirs(out_dir, exist_ok=True)

        # 1) Render wall-only
        wall_only_img_path = os.path.join(out_dir, "floorplan_walls_only.png")
        render_jpg_image(
            polygon_list=zind_poly_list,
            polygon_list_points=polygon_list_points,
            jpg_file_name=wall_only_img_path,
            rendering_type="wall_only",
            output_path=out_dir,
            floor_scale=1.0 
        )

        # 2) Render semantic
        semantic_img_path = os.path.join(out_dir, "floorplan_semantic.png")
        render_jpg_image(
            polygon_list=zind_poly_list,
            polygon_list_points=polygon_list_points,
            jpg_file_name=semantic_img_path,
            rendering_type="semantic",
            output_path=out_dir,
            floor_scale=1.0
        )
            
        # Create combined poses and metadata files
        posses_file_name = os.path.join(out_dir, "poses.txt")
        rot_rads = create_posses_and_copy_images(
            scene_path=os.path.join(base_path, scene_id),
            output_path=out_dir,
            posses_file_name="poses.txt",
            fov=80,
            camera_poses_shifted=shifted_cameras 
        )
        
        img_walls = plt.imread(wall_only_img_path)
        img_semantic = plt.imread(semantic_img_path)

        # # Create raycast file and retrieve the ray data
        ray_data = create_raycast_file(posses_file_name, img_walls, img_semantic, out_dir, resolution, fov_segments, depth, rot_rad= rot_rads)
        
        create_additional_files(ray_data, img_semantic, out_dir)

        # Plot and save camera positions with rays on semantic image
        plot_camera_positions_and_rays(posses_file_name, img_semantic, ray_data, out_dir, resolution, dpi, position_key='semantic')



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

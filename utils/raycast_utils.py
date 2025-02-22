import numpy as np
from modules.semantic.semantic_mapper import ObjectType, object_to_color

def get_color_name(r, g, b):
    """
    Convert RGB values to a descriptive color name.
    Input:
        r, g, b: The RGB values of the pixel.
    Output:
        color_name: A string representing the color ('black', 'blue', 'red', etc.)
    """
    if r < 0.1 and g < 0.1 and b < 0.1:
        return 'black'
    elif r > 0.8 and g < 0.2 and b < 0.2:
        return 'red'
    elif r < 0.2 and g < 0.2 and b > 0.8:
        return 'blue'
    else:
        return None

def cast_ray(occ, pos, theta, dist_max=1500, min_dist=5):
    """
    Cast a single ray and return the hit information.
    """
    img_height, img_width = occ.shape[:2]
    pos_x, pos_y = pos

    sin_ang = np.sin(theta)
    cos_ang = np.cos(theta)

    for dist in range(5, dist_max, 1):  # Iterate over each mm until dist_max
        new_x = int(pos_x + (dist * cos_ang))
        new_y = int(pos_y - (dist * sin_ang))  # Negative because image y-coordinates increase downwards

        if new_x < 0 or new_x >= img_width or new_y < 0 or new_y >= img_height:
            return dist, ObjectType.UNKNOWN.value, (new_x, new_y)  # Hit the boundary of the image

        occ_value = occ[new_y, new_x]

        # If occ_value has more than three channels, only take the first three (assuming RGB)
        if len(occ_value) > 3:
            occ_value = occ_value[:3]

        r, g, b = occ_value  # Unpack RGB values

        color_name = get_color_name(r, g, b)

        if color_name:
            for obj_type, obj_color in object_to_color.items():
                if color_name == obj_color:
                    if dist < min_dist and obj_type == ObjectType.DOOR:
                        continue
                    
                    return dist, obj_type.value, (new_x, new_y)
                
    return dist_max, ObjectType.UNKNOWN.value, (new_x, new_y) 


# def get_normal(occ, pos, theta, dist_max, epsilon):
#     """
#     Calculate the normal by casting rays at angles theta and theta + epsilon.
#     Special handling for corners.
#     """
#     # Cast rays at theta and theta + epsilon
#     dist_base, _, hit_coords_base = cast_ray(occ, pos, theta, dist_max)
#     dist_eps, _, hit_coords_eps = cast_ray(occ, pos, theta + epsilon, dist_max)

#     # Calculate the tangent vector based on the positions of the hits
#     tangent = np.array([hit_coords_eps[0] - hit_coords_base[0], 
#                         hit_coords_eps[1] - hit_coords_base[1]])

#     # Check if the tangent vector is non-zero
#     if np.linalg.norm(tangent) == 0:
#         # If tangent is zero, return a default normal angle
#         normal_angle_deg = np.degrees(theta + np.pi / 2)
#     else:
#         # Normalize the tangent vector
#         tangent = tangent / np.linalg.norm(tangent)

#         # The normal vector is perpendicular to the tangent vector
#         normal = np.array([-tangent[1], tangent[0]])

#         # Calculate the angle of the normal vector relative to the x-axis
#         normal_angle_rad = np.arctan2(normal[1], normal[0])
#         normal_angle_deg = np.degrees(normal_angle_rad)

#         # Special check for corner cases
#         if np.abs(hit_coords_base[0] - hit_coords_eps[0]) > 0 and np.abs(hit_coords_base[1] - hit_coords_eps[1]) > 0:
#             # Corner detected: decide which axis to align the normal with
#             if np.abs(hit_coords_base[0] - hit_coords_eps[0]) > np.abs(hit_coords_base[1] - hit_coords_eps[1]):
#                 normal_angle_deg = 0  # Align with horizontal
#             else:
#                 normal_angle_deg = 90  # Align with vertical

#     # Check if the normal angle is NaN
#     if np.isnan(normal_angle_deg):
#         # Return a default value or handle as needed
#         normal_angle_deg = np.degrees(theta + np.pi / 2)

#     # Round the normal angle to the nearest 45 degrees
#     normal_angle_deg = round(normal_angle_deg / 45) * 45

#     # Ensure the angle is within the range [-180, 180]
#     if normal_angle_deg > 180:
#         normal_angle_deg -= 360
#     elif normal_angle_deg <= -180:
#         normal_angle_deg += 360

    # return normal_angle_deg



def ray_cast(occ, pos, theta, dist_max=1500, epsilon=0.02, min_dist=5):
    """
    Cast ray in the occupancy map and calculate the normal.
    Input:
        occ: Occupancy map (3D array for RGB).
        pos: in image coordinate, in pixels, [h, w]
        theta: Ray shooting angle, in radians.
        dist_max: Maximum distance to cast the ray, in mm.
        epsilon: Small angle difference for secondary ray.
    Output:
        dist: Distance in mm to the first obstacle or boundary.
        object_type_number: Integer representing the ObjectType that was hit (0 for WALL, 1 for WINDOW, 2 for DOOR).
        hit_coords: (new_x, new_y) coordinates of the hit point in image space.
        normal_angle: Normal angle at the hit point in degrees.
    """
    # Cast the primary ray
    dist, obj_type, hit_coords = cast_ray(occ, pos, theta, dist_max, min_dist)

    # Calculate the normal angle using the get_normal method
    # normal_angle = get_normal(occ, pos, theta, dist_max, epsilon)
    normal_angle = 0
    return dist, obj_type, hit_coords, normal_angle
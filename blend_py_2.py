#!/usr/bin/env python3

import blenderproc as bproc
import numpy as np
import argparse
import os
import json
import csv
import uuid
import random
from glob import glob
from mathutils import Matrix, Vector, Euler
from scipy.spatial.distance import pdist, squareform
from filelock import FileLock
from PIL import Image

def load_global_poses(global_poses_file):
    if os.path.exists(global_poses_file):
        with open(global_poses_file, 'r') as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []

def write_global_poses(global_poses_file, poses_list):
    with open(global_poses_file, 'w') as f:
        json.dump(poses_list, f, indent=2)

def check_and_reserve_pose(pose_data, current_poses, global_poses_file, lock_file, trans_threshold=0.001, rot_threshold=0.001):
    pose_summary = {
        'id': str(uuid.uuid4()),
        'translation': pose_data['translation'],
        'euler_angles_xyz_radians': pose_data['euler_angles_xyz_radians']
    }

    with FileLock(lock_file):
        global_poses = load_global_poses(global_poses_file)

        for existing_pose in global_poses:
            trans_diff = np.linalg.norm(np.array(pose_summary['translation']) - np.array(existing_pose['translation']))
            rot_diff = np.linalg.norm(np.array(pose_summary['euler_angles_xyz_radians']) - np.array(existing_pose['euler_angles_xyz_radians']))
            if trans_diff < trans_threshold and rot_diff < rot_threshold:
                return False

        for existing_pose in current_poses:
            trans_diff = np.linalg.norm(np.array(pose_summary['translation']) - np.array(existing_pose['translation']))
            rot_diff = np.linalg.norm(np.array(pose_summary['euler_angles_xyz_radians']) - np.array(existing_pose['euler_angles_xyz_radians']))
            if trans_diff < trans_threshold and rot_diff < rot_threshold:
                return False

        global_poses.append(pose_summary)
        write_global_poses(global_poses_file, global_poses)
        return True

def load_object(object_path):
    if object_path.endswith('.obj'):
        objs = bproc.loader.load_obj(object_path)
    elif object_path.endswith('.blend'):
        objs = bproc.loader.load_blend(object_path)
    elif object_path.endswith('.ply'):
        objs = bproc.loader.load_ply(object_path)
    elif object_path.endswith('.glb') or object_path.endswith('.gltf'):
        objs = bproc.loader.load_glb(object_path)
    else:
        raise ValueError(f"Unsupported object format: {object_path}")
    return objs

def setup_lighting():
    light_positions = [
        [0, 0, 20],     
        [15, 0, 10],    
        [-15, 0, 10],   
        [0, 15, 10],    
        [0, -15, 10],   
    ]
    
    for i, pos in enumerate(light_positions):
        light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=pos)
        light_plane.set_name(f"light_plane_{i}")
        mat = bproc.material.create(name=f"light_material_{i}")
        mat.make_emissive(emission_strength=10.0, emission_color=[1, 1, 1, 1])
        try:
            mat.set_principled_shader_value("Alpha", 0.0)
            mat.set_principled_shader_value("Transmission", 1.0)
        except Exception:
            pass
        light_plane.add_material(mat)
        light_plane.hide_render(True)

    bproc.world.set_world_background_color([0.05, 0.05, 0.05, 0.0])

    sun_light = bproc.types.Light()
    sun_light.set_type("SUN")
    sun_light.set_location([10, 10, 15])
    sun_light.set_energy(6.0)
    sun_light.set_color([1.0, 1.0, 1.0])

def get_object_bounds(objs):
    all_vertices = []
    for obj in objs:
        mesh = obj.get_mesh()
        world_matrix = obj.get_local2world_mat()
        for v in mesh.vertices:
            local_pos = Vector((v.co[0], v.co[1], v.co[2], 1.0))
            world_pos = world_matrix @ local_pos
            all_vertices.append(Vector((world_pos[0], world_pos[1], world_pos[2])))
    if not all_vertices:
        return Vector((0, 0, 0)), 1.0
    min_coords = Vector([min(v[i] for v in all_vertices) for i in range(3)])
    max_coords = Vector([max(v[i] for v in all_vertices) for i in range(3)])
    center = (min_coords + max_coords) / 2
    size = max(max_coords - min_coords)
    return center, size

def poses_are_similar(pose1, pose2, trans_threshold=0.001, rot_threshold=0.001):
    trans_diff = np.linalg.norm(np.array(pose1['translation']) - np.array(pose2['translation']))
    rot_diff = np.linalg.norm(np.array(pose1['euler_angles_xyz_radians']) - np.array(pose2['euler_angles_xyz_radians']))
    return trans_diff < trans_threshold and rot_diff < rot_threshold

def write_pose_to_csv(pose_data, csv_file_path, is_first_write=False):
    fieldnames = ['frame_id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'rx_deg', 'ry_deg', 'rz_deg']
    mode = 'w' if is_first_write else 'a'
    with open(csv_file_path, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if is_first_write:
            writer.writeheader()
        writer.writerow({
            'frame_id': pose_data['frame_id'],
            'tx': pose_data['translation'][0],
            'ty': pose_data['translation'][1], 
            'tz': pose_data['translation'][2],
            'rx': pose_data['euler_angles_xyz_radians'][0],
            'ry': pose_data['euler_angles_xyz_radians'][1],
            'rz': pose_data['euler_angles_xyz_radians'][2],
            'rx_deg': pose_data['euler_angles_xyz_degrees'][0],
            'ry_deg': pose_data['euler_angles_xyz_degrees'][1],
            'rz_deg': pose_data['euler_angles_xyz_degrees'][2]
        })

def load_background_images(bg_folder):
    if bg_folder is None:
        return []
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
    bg_images = []
    for ext in exts:
        bg_images.extend(glob(os.path.join(bg_folder, ext)))
    bg_images = sorted(bg_images)
    if not bg_images:
        print(f"Warning: No background images found in {bg_folder}")
    return bg_images

def generate_variance_controlled_poses(num_poses, center, size, 
                                     translation_variance=0.5,
                                     rotation_variance=0.5,
                                     variance_tolerance=0.1,
                                     max_attempts=20000,
                                     seed=42,
                                     output_dir="./output_variance_controlled",
                                     global_poses_file="./global_generated_poses.json",
                                     lock_file="./global_generated_poses.lock"):
    np.random.seed(seed)
    print(f"Generating {num_poses} poses with controlled variance:")
    print(f"  Translation variance: {translation_variance:.3f}")
    print(f"  Rotation variance: {rotation_variance:.3f}")
    print(f"  Variance tolerance: ±{variance_tolerance:.3f}")
    
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, 'poses.csv')

    poses = []
    base_trans_range = size * 2.0
    trans_std = translation_variance * base_trans_range * 0.3
    base_rot_range = 2 * np.pi
    rot_std = rotation_variance * base_rot_range * 0.3  
    print(f"  Translation std dev: {trans_std:.3f} units")
    print(f"  Rotation std dev: {np.degrees(rot_std):.1f} degrees")
    
    attempts = 0
    best_poses = None
    best_variance_error = float('inf')
    
    while attempts < max_attempts:
        attempts += 1
        current_poses = []
        first_write = True
        
        for frame_id in range(num_poses):
            max_duplicates = 1000
            duplicate_attempts = 0
            
            while duplicate_attempts < max_duplicates:
                tx = center.x + np.random.normal(0, trans_std)
                ty = center.y + np.random.normal(0, trans_std)
                tz = center.z + np.random.normal(0, trans_std)
                rx = np.random.normal(np.pi, rot_std)  
                ry = np.random.normal(np.pi, rot_std)  
                rz = np.random.normal(np.pi, rot_std)              
                rx = rx % (2 * np.pi)
                ry = ry % (2 * np.pi)
                rz = rz % (2 * np.pi)
                
                translation = Vector((tx, ty, tz))
                euler = Euler((rx, ry, rz), 'XYZ')
                rotation_matrix = euler.to_matrix().to_4x4()
                transformation_matrix = Matrix.Translation(translation) @ rotation_matrix
                R_matrix = rotation_matrix[:3][:3]  
                T_vector = translation  
                
                pose_data = {
                    'frame_id': frame_id,
                    'transformation_matrix': transformation_matrix,
                    'R_matrix_3x3': [list(row) for row in R_matrix],
                    'T_vector': list(T_vector),
                    'translation': [tx, ty, tz],
                    'euler_angles_xyz_radians': [rx, ry, rz],
                    'euler_angles_xyz_degrees': [np.degrees(rx), np.degrees(ry), np.degrees(rz)],
                    'roll_pitch_yaw_radians': [rx, ry, rz],
                    'roll_pitch_yaw_degrees': [np.degrees(rx), np.degrees(ry), np.degrees(rz)]
                }

                reserved = check_and_reserve_pose(pose_data, current_poses, global_poses_file, lock_file)

                if reserved:
                    current_poses.append(pose_data)
                    write_pose_to_csv(pose_data, csv_file_path, first_write)
                    first_write = False
                    break

                duplicate_attempts += 1

            if duplicate_attempts >= max_duplicates:
                print(f"Warning: Could not generate unique pose for frame {frame_id} after {max_duplicates} attempts")
                break
        
        if len(current_poses) < num_poses:
            continue
            
        translations = np.array([pose['translation'] for pose in current_poses])
        rotations = np.array([pose['euler_angles_xyz_radians'] for pose in current_poses])
        trans_actual_var = np.mean(np.var(translations, axis=0))  
        rot_actual_var = np.mean(np.var(rotations, axis=0))      
        trans_actual_var_norm = trans_actual_var / (base_trans_range * 0.3) ** 2
        rot_actual_var_norm = rot_actual_var / (base_rot_range * 0.3) ** 2
        trans_error = abs(trans_actual_var_norm - translation_variance)
        rot_error = abs(rot_actual_var_norm - rotation_variance)
        total_error = trans_error + rot_error
        
        if trans_error <= variance_tolerance and rot_error <= variance_tolerance:
            poses = current_poses
            print(f"  Target trans variance: {translation_variance:.3f}, Actual: {trans_actual_var_norm:.3f}")
            print(f"  Target rot variance: {rotation_variance:.3f}, Actual: {rot_actual_var_norm:.3f}")
            break
            
        if total_error < best_variance_error:
            best_variance_error = total_error
            best_poses = current_poses
            best_trans_var = trans_actual_var_norm
            best_rot_var = rot_actual_var_norm
            
        if attempts % 1000 == 0:
            print(f"  Attempt {attempts}")
    
    if not poses:
        poses = best_poses
        print(f"  Target trans variance: {translation_variance:.3f}, Best achieved: {best_trans_var:.3f}")
        print(f"  Target rot variance: {rotation_variance:.3f}, Best achieved: {best_rot_var:.3f}")
        print(f"  Total variance error: {best_variance_error:.4f}")
        
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['frame_id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'rx_deg', 'ry_deg', 'rz_deg']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for pose in poses:
                writer.writerow({
                    'frame_id': pose['frame_id'],
                    'tx': pose['translation'][0],
                    'ty': pose['translation'][1], 
                    'tz': pose['translation'][2],
                    'rx': pose['euler_angles_xyz_radians'][0],
                    'ry': pose['euler_angles_xyz_radians'][1],
                    'rz': pose['euler_angles_xyz_radians'][2],
                    'rx_deg': pose['euler_angles_xyz_degrees'][0],
                    'ry_deg': pose['euler_angles_xyz_degrees'][1],
                    'rz_deg': pose['euler_angles_xyz_degrees'][2]
                })
    
    if poses:
        translations = np.array([pose['translation'] for pose in poses])
        rotations = np.array([pose['euler_angles_xyz_radians'] for pose in poses])
        trans_vars = np.var(translations, axis=0)
        rot_vars = np.var(rotations, axis=0)
        
        variance_stats = {
            'target_translation_variance': translation_variance,
            'target_rotation_variance': rotation_variance,
            'actual_translation_variance_normalized': float(np.mean(trans_vars) / (base_trans_range * 0.3) ** 2),
            'actual_rotation_variance_normalized': float(np.mean(rot_vars) / (base_rot_range * 0.3) ** 2),
            'translation_variance_per_axis': {
                'x': float(trans_vars[0]),
                'y': float(trans_vars[1]), 
                'z': float(trans_vars[2])
            },
            'rotation_variance_per_axis': {
                'roll': float(rot_vars[0]),
                'pitch': float(rot_vars[1]),
                'yaw': float(rot_vars[2])
            },
            'translation_std_dev': float(np.sqrt(np.mean(trans_vars))),
            'rotation_std_dev_degrees': float(np.degrees(np.sqrt(np.mean(rot_vars)))),
            'generation_attempts': attempts,
            'variance_tolerance': variance_tolerance
        }
        
        print(f"  Translation std dev: {variance_stats['translation_std_dev']:.3f} units")
        print(f"  Rotation std dev: {variance_stats['rotation_std_dev_degrees']:.1f} degrees")
    else:
        variance_stats = {}
    
    generation_info = {
        'total_requested': num_poses,
        'total_generated': len(poses) if poses else 0,
        'total_attempts': attempts,
        'target_translation_variance': translation_variance,
        'target_rotation_variance': rotation_variance,
        'variance_tolerance': variance_tolerance,
        'variance_statistics': variance_stats,
        'seed': seed,
        'global_poses_file': global_poses_file,
        'lock_file': lock_file
    }
    
    return poses if poses else [], generation_info

def setup_camera_and_render_poses(objs, poses, generation_info, output_dir, camera_distance_factor=3.0, bg_folder=None):
    center, size = get_object_bounds(objs)
    camera_distance = size * camera_distance_factor
    camera_location = center + Vector((camera_distance, 0, camera_distance * 0.5))
    direction = (center - camera_location).normalized()
    world_up = Vector((0, 0, 1))
    if abs(direction.dot(world_up)) > 0.999:
        world_up = Vector((1, 0, 0))
    right = direction.cross(world_up).normalized()
    up = right.cross(direction).normalized()
    rotation_matrix = Matrix([
        [right.x, up.x, -direction.x, 0],
        [right.y, up.y, -direction.y, 0],
        [right.z, up.z, -direction.z, 0],
        [0, 0, 0, 1]
    ])
    camera_matrix = Matrix.Translation(camera_location) @ rotation_matrix
    image_w, image_h = 512, 512
    bproc.camera.set_resolution(image_w, image_h)
    fov_degrees = 40
    fov_radians = np.radians(fov_degrees)
    focal_length_px = (image_w / 2) / np.tan(fov_radians / 2)
    bproc.camera.set_intrinsics_from_K_matrix(
        K=[[focal_length_px, 0, image_w/2],
           [0, focal_length_px, image_h/2],
           [0, 0, 1]],
        image_width=image_w,
        image_height=image_h
    )
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()

    bg_images = load_background_images(bg_folder)

    original_transforms = []
    for obj in objs:
        original_transforms.append(obj.get_local2world_mat().copy())
    all_pose_data = []    
    for i, pose_data in enumerate(poses):
        if i % 50 == 0:
            print(f"Processing frame {i+1}/{len(poses)}")
        
        try:
            for obj in objs:
                obj.set_local2world_mat(pose_data['transformation_matrix'])

            bproc.camera.add_camera_pose(camera_matrix)
            data = bproc.renderer.render()

            if len(data.get('colors', [])) > 0:
                try:
                    rgb_image = data['colors'][0]

                    if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
                        rgb_u8 = (np.clip(rgb_image, 0.0, 1.0) * 255).astype(np.uint8)
                    else:
                        rgb_u8 = rgb_image.astype(np.uint8)

                    if rgb_u8.shape[2] == 4:
                        fg_img = Image.fromarray(rgb_u8, mode='RGBA')
                    else:
                        fg_img = Image.fromarray(rgb_u8, mode='RGB').convert('RGBA')

                    if bg_images:
                        bg_path = random.choice(bg_images)
                        bg_img = Image.open(bg_path).convert('RGBA')
                        bg_img = bg_img.resize((image_w, image_h), Image.ANTIALIAS)
                        composite = Image.alpha_composite(bg_img, fg_img)
                    else:
                        bg_img = Image.new('RGBA', (image_w, image_h), (13, 13, 13, 255))
                        composite = Image.alpha_composite(bg_img, fg_img)

                    frame_id = pose_data['frame_id']
                    os.makedirs(output_dir, exist_ok=True)
                    img_filename = f"image_{frame_id:04d}.png"
                    composite.convert('RGB').save(os.path.join(output_dir, img_filename))

                    if 'depth' in data and len(data['depth']) > 0:
                        depth_image = data['depth'][0]
                        if np.isfinite(depth_image).any():
                            valid = np.isfinite(depth_image)
                            if valid.any():
                                dmin = float(np.nanmin(depth_image[valid]))
                                dmax = float(np.nanmax(depth_image[valid]))
                                if dmax - dmin > 1e-6:
                                    depth_normalized = ((depth_image - dmin) / (dmax - dmin) * 255).astype(np.uint8)
                                else:
                                    depth_normalized = (np.clip(depth_image - dmin, 0, 1) * 255).astype(np.uint8)
                            else:
                                depth_normalized = (np.zeros_like(depth_image) * 255).astype(np.uint8)
                        else:
                            depth_normalized = (np.zeros_like(depth_image) * 255).astype(np.uint8)

                        depth_filename = f"depth_{frame_id:04d}.png"
                        Image.fromarray(depth_normalized, mode='L').save(os.path.join(output_dir, depth_filename))

                    object_position = Vector(pose_data['T_vector'])
                    distance_from_camera = (camera_location - object_position).length
                    complete_pose_data = {
                        'frame_id': frame_id,
                        'image_filename': img_filename,
                        'object_R_matrix': pose_data['R_matrix_3x3'],  
                        'object_T_vector': pose_data['T_vector'],      
                        'object_distance_from_camera': float(distance_from_camera),  
                        'object_transformation_matrix': [list(row) for row in pose_data['transformation_matrix']],
                        'translation': pose_data['translation'],
                        'euler_angles_xyz_radians': pose_data['euler_angles_xyz_radians'],
                        'euler_angles_xyz_degrees': pose_data['euler_angles_xyz_degrees'],
                        'roll_pitch_yaw_radians': pose_data['roll_pitch_yaw_radians'],
                        'roll_pitch_yaw_degrees': pose_data['roll_pitch_yaw_degrees'],
                        'camera_location': list(camera_location),
                        'camera_matrix': [list(row) for row in camera_matrix],
                        'object_center': list(center),
                        'object_size': size
                    }
                    all_pose_data.append(complete_pose_data)

                except Exception as e:
                    print(f"Warning: Could not save image for frame {frame_id}: {e}")
                finally:
                    bproc.camera.set_camera_poses([])
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue
    for obj, original_transform in zip(objs, original_transforms):
        obj.set_local2world_mat(original_transform)
    camera_params = {
        'camera_location': list(camera_location),
        'camera_matrix': [list(row) for row in camera_matrix],
        'K_matrix': [[focal_length_px, 0, image_w/2], [0, focal_length_px, image_h/2], [0, 0, 1]],
        'focal_length_px': focal_length_px,
        'fov_radians': fov_radians,
        'fov_degrees': fov_degrees,
        'resolution': [image_w, image_h]
    }
    combined_data = {
        'rendering_type': 'fixed_camera_variance_controlled_6dof',
        'total_images': len(all_pose_data),
        'pose_data': sorted(all_pose_data, key=lambda x: x['frame_id']),
        'generation_info': generation_info,
        'camera_parameters': camera_params,
        'rendering_info': {
            'sequential_rendering': True,
            'object_center': list(center),
            'object_size': size,
            'camera_distance_factor': camera_distance_factor
        }
    }
    with open(os.path.join(output_dir, 'variance_controlled_poses_RT_matrices.json'), 'w') as f:
        json.dump(combined_data, f, indent=2)
    rt_matrices = []
    distances = []
    for data in sorted(all_pose_data, key=lambda x: x['frame_id']):
        rt_entry = {
            'frame_id': data['frame_id'],
            'image_filename': data['image_filename'],
            'R_matrix': data['object_R_matrix'],  
            'T_vector': data['object_T_vector'],  
            'distance_from_camera': data['object_distance_from_camera'],  
            'translation': data['translation'],   
            'roll_pitch_yaw_degrees': data['roll_pitch_yaw_degrees']
        }
        rt_matrices.append(rt_entry)
        distances.append(data['object_distance_from_camera'])
    distance_stats = {
        'min_distance': float(np.min(distances)) if distances else 0,
        'max_distance': float(np.max(distances)) if distances else 0,
        'mean_distance': float(np.mean(distances)) if distances else 0,
        'std_distance': float(np.std(distances)) if distances else 0
    }
    
    with open(os.path.join(output_dir, 'RT_matrices_only.json'), 'w') as f:
        json.dump({
            'description': 'Object R (rotation) and T (translation) matrices with camera distances - Variance Controlled Dataset',
            'format': {
                'R_matrix': '3x3 rotation matrix (row-major order)',
                'T_vector': '3x1 translation vector [x, y, z]',
                'distance_from_camera': 'Euclidean distance from camera to object center (units)',
                'translation': '3x1 translation vector [x, y, z] (same as T_vector)',
                'roll_pitch_yaw_degrees': '[roll, pitch, yaw] rotations in degrees'
            },
            'distance_statistics': distance_stats,
            'generation_info': generation_info,
            'rt_data': rt_matrices
        }, f, indent=2)
    
    return all_pose_data

def main():
    parser = argparse.ArgumentParser(description='Fixed Camera with Variance-Controlled 6DoF Object Poses')
    parser.add_argument('object_path', help='Path to the 3D object file (.obj, .blend, .ply, .glb)')
    parser.add_argument('--num_poses', type=int, default=10, help='Number of poses to generate')
    parser.add_argument('--translation_variance', type=float, default=0.5,
                       help='Target translation variance (0.0=tight cluster, 1.0=wide spread)')
    parser.add_argument('--rotation_variance', type=float, default=0.5,
                       help='Target rotation variance (0.0=small rotations, 1.0=full rotations)')
    parser.add_argument('--variance_tolerance', type=float, default=0.1,
                       help='Acceptable deviation from target variance')
    parser.add_argument('--max_attempts', type=int, default=20000,
                       help='Maximum attempts to achieve target variance')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', default='./output_variance_controlled', help='Output directory')
    parser.add_argument('--camera_distance_factor', type=float, default=3.0,
                       help='Camera distance factor relative to object size')
    parser.add_argument('--global_poses_file', default='./global_generated_poses.json',
                       help='Global JSON file used to reserve generated poses across runs')
    parser.add_argument('--lock_file', default='./global_generated_poses.lock',
                       help='Lock file used to atomically read/write the global poses file')
    parser.add_argument('--background_folder', default=None,
                       help='Folder containing background images (png/jpg). If omitted, a gray background is used.')
    args = parser.parse_args()

    bproc.init()
    print(f"Loading object from {args.object_path}")
    objs = load_object(args.object_path)
    center, size = get_object_bounds(objs)
    setup_lighting()
    poses, generation_info = generate_variance_controlled_poses(
        args.num_poses, center, size,
        args.translation_variance, args.rotation_variance,
        args.variance_tolerance, args.max_attempts, args.seed,
        args.output_dir, args.global_poses_file, args.lock_file
    )
    print(f"Rendering images with variance-controlled poses to {args.output_dir}")
    pose_data = setup_camera_and_render_poses(
        objs, poses, generation_info, args.output_dir, args.camera_distance_factor, args.background_folder
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import blenderproc as bproc
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
import csv
import uuid
import random
from glob import glob
from mathutils import Matrix, Vector, Euler
from PIL import Image
import fcntl   


def load_global_poses(global_poses_file):
    if os.path.exists(global_poses_file):
        with open(global_poses_file, 'r') as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []


def write_global_poses(global_poses_file, poses_list) -> None:
    with open(global_poses_file, 'w') as f:
        json.dump(poses_list, f, indent=2)


def check_and_reserve_poses_batch(poses_data, global_poses_file, trans_threshold=0.001):
    """Batch reserve multiple poses at once to reduce file I/O"""
    reserved_poses = []
    
    # Lock using fcntl
    with open(global_poses_file, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            global_poses = json.load(f)
        except Exception:
            global_poses = []

        # Pre-compute existing translations for faster lookup
        existing_translations = np.array([p.get('T', p.get('translation', p.get('T_vector', [0,0,0])))[0:3] for p in global_poses if p.get('T') or p.get('translation') or p.get('T_vector')])
        
        for pose_data in poses_data:
            pose_summary = {
                'id': str(uuid.uuid4()),
                'T': pose_data['T']
            }
            
            current_T = np.array(pose_summary['T'])
            
            # Check against existing poses using vectorized operations
            if len(existing_translations) > 0:
                distances = np.linalg.norm(existing_translations - current_T, axis=1)
                if np.any(distances < trans_threshold):
                    continue
            
            # Check against already reserved poses in this batch
            if reserved_poses:
                reserved_translations = np.array([p['T'] for p in reserved_poses])
                distances = np.linalg.norm(reserved_translations - current_T, axis=1)
                if np.any(distances < trans_threshold):
                    continue
            
            reserved_poses.append(pose_summary)
            global_poses.append(pose_summary)

        # Write all at once
        f.seek(0)
        f.truncate()
        json.dump(global_poses, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
        
    return reserved_poses

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

    bproc.renderer.set_world_background(color=[0, 0, 0], strength=1.0)
    sun_light = bproc.types.Light()
    sun_light.set_type("SUN")
    sun_light.set_location([10, 10, 15])
    sun_light.set_energy(20.0)
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

def write_poses_to_csv_batch(poses_data, csv_file_path):
    """Write all poses to CSV in one batch operation"""
    fieldnames = ['frame_id', 'tx', 'ty', 'tz',
                  'r11','r12','r13','r21','r22','r23','r31','r32','r33',
                  'qw','qx','qy','qz']
    
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for pose in poses_data:
            row = {
                'frame_id': pose['frame_id'],
                'tx': pose['T'][0],
                'ty': pose['T'][1],
                'tz': pose['T'][2],
                'qw': pose['q'][0],
                'qx': pose['q'][1],
                'qy': pose['q'][2],
                'qz': pose['q'][3],
            }
            R = pose['R']
            row.update({
                'r11': R[0][0],'r12': R[0][1],'r13': R[0][2],
                'r21': R[1][0],'r22': R[1][1],'r23': R[1][2],
                'r31': R[2][0],'r32': R[2][1],'r33': R[2][2],
            })
            writer.writerow(row)

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
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, 'poses_reduced.csv')

    poses = []
    base_trans_range = size * 2.0
    trans_std = translation_variance * base_trans_range * 0.3
    base_rot_range = 2 * np.pi
    rot_std = rotation_variance * base_rot_range * 0.3

    best_poses = None
    best_variance_error = float('inf')
    best_trans_var = 0.0
    best_rot_var = 0.0

    for attempts in tqdm(range(max_attempts), desc="Generating poses"):
        current_poses = []
        
        # Generate poses one by one but with optimized checks
        for frame_id in range(num_poses):
            max_duplicates = 1000
            duplicate_attempts = 0
            pose_generated = False

            while duplicate_attempts < max_duplicates and not pose_generated:
                tx = float(center.x + np.random.normal(0, trans_std))
                ty = float(center.y + np.random.normal(0, trans_std))
                tz = float(center.z + np.random.normal(0, trans_std))

                rx = float(np.random.normal(np.pi, rot_std)) % (2 * np.pi)
                ry = float(np.random.normal(np.pi, rot_std)) % (2 * np.pi)
                rz = float(np.random.normal(np.pi, rot_std)) % (2 * np.pi)

                euler = Euler((rx, ry, rz), 'XYZ')
                R3 = euler.to_matrix()
                M4 = euler.to_matrix().to_4x4()
                M4.translation = Vector((tx, ty, tz))
                q = euler.to_quaternion()

                pose_data = {
                    'frame_id': frame_id,
                    'R': [[float(R3[i][j]) for j in range(3)] for i in range(3)],
                    'T': [tx, ty, tz],
                    'q': [float(q.w), float(q.x), float(q.y), float(q.z)],
                    'M': [list(row) for row in M4]
                }

                # Quick local check first (much faster)
                current_T = np.array(pose_data['T'])
                collision = False
                
                # Check against current poses (in memory, very fast)
                for existing_pose in current_poses:
                    existing_T = np.array(existing_pose['T'])
                    if np.linalg.norm(current_T - existing_T) < 0.001:
                        collision = True
                        break
                
                if not collision:
                    # Only do file I/O check if local check passes
                    reserved = check_and_reserve_poses_batch([pose_data], global_poses_file)
                    if reserved:
                        current_poses.append(pose_data)
                        pose_generated = True

                duplicate_attempts += 1

            if not pose_generated:
                break

        if len(current_poses) < num_poses:
            continue

        # Variance calculation (vectorized for speed)
        translations = np.array([pose['T'] for pose in current_poses], dtype=float)
        rotations = np.array([np.array(pose['R']).reshape(-1) for pose in current_poses], dtype=float)
        
        trans_actual_var = float(np.mean(np.var(translations, axis=0)))
        rot_actual_var = float(np.mean(np.var(rotations, axis=0)))

        trans_actual_var_norm = trans_actual_var / ((base_trans_range * 0.3) ** 2)
        rot_actual_var_norm = rot_actual_var / ((base_rot_range * 0.3) ** 2)

        trans_error = abs(trans_actual_var_norm - translation_variance)
        rot_error = abs(rot_actual_var_norm - rotation_variance)
        total_error = trans_error + rot_error

        if trans_error <= variance_tolerance and rot_error <= variance_tolerance:
            poses = current_poses
            best_trans_var = trans_actual_var_norm
            best_rot_var = rot_actual_var_norm
            break

        if total_error < best_variance_error:
            best_variance_error = total_error
            best_poses = current_poses
            best_trans_var = trans_actual_var_norm
            best_rot_var = rot_actual_var_norm

    if not poses:
        poses = best_poses or []

    # Batch write CSV at the end
    if poses:
        write_poses_to_csv_batch(poses, csv_file_path)

    generation_info = {
        'total_requested': num_poses,
        'total_generated': len(poses),
        'total_attempts': attempts + 1,
        'target_translation_variance': translation_variance,
        'target_rotation_variance': rotation_variance,
        'variance_tolerance': variance_tolerance,
        'variance_statistics': {
            'actual_translation_variance_normalized': float(best_trans_var) if poses else 0.0,
            'actual_rotation_variance_normalized': float(best_rot_var) if poses else 0.0
        },
        'seed': seed,
        'global_poses_file': global_poses_file,
        'lock_file': lock_file
    }

    # Write JSON once at the end
    if poses:
        reduced_json_path = os.path.join(output_dir, 'poses_reduced.json')
        with open(reduced_json_path, 'w') as f:
            json.dump(poses, f, indent=2)

    return poses, generation_info

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
    image_w, image_h = 1920, 1200
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
    
    # Pre-load and resize background images for faster access
    bg_images_loaded = []
    if bg_images:
        print("Pre-loading background images...")
        for bg_path in bg_images[:10]:  # Limit to 10 to save memory
            try:
                bg_img = Image.open(bg_path).convert('RGBA')
                bg_img = bg_img.resize((image_w, image_h), Image.ANTIALIAS)
                bg_images_loaded.append(bg_img)
            except Exception:
                continue
    
    # Default background
    default_bg = Image.new('RGBA', (image_w, image_h), (13, 13, 13, 255))

    original_transforms = []
    for obj in objs:
        original_transforms.append(obj.get_local2world_mat().copy())
        
    all_pose_data = []
    
    # Pre-compute camera distances for all poses
    camera_distances = []
    for pose_data in poses:
        object_position = Vector(pose_data['T'])
        distance = (camera_location - object_position).length
        camera_distances.append(float(distance))
    
    for i, pose_data in enumerate(tqdm(poses, desc="Rendering poses")):
        if i % 50 == 0:
            print(f"Processing frame {i+1}/{len(poses)}")

        try:
            # Use pre-computed matrix if available
            M_list = pose_data.get('M')
            if M_list is None:
                R = pose_data['R']
                T = pose_data['T']
                M4 = Matrix([[R[0][0], R[0][1], R[0][2], T[0]],
                             [R[1][0], R[1][1], R[1][2], T[1]],
                             [R[2][0], R[2][1], R[2][2], T[2]],
                             [0, 0, 0, 1]])
            else:
                M4 = Matrix(M_list)

            for obj in objs:
                obj.set_local2world_mat(M4)

            bproc.camera.add_camera_pose(camera_matrix)
            data = bproc.renderer.render()

            if len(data.get('colors', [])) > 0:
                try:
                    rgb_image = data['colors'][0]
                    if rgb_image.dtype in (np.float32, np.float64):
                        rgb_u8 = (np.clip(rgb_image, 0.0, 1.0) * 255).astype(np.uint8)
                    else:
                        rgb_u8 = rgb_image.astype(np.uint8)

                    if rgb_u8.shape[2] == 4:
                        fg_img = Image.fromarray(rgb_u8, mode='RGBA')
                    else:
                        fg_img = Image.fromarray(rgb_u8, mode='RGB').convert('RGBA')

                    # Use pre-loaded background images
                    if bg_images_loaded:
                        bg_img = random.choice(bg_images_loaded).copy()
                        composite = Image.alpha_composite(bg_img, fg_img)
                    else:
                        composite = Image.alpha_composite(default_bg.copy(), fg_img)

                    frame_id = pose_data['frame_id']
                    img_filename = f"image_{frame_id:04d}.png"
                    composite.convert('RGB').save(os.path.join(output_dir, img_filename))

                    # Simplified depth processing
                    if 'depth' in data and len(data['depth']) > 0:
                        depth_image = data['depth'][0]
                        # Quick depth normalization without extensive checks
                        valid_mask = np.isfinite(depth_image)
                        if valid_mask.any():
                            dmin, dmax = depth_image[valid_mask].min(), depth_image[valid_mask].max()
                            if dmax > dmin:
                                depth_normalized = ((depth_image - dmin) / (dmax - dmin) * 255).astype(np.uint8)
                            else:
                                depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
                        else:
                            depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)

                        depth_filename = f"depth_{frame_id:04d}.png"
                        Image.fromarray(depth_normalized, mode='L').save(os.path.join(output_dir, depth_filename))

                    complete_pose_data = {
                        'frame_id': frame_id,
                        'image_filename': img_filename,
                        'R': pose_data['R'],
                        'T': pose_data['T'],
                        'q': pose_data['q'],
                        'distance_from_camera': camera_distances[i],  # Use pre-computed distance
                    }
                    all_pose_data.append(complete_pose_data)

                except Exception as e:
                    print(f"Warning: Could not save image for frame {frame_id}: {e}")
                finally:
                    bproc.camera.set_camera_poses([])

        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue

    # Restore original transforms
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

    # Batch write all JSON files at the end
    combined_data = {
        'rendering_type': 'fixed_camera_variance_controlled_6dof_reduced',
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

    with open(os.path.join(output_dir, 'variance_controlled_poses_RT_reduced.json'), 'w') as f:
        json.dump(combined_data, f, indent=2)

    # Pre-compute distance statistics
    distances = [data['distance_from_camera'] for data in all_pose_data]
    distance_stats = {
        'min_distance': float(min(distances)) if distances else 0,
        'max_distance': float(max(distances)) if distances else 0,
        'mean_distance': float(np.mean(distances)) if distances else 0,
        'std_distance': float(np.std(distances)) if distances else 0
    }

    rt_matrices = [{
        'frame_id': data['frame_id'],
        'image_filename': data['image_filename'],
        'R_matrix': data['R'],
        'T_vector': data['T'],
        'distance_from_camera': data['distance_from_camera']
    } for data in sorted(all_pose_data, key=lambda x: x['frame_id'])]

    with open(os.path.join(output_dir, 'RT_matrices_only_reduced.json'), 'w') as f:
        json.dump({
            'description': 'Reduced R,T,q dataset (row-major R, T vector)',
            'distance_statistics': distance_stats,
            'generation_info': generation_info,
            'rt_data': rt_matrices
        }, f, indent=2)

    return all_pose_data

def main():
    parser = argparse.ArgumentParser(description='Fixed Camera with Variance-Controlled 6DoF Object Poses (reduced R,T,q)')
    parser.add_argument('object_path', help='Path to the 3D object file (.obj, .blend, .ply, .glb)')
    parser.add_argument('--num_poses', type=int, default=10, help='Number of poses to generate')
    parser.add_argument('--translation_variance', type=float, default=0.5,
                       help='Target translation variance (0.0=tight cluster, 1.0=wide spread)')
    parser.add_argument('--rotation_variance', type=float, default=0.5,
                       help='Target rotation variance (0.0=small rotations, 1.0=full rotations)')
    parser.add_argument('--variance_tolerance', type=float, default=0.1,
                       help='Acceptable deviation from target variance')
    parser.add_argument('--max_attempts', type=int, default=2,
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Folder created: {args.output_dir}")
    else:
        print(f"Folder already exists: {args.output_dir}")

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
    _ = setup_camera_and_render_poses(
        objs, poses, generation_info, args.output_dir, args.camera_distance_factor, args.background_folder
    )

if __name__ == "__main__":
    main()
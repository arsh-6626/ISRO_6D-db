#!/bin/bash

# Activate conda environment
conda activate base

# Configuration parameters
NUM_POSES=10                           
RUNS=5                                  
TRANSLATION_VARIANCE=0.7
ROTATION_VARIANCE=0.5
VARIANCE_TOLERANCE=0.1
MAX_ATTEMPTS=2
SEED=42                                 
CAMERA_DISTANCE_FACTOR=2.0
# BACKGROUND_FOLDER="./"      
GLOBAL_POSES_FILE="./global_generated_poses.json"
LOCK_FILE="./global_generated_poses.lock"
OBJECT_PATH="/home/cha0s/Downloads/uploads_files_5684222_Satellite (2)/Satellite/Render.blend"
SCRIPT_NAME="blend_py.py"
if [ ! -f "$OBJECT_PATH" ]; then
    echo "Error: Object file '$OBJECT_PATH' not found!"
    exit 1
fi

if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Python script '$SCRIPT_NAME' not found!"
    exit 1
fi

# Clear global poses file for fresh start
echo "[]" > "$GLOBAL_POSES_FILE"

echo "============================================"
echo "Parallel Pose Generation and Rendering"
echo "============================================"
echo "Object: $OBJECT_PATH"
echo "Instances: $RUNS"
echo "Poses per instance: $NUM_POSES"
echo "Total poses: $((RUNS * NUM_POSES))"
echo "Global poses file: $GLOBAL_POSES_FILE"
echo "============================================"

# Record start time
start_time=$(date +%s)

# Run instances in parallel
for i in $(seq 1 $RUNS); do
    OUTPUT_DIR="./output_run_$i"
    INSTANCE_SEED=$((SEED + i))  # Different seed for each instance
    
    echo ">>> Starting instance $i, saving to $OUTPUT_DIR (seed: $INSTANCE_SEED)"
    
    # Build command with background folder if specified
    cmd="blenderproc run $SCRIPT_NAME \"$OBJECT_PATH\" \
        --num_poses $NUM_POSES \
        --translation_variance $TRANSLATION_VARIANCE \
        --rotation_variance $ROTATION_VARIANCE \
        --variance_tolerance $VARIANCE_TOLERANCE \
        --max_attempts $MAX_ATTEMPTS \
        --seed $INSTANCE_SEED \
        --output_dir \"$OUTPUT_DIR\" \
        --camera_distance_factor $CAMERA_DISTANCE_FACTOR \
        --global_poses_file \"$GLOBAL_POSES_FILE\" \
        --lock_file \"$LOCK_FILE\""
    
    # Add background folder if specified and exists
    if [ -n "$BACKGROUND_FOLDER" ] && [ -d "$BACKGROUND_FOLDER" ]; then
        cmd="$cmd --background_folder \"$BACKGROUND_FOLDER\""
    fi
    
    # Run in background and redirect output to log file
    eval $cmd > "./output_run_${i}.log" 2>&1 &
    
    # Small delay to prevent simultaneous file access issues
    sleep 2
done

# Wait for all background processes to complete
echo "Waiting for all instances to complete..."
wait

# Record end time
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "============================================"
echo "✅ All runs completed!"
echo "Total runtime: ${duration} seconds"
echo "============================================"

# Collect statistics
total_generated=0
successful_runs=0

echo "Instance Results:"
echo "----------------"
for i in $(seq 1 $RUNS); do
    output_dir="./output_run_$i"
    log_file="./output_run_${i}.log"
    
    if [ -d "$output_dir" ] && [ -f "${output_dir}/poses_reduced.csv" ]; then
        # Count lines in CSV (subtract 1 for header)
        if [ -f "${output_dir}/poses_reduced.csv" ]; then
            csv_lines=$(wc -l < "${output_dir}/poses_reduced.csv" 2>/dev/null || echo "1")
            poses_count=$((csv_lines - 1))
            if [ $poses_count -lt 0 ]; then
                poses_count=0
            fi
        else
            poses_count=0
        fi
        
        total_generated=$((total_generated + poses_count))
        successful_runs=$((successful_runs + 1))
        echo "Run $i: $poses_count poses generated ✓"
    else
        echo "Run $i: Failed ✗"
        if [ -f "$log_file" ]; then
            echo "  Last few log lines:"
            tail -3 "$log_file" 2>/dev/null | sed 's/^/    /' || echo "    (no log available)"
        fi
    fi
done

echo "============================================"
echo "Final Statistics:"
echo "  Successful runs: $successful_runs/$RUNS"
echo "  Total poses generated: $total_generated"
if [ $successful_runs -gt 0 ]; then
    echo "  Average poses per successful run: $((total_generated / successful_runs))"
fi
if [ $total_generated -gt 0 ]; then
    echo "  Average time per pose: $((duration / total_generated)) seconds"
fi
echo "  Total runtime: ${duration} seconds"
echo "============================================"

read -p "Do you want to merge all results into a single dataset? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Merging results..."
    
    MERGED_DIR="./merged_output"
    mkdir -p "$MERGED_DIR"
    
    # Create Python script to merge results
    cat > merge_results.py << 'EOF'
import json
import csv
import os
import glob
import sys
from collections import defaultdict

def merge_csv_files(output_dir):
    """Merge all CSV files from different runs"""
    all_poses = []
    csv_files = glob.glob("./output_run_*/poses_reduced.csv")
    
    for csv_file in sorted(csv_files):
        if os.path.exists(csv_file):
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_poses.append(row)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    # Write merged CSV with renumbered frame_ids
    if all_poses:
        merged_csv = os.path.join(output_dir, "merged_poses.csv")
        with open(merged_csv, 'w', newline='') as f:
            fieldnames = ['frame_id', 'tx', 'ty', 'tz', 'r11','r12','r13','r21','r22','r23','r31','r32','r33', 'qw','qx','qy','qz']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, pose in enumerate(all_poses):
                pose['frame_id'] = i
                writer.writerow(pose)
        
        print(f"Merged {len(all_poses)} poses into {merged_csv}")
    
    return len(all_poses)

def merge_images(output_dir):
    """Copy and renumber all images"""
    image_dirs = glob.glob("./output_run_*")
    frame_counter = 0
    
    for run_dir in sorted(image_dirs):
        image_files = glob.glob(os.path.join(run_dir, "image_*.png"))
        depth_files = glob.glob(os.path.join(run_dir, "depth_*.png"))
        
        for img_file in sorted(image_files):
            new_name = f"image_{frame_counter:04d}.png"
            os.system(f"cp '{img_file}' '{os.path.join(output_dir, new_name)}'")
            frame_counter += 1
        
        frame_counter = 0  # Reset for depth files
        for depth_file in sorted(depth_files):
            new_name = f"depth_{frame_counter:04d}.png"
            os.system(f"cp '{depth_file}' '{os.path.join(output_dir, new_name)}'")
            frame_counter += 1
    
    return frame_counter

if __name__ == "__main__":
    output_dir = "./merged_output"
    
    print("Merging CSV files...")
    total_poses = merge_csv_files(output_dir)
    
    print("Copying and renumbering images...")
    total_images = merge_images(output_dir)
    
    # Create summary
    summary = {
        "total_runs": int(os.environ.get("RUNS", 0)),
        "successful_poses": total_poses,
        "total_images": total_images,
        "generation_parameters": {
            "num_poses_per_run": int(os.environ.get("NUM_POSES", 0)),
            "translation_variance": float(os.environ.get("TRANSLATION_VARIANCE", 0)),
            "rotation_variance": float(os.environ.get("ROTATION_VARIANCE", 0)),
            "variance_tolerance": float(os.environ.get("VARIANCE_TOLERANCE", 0)),
            "max_attempts": int(os.environ.get("MAX_ATTEMPTS", 0)),
            "camera_distance_factor": float(os.environ.get("CAMERA_DISTANCE_FACTOR", 0))
        },
        "object_file": os.environ.get("OBJECT_PATH", "")
    }
    
    with open(os.path.join(output_dir, "merge_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Merge completed: {total_poses} poses, {total_images} images")
EOF
    
    # Export environment variables for the Python script
    export RUNS NUM_POSES TRANSLATION_VARIANCE ROTATION_VARIANCE VARIANCE_TOLERANCE
    export MAX_ATTEMPTS CAMERA_DISTANCE_FACTOR OBJECT_PATH
    
    # Run merge script
    python3 merge_results.py
    
    # Clean up temporary merge script
    rm merge_results.py
    
    echo "✅ Merge completed! Check ./merged_output/ for combined results."
fi

echo "Script finished!"
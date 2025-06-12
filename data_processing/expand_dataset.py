

import json
import os
from glob import glob
import tempfile
from typing import List, Tuple
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

from data_processing.amplitudinal import amplitudinal_downsample
from data_processing.spatial import spatial_downsample

# Assuming these are defined elsewhere or passed in
# from data_processing.spatial import spatial_downsample
# from data_processing.amplitudinal import amplitudinal_downsample

# --- Helper functions for multiprocessing ---

# Function to encapsulate a single spatial task
def _process_spatial_task(args: Tuple[str, str, List[str], float, str, str]) -> None:
    """Helper for spatial downsampling in parallel."""
    img_path, label_path, label_values_to_scale, scale, output_img_dir, output_label_dir = args
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    spatial_img, label_dict = spatial_downsample(img_path, label_path, label_values_to_scale, scale)
    spatial_out_path = os.path.join(output_img_dir, f"{base_name}_spatial_{scale:.2f}.png")
    spatial_img.save(spatial_out_path)
    with open(os.path.join(output_label_dir, f"{base_name}_spatial_{scale:.2f}.json"), 'w') as f:
        json.dump(label_dict, f)

# Function to encapsulate a single amplitudinal task
def _process_amplitudinal_task(args: Tuple[str, str, int, str, str]) -> None:
    """Helper for amplitudinal downsampling in parallel."""
    img_path, label_path, qp, output_img_dir, output_label_dir = args
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Load label only once per image, as it's typically just copied for amplitude changes
    with open(label_path, 'r') as f:
        label_dict = json.load(f)

    amp_img = amplitudinal_downsample(img_path, qp)
    amp_out_path = os.path.join(output_img_dir, f"{base_name}_qp{qp}_out.png")
    amp_img.save(amp_out_path)
    with open(os.path.join(output_label_dir, f"{base_name}_qp{qp}.json"), 'w') as f:
        json.dump(label_dict, f)

# Function to encapsulate a single mixed task
def _process_mixed_task(args: Tuple[str, str, List[str], float, int, str, str]) -> None:
    """Helper for mixed (spatial then amplitudinal) downsampling in parallel."""
    img_path, label_path, label_values_to_scale, scale, qp, output_img_dir, output_label_dir = args
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    spatial_img, label_dict = spatial_downsample(img_path, label_path, label_values_to_scale, scale)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        spatial_img.save(temp_path) # Save spatial output to temp file for BPG input
    try:
        mixed_img = amplitudinal_downsample(temp_path, qp)
        mixed_out_path = os.path.join(output_img_dir, f"{base_name}_spatial_{scale:.2f}_qp{qp}.png")
        mixed_img.save(mixed_out_path)
        with open(os.path.join(output_label_dir, f"{base_name}_spatial_{scale:.2f}_qp{qp}.json"), 'w') as f:
            json.dump(label_dict, f)
    finally:
        if os.path.exists(temp_path): # Ensure temp file exists before trying to remove
            os.remove(temp_path)

# --- Main expansion function ---

def expand_dataset(
    input_dir: str,
    label_dir: str,
    label_values_to_scale: List[str],
    output_img_dir: str,
    output_label_dir: str,
    scale_factors: list,
    qp_values: list,
    expansion: str = "spatial",
    subsample_spatial: bool = True,
    subsample_amplitudinal: bool = True,
    cpu_count = 5, 
):
    """
    Expands the dataset by applying spatial and amplitudinal downsampling to images,
    using parallel processing.

    Args:
        input_dir (str): Directory containing input images.
        label_dir (str): Directory containing input label JSON files.
        label_values_to_scale (List[str]): List of label values to scale during spatial downsampling.
        output_img_dir (str): Directory to save the processed images.
        output_label_dir (str): Directory to save the processed label JSON files.
        scale_factors (list): List of scale factors for spatial downsampling.
        qp_values (list): List of quantization parameters for amplitudinal downsampling.
        expansion (str): Type of expansion: "spatial", "amplitudinal", or "mixed".
    Returns:
        None
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_paths = sorted(glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png")))
    label_paths = sorted(glob(os.path.join(label_dir, "*.json")))

    zipped = match_images_labels(image_paths, label_paths, partial_match=True)
    image_paths, label_paths = zip(*zipped) if zipped else ([], [])


    tasks = []
    task_description = ""

    # Prepare tasks based on the expansion type
    if expansion == "spatial":
        task_description = "Processing spatial downsampling"
        if not subsample_spatial:
            for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
                if not Path(label_path).stem in Path(img_path).stem:
                    print(f"Warning: Image {img_path} and label {label_path} do not match. Skipping this pair.")
                    continue

                assert Path(label_path).stem in Path(img_path).stem, "Image and label file names must match."
                
                for scale in scale_factors:
                    tasks.append((img_path, label_path, label_values_to_scale, scale, output_img_dir, output_label_dir))
        else: 
            for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
                if not Path(label_path).stem in Path(img_path).stem:
                    print(f"Warning: Image {img_path} and label {label_path} do not match. Skipping this pair.")
                    continue

                assert Path(label_path).stem in Path(img_path).stem, "Image and label file names must match."
                scale_factor_index = int(i // (len(image_paths) // len(scale_factors)))
                tasks.append((img_path, label_path, label_values_to_scale, scale_factors[scale_factor_index], output_img_dir, output_label_dir))
        process_func = _process_spatial_task

    elif expansion == "amplitudinal":
        task_description = "Processing amplitudinal downsampling"
        for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
            if not Path(label_path).stem in Path(img_path).stem:
                print(f"Warning: Image {img_path} and label {label_path} do not match. Skipping this pair.")
                continue

            assert Path(label_path).stem in Path(img_path).stem, "Image and label file names must match."
            for qp in qp_values:
                tasks.append((img_path, label_path, qp, output_img_dir, output_label_dir))
        process_func = _process_amplitudinal_task

    elif expansion == "mixed":
        task_description = "Processing mixed (spatial & amplitudinal) downsampling"

        if subsample_amplitudinal and subsample_spatial:
            scale_factor_index = 0 
            qp_index = 0
            for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
                if not Path(label_path).stem in Path(img_path).stem:
                    print(f"Warning: Image {img_path} and label {label_path} do not match. Skipping this pair.")
                    continue

                assert Path(label_path).stem in Path(img_path).stem, "Image and label file names must match."

                scale_factor_index = scale_factor_index % len(scale_factors)
                qp_index = qp_index % len(qp_values)
                tasks.append((img_path, label_path, label_values_to_scale, scale_factors[scale_factor_index], qp_values[qp_index], output_img_dir, output_label_dir))
                scale_factor_index += 1
                qp_index += 1
        elif subsample_amplitudinal:
            for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
                if not Path(label_path).stem in Path(img_path).stem:
                    print(f"Warning: Image {img_path} and label {label_path} do not match. Skipping this pair.")
                    continue

                assert Path(label_path).stem in Path(img_path).stem, "Image and label file names must match."
                qp_index = int(i // (len(image_paths) // len(qp_values)))
                for scale in scale_factors:
                    tasks.append((img_path, label_path, label_values_to_scale, scale, qp_values[qp_index], output_img_dir, output_label_dir))
        elif subsample_spatial:
            for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
                if not Path(label_path).stem in Path(img_path).stem:
                    print(f"Warning: Image {img_path} and label {label_path} do not match. Skipping this pair.")
                    continue

                assert Path(label_path).stem in Path(img_path).stem, "Image and label file names must match."
                
                
                scale_factor_index = int(i // (len(image_paths) // len(scale_factors)))
                for qp_index in range(len(qp_values)):
                    tasks.append((img_path, label_path, label_values_to_scale, scale_factors[scale_factor_index], qp_values[qp_index], output_img_dir, output_label_dir))
        else: 
            for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
                
                if not Path(label_path).stem in Path(img_path).stem:
                    print(f"Warning: Image {img_path} and label {label_path} do not match. Skipping this pair.")
                    continue

                assert Path(label_path).stem in Path(img_path).stem, "Image and label file names must match."
                for scale in scale_factors:
                    for qp in qp_values:
                        tasks.append((img_path, label_path, label_values_to_scale, scale, qp, output_img_dir, output_label_dir))
        process_func = _process_mixed_task

    else:
        raise ValueError("Invalid expansion type. Choose 'spatial', 'amplitudinal', or 'mixed'.")

    # Execute tasks in parallel
    if not tasks:
        print("No tasks generated. Check input directories and parameters.")
        return

    # Use ProcessPoolExecutor for parallel processing
    # Set max_workers based on your CPU cores (leave one free for OS responsiveness)
    # Using 'mp.cpu_count()' or 'os.cpu_count()' if you import os
    num_workers = max(os.cpu_count() - 1, cpu_count) if os.cpu_count() > 1 else 1 # Ensure at least 1 worker
    print(f"Starting {task_description} with {num_workers} parallel processes...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_func, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc=task_description):
            try:
                # Retrieve result to catch exceptions raised in worker processes
                future.result()
            except Exception as exc:
                task_args = futures[future]
                print(f'\nError processing task with arguments {task_args}: {exc}')

    print(f"All images expanded and saved to {output_img_dir} and {output_label_dir}")
    
    
from pathlib import Path
from typing import List, Tuple

def match_images_labels(
    image_paths: List[str], 
    label_paths: List[str], 
    partial_match: bool = True
) -> List[Tuple[str, str]]:
    """
    Matches images and label files based on their filename stems.
    
    Args:
        image_paths (List[str]): List of image file paths.
        label_paths (List[str]): List of label file paths.
        partial_match (bool): If True, label stem can be contained within image stem.
                              If False, requires exact stem match.
                              
    Returns:
        List[Tuple[str, str]]: List of (image_path, label_path) pairs matched.
    """
    image_dict = {Path(p).stem: p for p in image_paths}
    label_dict = {Path(p).stem: p for p in label_paths}
    matched_pairs = []

    for label_stem, label_path in label_dict.items():
        if partial_match:
            # Find all image stems containing label stem
            candidates = [img_path for img_stem, img_path in image_dict.items() if label_stem in img_stem]
            if candidates:
                for img_path in candidates:
                    matched_pairs.append((img_path, label_path))
            else:
                print(f"No matching image found for label: {label_path}")
        else:
            # Exact stem match required
            if label_stem in image_dict:
                matched_pairs.append((image_dict[label_stem], label_path))
            else:
                print(f"No exact matching image found for label: {label_path}")

    return matched_pairs

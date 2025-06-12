# import json
# import os
# from glob import glob
# import tempfile
# from typing import List
# from data_processing.spatial import spatial_downsample
# from data_processing.amplitudinal import amplitudinal_downsample
# from tqdm import tqdm


# def expand_dataset(input_dir: str, label_dir: str, label_values_to_scale:List[str], output_img_dir: str, output_label_dir: str, scale_factors: list, qp_values: list, expansion="spatial"):
#     """
#     Expands the dataset by applying spatial and amplitudinal downsampling to images.
#     Args:
#         input_dir (str): Directory containing input images.
#         output_dir (str): Directory to save the processed images.
#         scale_factors (list): List of scale factors for spatial downsampling.
#         qp_values (list): List of quantization parameters for amplitudinal downsampling.
#     Returns:
#         None
#     """
#     os.makedirs(output_img_dir, exist_ok=True)
#     os.makedirs(output_label_dir, exist_ok=True)
#     image_paths = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png"))
#     label_paths = glob(os.path.join(label_dir, "*.json"))

#     for img_path, label_path in tqdm(list(zip(image_paths, label_paths)), desc="Processing images"):

#         base_name = os.path.splitext(os.path.basename(img_path))[0]

#         if expansion == "spatial":
#             # Only spatial downsampling
#             for scale in scale_factors:
#                 spatial_img, label_dict = spatial_downsample(img_path,label_path, label_values_to_scale, scale)
#                 spatial_out_path = os.path.join(
#                     output_img_dir, f"{base_name}_spatial_{scale:.2f}.png"
#                 )
#                 spatial_img.save(spatial_out_path)
#                 with open(os.path.join(output_label_dir, f"{base_name}_spatial_{scale:.2f}.json"), 'w') as f:
#                     json.dump(label_dict, f)

#         if expansion == "amplitudinal":
#             # Only amplitude downsampling
#             with open(label_path, 'r') as f:
#                 label_dict = json.load(f)
#             for qp in qp_values:
#                 amp_img = amplitudinal_downsample(img_path, qp)
#                 amp_out_path = os.path.join(output_img_dir, f"{base_name}_qp{qp}_out.png")
#                 amp_img.save(amp_out_path)
#                 with open(os.path.join(output_label_dir, f"{base_name}_qp{qp}.json"), 'w') as f:
#                     json.dump(label_dict, f)
#         if expansion == "mixed":
#             # Mixed: spatial then amplitude
#             for scale in scale_factors:
#                 spatial_img, label_dict = spatial_downsample(img_path,label_path, label_values_to_scale, scale)

#                 with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
#                     temp_path = tmp.name
#                     spatial_img.save(temp_path)
#                 try:
#                     for qp in qp_values:
#                         mixed_img = amplitudinal_downsample(temp_path, qp)
#                         mixed_out_path = os.path.join(
#                             output_img_dir, f"{base_name}_spatial_{scale:.2f}_qp{qp}.png"
#                         )
#                         mixed_img.save(mixed_out_path)
#                         with open(os.path.join(output_label_dir, f"{base_name}_spatial_{scale:.2f}_qp{qp}.json"), 'w') as f:
#                             json.dump(label_dict, f)
#                 finally:
#                     os.remove(temp_path)
#     print(f"Images expanded and saved to {output_img_dir} and {output_label_dir}")


import json
import os
from glob import glob
import tempfile
from typing import List, Tuple
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path
from tqdm import tqdm

from data_processing.amplitudinal import amplitudinal_downsample
from data_processing.spatial import spatial_downsample

# Assuming these are defined elsewhere or passed in
# from data_processing.spatial import spatial_downsample
# from data_processing.amplitudinal import amplitudinal_downsample

# --- Helper functions for multiprocessing ---

# Function to encapsulate a single spatial task
def _process_spatial_task(args: Tuple[str, str, List[str], float, str, str, str]) -> None:
    """Helper for spatial downsampling in parallel."""
    img_path, label_path, label_values_to_scale, scale, output_img_dir, output_label_dir, metadata_dir = args
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    spatial_img, label_dict = spatial_downsample(img_path, label_path, label_values_to_scale, scale)
    spatial_out_path = os.path.join(output_img_dir, f"{base_name}_spatial_{scale:.2f}.png")
    spatial_img.save(spatial_out_path)
    with open(os.path.join(output_label_dir, f"{base_name}_spatial_{scale:.2f}.json"), 'w') as f:
        json.dump(label_dict, f)

# Function to encapsulate a single amplitudinal task
def _process_amplitudinal_task(args: Tuple[str, str, int, str, str, str]) -> None:
    """Helper for amplitudinal downsampling in parallel."""
    img_path, label_path, qp, output_img_dir, output_label_dir, metadata_dir = args
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Load label only once per image, as it's typically just copied for amplitude changes
    with open(label_path, 'r') as f:
        label_dict = json.load(f)

    amp_img, image_size_mb, psnr = amplitudinal_downsample(img_path, qp)
    amp_out_path = os.path.join(output_img_dir, f"{base_name}_qp{qp}_out.png")
    amp_img.save(amp_out_path)
    with open(os.path.join(output_label_dir, f"{base_name}_qp{qp}.json"), 'w') as f:
        json.dump(label_dict, f)
    with open(os.path.join(metadata_dir, f"{base_name}_qp{qp}.json"), 'w') as f:
        json.dump({"image_size_mb": image_size_mb, "psnr": psnr}, f)

# Function to encapsulate a single mixed task
def _process_mixed_task(args: Tuple[str, str, List[str], float, int, str, str, str]) -> None:
    """Helper for mixed (spatial then amplitudinal) downsampling in parallel."""
    img_path, label_path, label_values_to_scale, scale, qp, output_img_dir, output_label_dir, metadata_dir = args
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    spatial_img, label_dict = spatial_downsample(img_path, label_path, label_values_to_scale, scale)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        spatial_img.save(temp_path) # Save spatial output to temp file for BPG input
    try:
        mixed_img, image_size_mb, psnr = amplitudinal_downsample(temp_path, qp)
        mixed_out_path = os.path.join(output_img_dir, f"{base_name}_spatial_{scale:.2f}_qp{qp}.png")
        mixed_img.save(mixed_out_path)
        with open(os.path.join(output_label_dir, f"{base_name}_spatial_{scale:.2f}_qp{qp}.json"), 'w') as f:
            json.dump(label_dict, f)
        with open(os.path.join(metadata_dir, f"{base_name}_spatial_{scale:.2f}_qp{qp}.json"), 'w') as f:
            json.dump({"image_size_mb": image_size_mb, "psnr": psnr}, f)
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
    metadata_dir: str = None,
    expansion: str = "spatial",
    subsample_spatial: bool = True,
    subsample_amplitudinal: bool = True
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

    if metadata_dir is not None:
        os.makedirs(metadata_dir, exist_ok=True)

    input_dir = str(Path(input_dir).resolve().absolute())
    label_dir = str(Path(label_dir).resolve().absolute())
    output_img_dir = str(Path(output_img_dir).resolve().absolute())
    output_label_dir = str(Path(output_label_dir).resolve().absolute())
    if metadata_dir is not None:
        metadata_dir = str(Path(metadata_dir).resolve().absolute())

    image_paths = sorted(glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png")))
    label_paths = sorted(glob(os.path.join(label_dir, "*.json")))

    if len(image_paths) != len(label_paths):
        print("Images", len(image_paths), "Label paths", len(label_paths))
        print("Warning: Number of images and label files do not match. Ensure 1:1 correspondence.")

    tasks = []
    task_description = ""

    # Prepare tasks based on the expansion type
    if expansion == "spatial":
        task_description = "Processing spatial downsampling"
        if not subsample_spatial:
            for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
                for scale in scale_factors:
                    tasks.append((img_path, label_path, label_values_to_scale, scale, output_img_dir, output_label_dir, metadata_dir))
        else:
            for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
                scale_factor_index = int(i // (len(image_paths) // len(scale_factors)))
                tasks.append((img_path, label_path, label_values_to_scale, scale_factors[scale_factor_index], output_img_dir, output_label_dir, metadata_dir))
        process_func = _process_spatial_task

    elif expansion == "amplitudinal":
        task_description = "Processing amplitudinal downsampling"
        for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
            for qp in qp_values:
                tasks.append((img_path, label_path, qp, output_img_dir, output_label_dir, metadata_dir))
        process_func = _process_amplitudinal_task

    elif expansion == "mixed":
        task_description = "Processing mixed (spatial & amplitudinal) downsampling"

        if subsample_amplitudinal and subsample_spatial:
            scale_factor_index = 0
            qp_index = 0
            for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
                scale_factor_index = scale_factor_index % len(scale_factors)
                qp_index = qp_index % len(qp_values)
                tasks.append((img_path, label_path, label_values_to_scale, scale_factors[scale_factor_index], qp_values[qp_index], output_img_dir, output_label_dir, metadata_dir))
                scale_factor_index += 1
                qp_index += 1
        elif subsample_amplitudinal:
            for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
               qp_index = int(i // (len(image_paths) // len(qp_values)))
               for scale in scale_factors:
                    tasks.append((img_path, label_path, label_values_to_scale, scale, qp_values[qp_index], output_img_dir, output_label_dir, metadata_dir))
        elif subsample_spatial:
            for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
                scale_factor_index = int(i // (len(image_paths) // len(scale_factors)))
                for qp_index in range(len(qp_values)):
                    tasks.append((img_path, label_path, label_values_to_scale, scale_factors[scale_factor_index], qp_values[qp_index], output_img_dir, output_label_dir, metadata_dir))
        else:
            for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
                for scale in scale_factors:
                    for qp in qp_values:
                        tasks.append((img_path, label_path, label_values_to_scale, scale, qp, output_img_dir, output_label_dir, metadata_dir))
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
    num_workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 1 # Ensure at least 1 worker
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
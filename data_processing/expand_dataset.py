import json
import os
from glob import glob
import tempfile
from typing import List
from data_processing.spatial import spatial_downsample
from data_processing.amplitudinal import amplitudinal_downsample
from tqdm import tqdm


def expand_dataset(input_dir: str, label_dir: str, label_values_to_scale:List[str], output_img_dir: str, output_label_dir: str, scale_factors: list, qp_values: list, expansion="spatial"):
    """
    Expands the dataset by applying spatial and amplitudinal downsampling to images.
    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save the processed images.
        scale_factors (list): List of scale factors for spatial downsampling.
        qp_values (list): List of quantization parameters for amplitudinal downsampling.
    Returns:
        None
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    image_paths = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png"))
    label_paths = glob(os.path.join(label_dir, "*.json"))

    for img_path, label_path in tqdm(list(zip(image_paths, label_paths)), desc="Processing images"):

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        if expansion == "spatial":
            # Only spatial downsampling
            for scale in scale_factors:
                spatial_img, label_dict = spatial_downsample(img_path,label_path, label_values_to_scale, scale)
                spatial_out_path = os.path.join(
                    output_img_dir, f"{base_name}_spatial_{scale:.2f}.png"
                )
                spatial_img.save(spatial_out_path)
                with open(os.path.join(output_label_dir, f"{base_name}_spatial_{scale:.2f}.json"), 'w') as f:
                    json.dump(label_dict, f)

        if expansion == "amplitudinal":
            # Only amplitude downsampling
            with open(label_path, 'r') as f:
                label_dict = json.load(f)
            for qp in qp_values:
                amp_img = amplitudinal_downsample(img_path, qp)
                amp_out_path = os.path.join(output_img_dir, f"{base_name}_qp{qp}_out.png")
                amp_img.save(amp_out_path)
                with open(os.path.join(output_label_dir, f"{base_name}_qp{qp}.json"), 'w') as f:
                    json.dump(label_dict, f)
        if expansion == "mixed":
            # Mixed: spatial then amplitude
            for scale in scale_factors:
                spatial_img, label_dict = spatial_downsample(img_path,label_path, label_values_to_scale, scale)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_path = tmp.name
                    spatial_img.save(temp_path)
                try:
                    for qp in qp_values:
                        mixed_img = amplitudinal_downsample(temp_path, qp)
                        mixed_out_path = os.path.join(
                            output_img_dir, f"{base_name}_spatial_{scale:.2f}_qp{qp}.png"
                        )
                        mixed_img.save(mixed_out_path)
                        with open(os.path.join(output_label_dir, f"{base_name}_spatial_{scale:.2f}_qp{qp}.json"), 'w') as f:
                            json.dump(label_dict, f)
                finally:
                    os.remove(temp_path)
    print(f"Images expanded and saved to {output_img_dir} and {output_label_dir}")
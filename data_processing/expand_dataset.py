import os
from glob import glob
import tempfile
from data_processing.spatial import spatial_downsample
from data_processing.amplitudinal import amplitudinal_downsample
from tqdm import tqdm


def expand_dataset(input_dir: str, output_dir: str, scale_factors: list, qp_values: list):
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
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png"))

    for img_path in tqdm(image_paths, desc="Processing images"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Only spatial downsampling
        for scale in scale_factors:
            spatial_img = spatial_downsample(img_path, scale)
            spatial_out_path = os.path.join(
                output_dir, f"{base_name}_spatial_{scale:.2f}.png"
            )
            spatial_img.save(spatial_out_path)

        # Only amplitude downsampling
        for qp in qp_values:
            amp_img = amplitudinal_downsample(img_path, qp)
            amp_out_path = os.path.join(output_dir, f"{base_name}_qp{qp}_out.png")
            amp_img.save(amp_out_path)

        # Mixed: spatial then amplitude
        for scale in scale_factors:
            spatial_img = spatial_downsample(img_path, scale)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = tmp.name
                spatial_img.save(temp_path)
            try:
                for qp in qp_values:
                    mixed_img = amplitudinal_downsample(temp_path, qp)
                    mixed_out_path = os.path.join(
                        output_dir, f"{base_name}_spatial_{scale:.2f}_qp{qp}.png"
                    )
                    mixed_img.save(mixed_out_path)
            finally:
                os.remove(temp_path)
    print(f"Dataset expanded and saved to {output_dir}")
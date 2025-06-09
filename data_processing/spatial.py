from typing import List
from PIL import Image
import json 

def spatial_downsample(input_path: str, label_path:str, label_values_to_scale:List[str], scale_factor: float) -> Image.Image:
    """
    Downsamples the image resolution by a scale factor.

    Args:
        input_path (str): Path to the input image.
        scale_factor (float): Factor to downsample resolution.
                              Must be between 0 and 1 (e.g. 0.5 halves resolution).

    Returns:
        Image.Image: The downsampled image.
    """
    if not (0 < scale_factor <= 1):
        raise ValueError("scale_factor must be between 0 and 1.")

    # Open image
    img = Image.open(input_path)
    with open(label_path, 'r') as f:
        label_dict = json.load(f)
    

    # Compute new size
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    for label_key in label_values_to_scale:
        if label_key in label_dict.keys():
            label_dict[label_key] *= scale_factor
            label_dict[label_key] = int(label_dict[label_key])
        elif "children" in label_dict.keys():
            for child in label_dict["children"]:
                if label_key in child.keys():
                    child[label_key] *= scale_factor
                    child[label_key] = int(child[label_key])  

    # Downsample using high-quality resampling
    downsampled_img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

    return downsampled_img, label_dict

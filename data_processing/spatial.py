from PIL import Image


def spatial_downsample(input_path: str, scale_factor: float) -> Image.Image:
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

    # Compute new size
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)

    # Downsample using high-quality resampling
    downsampled_img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

    return downsampled_img

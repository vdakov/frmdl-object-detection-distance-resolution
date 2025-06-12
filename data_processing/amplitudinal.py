import os
import subprocess
from PIL import Image
import tempfile
import numpy as np


def amplitudinal_downsample(input_path: str, qp: int) -> Image.Image:
    """
    Compresses and decompresses an image using BPG to reduce amplitude resolution.
    This function uses temporary files for intermediate results.

    Args:
        input_path (str): Path to the input image.
        qp (int): Quantization parameter for BPG compression (0-51).

    Returns:
        Image.Image: The decompressed image.
    """
    with tempfile.NamedTemporaryFile(suffix=".bpg", delete=False) as bpg_tmp, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out_tmp:
        bpg_path = bpg_tmp.name
        output_path = out_tmp.name

    try:
        # Perform BPG compression
        subprocess.run(
            ["bpgenc", "-q", str(qp), "-o", bpg_path, input_path], check=True
        )

        # Perform BPG decompression
        subprocess.run(["bpgdec", "-o", output_path, bpg_path], check=True)

        input_img = Image.open(input_path)
        output_img = Image.open(output_path).copy()

        image_size_mb = compute_mb_per_image(bpg_path)
        psnr = compute_PSNR(input_img, output_img)

    finally:
        if os.path.exists(bpg_path):
            os.remove(bpg_path)
        if os.path.exists(output_path):
            os.remove(output_path)

    return output_img, image_size_mb, psnr


def compute_mb_per_image(file_path_compressed):
    return os.path.getsize(file_path_compressed) / (1024**2)


def compute_PSNR(file_original, file_decompressed):
    mse = max(float(np.mean((np.array(file_original) - np.array(file_decompressed)) ** 2)), 1e-8)
    psnr = 10 * np.log10(255**2 / mse)
    return psnr.item()

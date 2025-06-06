import os
import subprocess
from PIL import Image
import tempfile


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

        img = Image.open(output_path).copy()
    finally:
        if os.path.exists(bpg_path):
            os.remove(bpg_path)
        if os.path.exists(output_path):
            os.remove(output_path)

    return img

import os
import subprocess
from PIL import Image

def amplitudinal_downsample(input_path : str, qp : int, output_path : str) -> Image.Image:
    """
    Compresses and decompresses an image using BPG to reduce amplitude resolution.
    Essentially, this function is a python wrapper around the BPG encoder and decoder.

    Args:
        input_path (str): Path to the input image.
        qp (int): Quantization parameter for BPG compression (0-51).
        output_path (str): Path to save the decompressed image.
    
    Returns:
        Image.Image: The decompressed image.
    """
    # Set intermediate and output paths
    bpg_path = input_path.replace('.png', f'_qp{qp}.bpg').replace('.jpg', f'_qp{qp}.bpg')
    output_path = input_path.replace(".png", f"_qp{qp}_out.png").replace(".jpg", f"_qp{qp}_out.png")

    # Perform BPG compression
    subprocess.run(['bpgenc', '-q', str(qp), '-o', bpg_path, input_path], check=True)

    # Perform BPG decompression
    subprocess.run(['bpgdec', '-o', output_path, bpg_path], check=True)

    # Clean up the intermediate BPG file
    os.remove(bpg_path)

    return Image.open(output_path)

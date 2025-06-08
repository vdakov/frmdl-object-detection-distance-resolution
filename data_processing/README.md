# Mixed Dataset Creation Utilities

This folder contains utilities for expanding image datasets by varying spatial and amplitude (quantization) resolution. These tools are useful for generating training or evaluation data at different qualities for tasks such as robust object detection.

## Modules

- **spatial.py**  
  Contains `spatial_downsample`, which downsamples an image by a given scale factor.

- **amplitudinal.py**  
  Contains `amplitudinal_downsample`, which reduces amplitude resolution (quantizes) an image using BPG compression at a specified quantization parameter (QP).

- **expand_dataset.py**  
  Provides `expand_dataset`, a function to automate dataset expansion by generating all combinations of spatial and amplitude downsampling for a directory of images.

## Usage

### 1. Requirements

- Python 3.8+
- [Pillow](https://python-pillow.org/) for image processing
- BPG encoder/decoder (`bpgenc`, `bpgdec`) must be installed and available in your system PATH for amplitude downsampling

Install Python dependencies by entering this command in the project root:
```sh
pip install -r requirements.txt
```

### 2. Example

Place your input images (`.jpg` or `.png`) in a directory, e.g., `./data`.

Run the expansion from your project root:
```sh
python main.py
```

This will:
- Downsample each image spatially (by each scale factor)
- Downsample each image amplitudinally (by each QP value)
- Create all mixed combinations (spatial + amplitude)

Expanded images are saved to `./output/expanded` (see `main.py` for configuration).

### 3. API Reference

#### spatial_downsample

```python
from data_processing.spatial import spatial_downsample

img = spatial_downsample("path/to/image.png", 0.5)
img.save("downsampled.png")
```

#### amplitudinal_downsample

```python
from data_processing.amplitudinal import amplitudinal_downsample

img = amplitudinal_downsample("path/to/image.png", 40)
img.save("quantized.png")
```

#### expand_dataset

```python
from data_processing.expand_dataset import expand_dataset

expand_dataset(
    input_dir="./data",
    output_dir="./output/expanded",
    scale_factors=[0.5, 0.75],
    qp_values=[20, 40],
)
```

## Notes

- Temporary files are used for intermediate results and are cleaned up automatically.
- The amplitude downsampling requires BPG tools (`bpgenc`, `bpgdec`).  
  [BPG GitHub](https://github.com/richgel999/bpg)
- Output files are named to indicate their processing parameters.

---

**Author:** Team 15 - Lászlo Roovers, Lauri Warsen, Vasil Dakov
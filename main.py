from data_processing.amplitudinal import amplitudinal_downsample
from data_processing.spatial import spatial_downsample

amplitudinal_downsample("./data/example.jpg", 51)
spatial_downsample("./data/example.jpg", 0.5)
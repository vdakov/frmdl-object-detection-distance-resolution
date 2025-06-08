from data_processing.expand_dataset import expand_dataset

# Example usage: adjust paths and parameters as needed
expand_dataset(
    input_dir="./data",  # Directory with test images
    output_dir="./output/expanded",  # Where to save the results
    scale_factors=[0.5, 0.75],  # Example scale factors
    qp_values=[20, 40],  # Example QP values
)

def bin_by_distance(data, distance_threshold):
    """
    Bins data based on a specified distance threshold.

    Parameters:
    - data: List of tuples, where each tuple contains (x, y) coordinates.
    - distance_threshold: The maximum distance to consider for binning.

    Returns:
    - List of lists, where each inner list contains points that are within the distance threshold.
    """
    binned_data = []
    
    for point in data:
        x, y = point
        found_bin = False
        
        for bin in binned_data:
            if any(((x - bx) ** 2 + (y - by) ** 2) ** 0.5 <= distance_threshold for bx, by in bin):
                bin.append(point)
                found_bin = True
                break
        
        if not found_bin:
            binned_data.append([point])
    
    return binned_data


def bin_by_amplitudinal_resolution():
    pass 
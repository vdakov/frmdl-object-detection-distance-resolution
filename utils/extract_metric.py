def extract_metric_by_distance(input_images, inputs_labels, binning_distance=10):
    label = []
    
    for img_path, label_path in zip(input_images, inputs_labels):
        json = 
        img_center_x = img.width / 2
        img_center_y = img.height / 2
        
        label_x = label['x0'] + (label['x1'] - label['x0']) / 2
        label_y = label['y0'] + (label['y1'] - label['y0']) / 2
        
        distance = ((img_center_x - label_x) ** 2 + (img_center_y - label_y) ** 2) ** 0.5
        
        # Determine the bin based on the distance
        bin_index = int(distance // binning_distance)
        
        # Append to output list
        out.append((img, label, bin_index))
    
    return out

def extract_metric_by_amplitudinal_resolution(input_images, inputs_labels, binning_distance=10):
    out = []
    
    for img_path, label_path in zip(input_images, inputs_labels):
        json = 
        img_center_x = img.width / 2
        img_center_y = img.height / 2
        
        label_x = label['x0'] + (label['x1'] - label['x0']) / 2
        label_y = label['y0'] + (label['y1'] - label['y0']) / 2
        
        distance = ((img_center_x - label_x) ** 2 + (img_center_y - label_y) ** 2) ** 0.5
        
        # Determine the bin based on the distance
        bin_index = int(distance // binning_distance)
        
        # Append to output list
        out.append((img, label, bin_index))
    
    return out
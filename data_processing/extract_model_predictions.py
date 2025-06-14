
import pandas as pd
from tqdm import tqdm

from utils.extract_metrics import calculate_distance_to_gt, calculate_iou


def extract_predictions_from_run(json_path, label_dir=None):
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    predictions = {}
    labels = {}
    
    for item in data:
        if item['image_id'] not in predictions.keys():
            predictions[item['image_id']] = []
        if item['image_id'] not in labels.keys():
            labels[item['image_id']] = json.load(open(f"{label_dir}/{item['image_id']}.json")) if label_dir else {}
            
        predictions[item['image_id']].append(item)
    
    return predictions, labels


def create_predictions_dataframe(predictions, labels, iou_threshold=0.5, image_dir=None):
    rows = []  # Collect all rows here for batch DataFrame creation
    
    for im_name in tqdm(list(predictions.keys())[:50]):
        original_image_name = im_name.split('_')  # Extract original image name
        original_image_name[5] = '1.00'
        original_image_name[6] = "qp16"
        original_image_name = '_'.join(original_image_name) + ".png"

        if image_dir:
            image_path = os.path.join(image_dir, im_name + ".png")
            try:
                size_bytes = os.path.getsize(image_path)
                size_mb = size_bytes / (1024 * 1024)
            except FileNotFoundError:
                size_mb = None
        else:
            size_mb = None
        # Extract resolutions from filename
        parts = im_name.split('_')
        spatial_res = parts[5]
        amp_val = parts[6][2:].split('.')[0]  # Handle file extensions
        amplitudal_res = int(amp_val) if amp_val != '' else 0
        
        # Get current predictions and labels
        curr_preds = predictions[im_name]
        curr_labels = [obj for obj in labels[im_name]["children"] 
                      if obj['identity'] == "pedestrian"]
        
        gt_bboxes = [
            [obj['x0'], obj['y0'], obj['x1'], obj['y1'], obj['3dp'] if '3dp' in obj else {"x": None, "y": None, "z": None}]
            for obj in curr_labels
        ]
        pred_bboxes = [(pred['bbox'], pred.get('score', 1.0)) for pred in curr_preds]

        
        matched_pred_indices = set()  # Track indices of matched predictions
        
        # Process ground truths
        for gt_bbox in gt_bboxes:
            matched = False
            for i, pred_bbox in enumerate(pred_bboxes):
                bbox, score = pred_bbox
                if i in matched_pred_indices:
                    continue  # Skip already matched predictions
                if calculate_iou(bbox, gt_bbox[:4]) > iou_threshold:
                    matched_pred_indices.add(i)
                    matched = True
                    rows.append({
                        'image_id': im_name,
                        'distance': calculate_distance_to_gt(gt_bbox),
                        'spatial_res': spatial_res,
                        'amplitudal_res': amplitudal_res,
                        'distance_to_gt': 0,
                        'image_size_mb': size_mb,
                        'tp': 1,
                        'fp': 0,
                        'fn': 0,
                        'confidence': score,
                        'label': 'gt',
                    }) 
                    break  # Stop after first match
            
        
        # Process unmatched predictions (false positives)
        for i, _ in enumerate(pred_bboxes):
            if i not in matched_pred_indices:
                bbox, score = pred_bboxes[i]
                rows.append({
                    'image_id': im_name,
                    'distance': 0,
                    'spatial_res': spatial_res,
                    'amplitudal_res': amplitudal_res,
                    'distance_to_gt': 0,
                    'image_size_mb': size_mb,
                    'tp': 0,
                    'fp': 1,
                    'fn': 0,
                    'confidence': score,
                    'label': 'pred',
                })

    
    return pd.DataFrame(rows)
    
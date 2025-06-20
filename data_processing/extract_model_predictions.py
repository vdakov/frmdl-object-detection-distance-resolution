import os
import json
import pandas as pd
from tqdm import tqdm

from metrics.extract_metrics import calculate_distance_to_gt, calculate_iou
import torch
from torchvision.ops import box_iou


def match_single_iou(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.5):
    """
    Matches predictions to GT using a single IoU threshold and returns a list of TP flags per prediction.
    """
    num_preds = len(pred_boxes)
    if len(gt_boxes) == 0 or num_preds == 0:
        return [False] * num_preds

    pred_boxes = torch.tensor(pred_boxes).float()
    gt_boxes = torch.tensor(gt_boxes).float()
    pred_labels = torch.tensor(pred_labels)
    gt_labels = torch.tensor(gt_labels)

    ious = box_iou(gt_boxes, pred_boxes)  # [num_gt, num_pred]
    class_match = gt_labels[:, None] == pred_labels[None, :]  # shape [num_gt, num_pred]

    matches = (ious >= iou_thresh) & class_match

    tp_flags = [False] * num_preds
    match_indices = {}

    if matches.any():
        t_inds, p_inds = torch.where(matches)
        iou_vals = ious[t_inds, p_inds]

        sort = torch.argsort(iou_vals, descending=True)
        t_inds = t_inds[sort]
        p_inds = p_inds[sort]

        used_t = set()
        used_p = set()
        for t, p in zip(t_inds.tolist(), p_inds.tolist()):
            if t not in used_t and p not in used_p:
                tp_flags[p] = True
                match_indices[p] = t
                used_t.add(t)
                used_p.add(p)

    return tp_flags, match_indices


def extract_predictions_from_run(json_path, label_dir=None):
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


def create_predictions_dataframe(predictions, labels, metadata_dir, iou_threshold=0.5):
    rows = []  # Collect all rows here for batch DataFrame creation
    
    for im_name in tqdm(list(predictions.keys())):

        with open(os.path.join(metadata_dir, im_name + ".json"), 'r') as meta_file:
            metadata = json.load(meta_file)

        # Extract resolutions from filename
        parts = im_name.split('_')
        spatial_res = parts[6]
        amp_val = parts[7][2:].split('.')[0]  # Handle file extensions
        amplitudal_res = int(amp_val) if amp_val != '' else 0

        # Collect image size
        size_mb = metadata["image_size_mb"]

        # Collect psnr
        psnr = metadata["psnr"]
        
        # Get current predictions and labels
        curr_preds = predictions[im_name]
        curr_labels = [obj for obj in labels[im_name]["children"] 
                      if obj['identity'] == "pedestrian"]
        
        gt_boxes = [[obj['x0'], obj['y0'], obj['x1'], obj['y1']] for obj in curr_labels]
        gt_labels = [0] * len(gt_boxes)  # assume all pedestrians have class 0
        gt_coordinates =  [obj['3dp'] if '3dp' in obj else {"x": None, "y": None, "z": None} for obj in curr_labels]

        
        pred_boxes = [pred['bbox'] for pred in curr_preds]
        pred_scores = [pred.get('score', 1.0) for pred in curr_preds]
        pred_labels = [0] * len(pred_boxes)
        
        
        tp_flags, match_indices = match_single_iou(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=iou_threshold)
        for i, (bbox, score, is_tp) in enumerate(zip(pred_boxes, pred_scores, tp_flags)):
            if is_tp:
                gt_coord = gt_coordinates[match_indices[i]]
                distance = calculate_distance_to_gt(gt_coord)
            else:

                distance = 0
            rows.append({
                'image_id': im_name,
                'spatial_res': spatial_res,
                'amplitudal_res': amplitudal_res,
                'image_size_mb': size_mb,
                'psnr': psnr,
                'confidence': score,
                'tp': int(is_tp),
                'fp': int(not is_tp),
                'label': 'pred',
                'distance': 'distance'
            })
        
        # matched_pred_indices = set()  # Track indices of matched predictions

        # # Process ground truths
        # for gt_bbox in gt_bboxes:
        #     matched = False
        #     for i, pred_bbox in enumerate(pred_bboxes):
        #         bbox, score = pred_bbox
        #         if i in matched_pred_indices:
        #             continue  # Skip already matched predictions
        #         iou = calculate_iou(bbox, gt_bbox[:4])
        #         if iou > iou_threshold:
        #             print("check 3")
        #             matched_pred_indices.add(i)
        #             matched = True
        #             rows.append({
        #                 'image_id': im_name,
        #                 'spatial_res': spatial_res,
        #                 'amplitudal_res': amplitudal_res,
        #                 'image_size_mb': size_mb,
        #                 'psnr': psnr,
        #                 # Takes norm of LIDAR position vector relative to camera
        #                 'distance': calculate_distance_to_gt(gt_bbox),
        #                 'tp': 1,
        #                 'fp': 0,
        #                 'confidence': score,
        #                 'iou': calculate_iou(bbox, gt_bbox[:4]),
        #                 'label': 'gt',
        #             }) 
        #             break  # Stop after first match
            
        
        # # Process unmatched predictions (false positives)
        # for i, _ in enumerate(pred_bboxes):
        #     if i not in matched_pred_indices:
        #         bbox, score = pred_bboxes[i]
        #         rows.append({
        #             'image_id': im_name,
        #             'spatial_res': spatial_res,
        #             'amplitudal_res': amplitudal_res,
        #             'image_size_mb': size_mb,
        #             'distance': 0,
        #             'tp': 0,
        #             'fp': 1,
        #             'confidence': score,
        #             'iou': 0,
        #             'label': 'pred',
        #         })

    
    return pd.DataFrame(rows)
    
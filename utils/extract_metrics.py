from matplotlib import pyplot as plt
from sklearn.metrics import auc


def calculate_iou(bbox1, bbox2):
    # Existing IoU calculation remains unchanged
    x0_pred, y0_pred, x1_pred, y1_pred = bbox1
    x0_gt, y0_gt, x1_gt, y1_gt = bbox2
    
    
    x0_inter = max(x0_pred, x0_gt)
    y0_inter = max(y0_pred, y0_gt)
    x1_inter = min(x1_pred, x1_gt)
    y1_inter = min(y1_pred, y1_gt)
    
    if x0_inter < x1_inter and y0_inter < y1_inter:
        intersection_area = (x1_inter - x0_inter) * (y1_inter - y0_inter)
    else:
        intersection_area = 0
    
    area_pred = (x1_pred - x0_pred) * (y1_pred - y0_pred)
    area_gt = (x1_gt - x0_gt) * (y1_gt - y0_gt)
    union_area = area_pred + area_gt - intersection_area
    
    return intersection_area / union_area if union_area != 0 else 0.0

def calculate_distance_to_gt(label):
    xyz = label[4]
    x, y, z = xyz["x"], xyz["y"], xyz["z"]

    return (x**2 + y**2 + z**2)**0.5 if x is not None and y is not None and z is not None else 0.0


def compute_map50(df, plot=True):
    df_preds = df.copy()
    # Filter only rows that are either TP or FP
    detection_df = df_preds[df_preds['tp'] + df_preds['fp'] > 0]
    
    # Sort by confidence descending
    detection_df = detection_df.sort_values(by='confidence', ascending=False).reset_index(drop=True)
    
    # Cumulative TP/FP
    detection_df['cum_tp'] = detection_df['tp'].cumsum()
    detection_df['cum_fp'] = detection_df['fp'].cumsum()
    
    # Total ground truths   
    total_gt = df[df['label'] == 'gt'].shape[0]
    
    
    # Precision and Recall
    detection_df['precision'] = detection_df['cum_tp'] / (detection_df['cum_tp'] + detection_df['cum_fp'])\
        if df_preds.shape[0] > 0 else 0.0
    detection_df['recall'] = detection_df['cum_tp'] / total_gt\
        if total_gt > 0 else 0.0
    
    # Interpolate and calculate AP as area under the curve
    precision = detection_df['precision'].values
    recall = detection_df['recall'].values
    
    ap = auc(recall, precision)  # Trapezoidal integration
    
    if plot:
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, marker='.', label=f'mAP@50 = {ap:.4f}')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (IoU = 0.50)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return ap
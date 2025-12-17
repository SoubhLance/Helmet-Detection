"""
YOLO Model Accuracy Calculator
==============================
Uses the EXACT same mathematical formulas from Ultralytics YOLO source code.
Source: ultralytics/utils/metrics.py

This script tests your trained model on images WITH labels to calculate TRUE accuracy.
"""

# ============================================================================
# CBAM Module - Required for loading YOLOv8-CBAM models
# Must be defined BEFORE importing YOLO and in __main__ namespace
# MUST MATCH EXACTLY how it was defined during training!
# ============================================================================
import torch
import torch.nn as nn
import sys

class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Ensure reduced channels is at least 1 (MUST MATCH TRAINING!)
        reduced_channels = max(channels // reduction, 1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(
            self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module with lazy initialization.
    
    This version uses lazy initialization to determine channels from actual input,
    avoiding issues with width_multiple scaling.
    """
    def __init__(self, c1, *args):  # c1 = channel arg from YAML (may not match actual)
        super().__init__()
        self._channels = None
        self.ca = None
        self.sa = SpatialAttention()

    def _lazy_init(self, channels):
        """Initialize channel attention when we know actual input channels."""
        self.ca = ChannelAttention(channels)
        # Move to same device as spatial attention
        device = next(self.sa.parameters()).device
        self.ca = self.ca.to(device)
        self._channels = channels

    def forward(self, x):
        # Lazy init on first forward pass
        if self.ca is None or self._channels != x.shape[1]:
            self._lazy_init(x.shape[1])
        x = self.ca(x)
        return x * self.sa(x)

# Register CBAM classes in __main__ so PyTorch can find them when loading the model
sys.modules['__main__'].CBAM = CBAM
sys.modules['__main__'].ChannelAttention = ChannelAttention
sys.modules['__main__'].SpatialAttention = SpatialAttention
# ============================================================================

from ultralytics import YOLO
import numpy as np
import os
import glob
import yaml
from pathlib import Path
from collections import defaultdict
import cv2

# ============================================================================
# CONFIGURATION - Edit these paths
# ============================================================================
MODEL_PATH = "C:/Users/babud/Desktop/yolo/Helmet-Detection/Helmet-Detection/runs/detect/helmet_yolov8n_cbam_scratch/weights/best.pt"   # COCO pretrained model (better generalization)
DATA_YAML = "data_test.yaml"                   # Your data.yaml file
CONF_THRESHOLD = 0.5                      # Increased to reduce false positives
IOU_THRESHOLD = 0.5                       # IoU threshold for matching
SAVE_RESULTS = True                       # Save annotated images
OUTPUT_FOLDER = "results"        # Output folder
# ============================================================================


# ============================================================================
# MATHEMATICAL FORMULAS (Exact copy from ultralytics/utils/metrics.py)
# ============================================================================

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate IoU (Intersection over Union) between two boxes.
    
    FORMULA FROM ULTRALYTICS (metrics.py line 56-76):
        IoU = Area_of_Intersection / Area_of_Union
        
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value between 0 and 1
    """
    # Intersection area
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # IoU = Intersection / Union
    return inter_area / (union_area + eps)


def compute_ap(recall, precision):
    """
    Compute Average Precision (AP) - Area under Precision-Recall curve.
    
    EXACT FORMULA FROM ULTRALYTICS (metrics.py line 711-737):
        1. Append sentinel values
        2. Make precision monotonically decreasing
        3. 101-point interpolation (COCO style)
        4. Calculate area using trapezoidal rule
    
    Args:
        recall: Array of recall values
        precision: Array of precision values
    
    Returns:
        AP value between 0 and 1
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Compute the precision envelope (make monotonically decreasing)
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    
    # 101-point interpolation (COCO style)
    x = np.linspace(0, 1, 101)
    
    # Integrate area under curve using trapezoidal rule
    ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    
    return ap


def calculate_metrics(tp_list, conf_list, pred_cls_list, target_cls_list, num_classes, eps=1e-16):
    """
    Calculate all metrics per class.
    
    EXACT ALGORITHM FROM ULTRALYTICS (metrics.py ap_per_class function, line 743-823):
    
    Args:
        tp_list: List of True Positive arrays (binary, shape [N, 10] for 10 IoU thresholds)
        conf_list: List of confidence scores
        pred_cls_list: List of predicted classes
        target_cls_list: List of target classes
        num_classes: Number of classes
    
    Returns:
        Dictionary with precision, recall, f1, ap50, map, etc.
    """
    # Concatenate all results
    tp = np.concatenate(tp_list, axis=0) if tp_list else np.array([])
    conf = np.concatenate(conf_list) if conf_list else np.array([])
    pred_cls = np.concatenate(pred_cls_list) if pred_cls_list else np.array([])
    target_cls = np.concatenate(target_cls_list) if target_cls_list else np.array([])
    
    if len(tp) == 0:
        return {
            'precision': 0, 'recall': 0, 'f1': 0,
            'ap50': 0, 'ap': 0, 'map50': 0, 'map': 0
        }
    
    # Sort by confidence (descending)
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = len(unique_classes)
    
    # Initialize arrays
    ap = np.zeros((nc, tp.shape[1] if len(tp.shape) > 1 else 1))
    p_values = np.zeros(nc)
    r_values = np.zeros(nc)
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels for this class
        n_p = i.sum()  # number of predictions for this class
        
        if n_p == 0 or n_l == 0:
            continue
        
        # Accumulate FPs and TPs
        tpc = tp[i].cumsum(axis=0)
        fpc = (1 - tp[i]).cumsum(axis=0)
        
        # Recall = TP / (TP + FN) = TP / total_ground_truth
        recall = tpc / (n_l + eps)
        
        # Precision = TP / (TP + FP)
        precision = tpc / (tpc + fpc + eps)
        
        # Store final precision and recall
        if len(precision) > 0:
            p_values[ci] = precision[-1, 0] if len(precision.shape) > 1 else precision[-1]
            r_values[ci] = recall[-1, 0] if len(recall.shape) > 1 else recall[-1]
        
        # AP for each IoU threshold
        for j in range(tp.shape[1] if len(tp.shape) > 1 else 1):
            rec = recall[:, j] if len(recall.shape) > 1 else recall
            prec = precision[:, j] if len(precision.shape) > 1 else precision
            ap[ci, j] = compute_ap(rec, prec)
    
    # Calculate mean values
    mp = p_values.mean() if len(p_values) > 0 else 0  # mean precision
    mr = r_values.mean() if len(r_values) > 0 else 0  # mean recall
    map50 = ap[:, 0].mean() if len(ap) > 0 else 0     # mAP@0.5
    map_val = ap.mean() if len(ap) > 0 else 0         # mAP@0.5:0.95
    
    # F1 score
    f1 = 2 * mp * mr / (mp + mr + eps)
    
    return {
        'precision': mp,
        'recall': mr,
        'f1': f1,
        'ap50': ap[:, 0] if len(ap) > 0 else np.array([0]),
        'ap': ap.mean(axis=1) if len(ap) > 0 else np.array([0]),
        'map50': map50,
        'map': map_val,
        'per_class_p': p_values,
        'per_class_r': r_values,
        'unique_classes': unique_classes
    }


def fitness(metrics):
    """
    Calculate fitness score - EXACT FORMULA from Ultralytics.
    
    FROM ULTRALYTICS (metrics.py line 955-957):
        w = [0.0, 0.0, 0.0, 1.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        fitness = (metrics * w).sum()
    
    This means: FITNESS = mAP@0.5:0.95 (100% weight)
    """
    w = np.array([0.0, 0.0, 0.0, 1.0])  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    values = np.array([metrics['precision'], metrics['recall'], metrics['map50'], metrics['map']])
    return (values * w).sum()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_yaml(yaml_path):
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def load_labels(label_path, img_width, img_height):
    """
    Load ground truth labels from YOLO format .txt file.
    Convert from normalized [x_center, y_center, width, height] to [x1, y1, x2, y2]
    """
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    # Convert to x1, y1, x2, y2
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    labels.append({
                        'class': cls_id,
                        'bbox': [x1, y1, x2, y2]
                    })
    return labels


def match_predictions(predictions, ground_truths, iou_thresholds):
    """
    Match predictions to ground truth using IoU at multiple thresholds.
    Returns TP array of shape [num_predictions, num_iou_thresholds]
    
    This follows the EXACT matching algorithm from Ultralytics.
    """
    num_preds = len(predictions)
    num_thresholds = len(iou_thresholds)
    
    if num_preds == 0:
        return np.zeros((0, num_thresholds), dtype=bool)
    
    tp = np.zeros((num_preds, num_thresholds), dtype=bool)
    
    if len(ground_truths) == 0:
        return tp
    
    # Sort predictions by confidence (descending)
    sorted_indices = sorted(range(num_preds), key=lambda i: predictions[i]['confidence'], reverse=True)
    
    matched_gt = set()
    
    for pred_idx in sorted_indices:
        pred = predictions[pred_idx]
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            if pred['class'] != gt['class']:
                continue
            
            iou = box_iou(pred['bbox'], gt['bbox'])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check each IoU threshold
        for t_idx, threshold in enumerate(iou_thresholds):
            if best_iou >= threshold and best_gt_idx >= 0 and best_gt_idx not in matched_gt:
                tp[pred_idx, t_idx] = True
        
        if best_iou >= iou_thresholds[0] and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
    
    return tp


# ============================================================================
# MAIN ACCURACY TEST FUNCTION
# ============================================================================

def test_accuracy():
    """
    Main function to test model accuracy using Ultralytics formulas.
    """
    print("=" * 70)
    print("YOLO MODEL ACCURACY CALCULATOR")
    print("Using Ultralytics Mathematical Formulas")
    print("=" * 70)
    
    # Load model
    print(f"\n📦 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    num_classes = len(class_names)
    print(f"   Classes: {class_names}")
    
    # Load data.yaml
    print(f"\n📄 Loading data config: {DATA_YAML}")
    data_config = load_yaml(DATA_YAML)
    
    # Get image and label paths
    data_path = data_config.get('path', '')
    val_path = data_config.get('val', data_config.get('test', ''))
    
    if data_path:
        images_folder = os.path.join(data_path, val_path)
    else:
        images_folder = val_path
    
    # Handle both images/ and labels/ folder structure
    if 'images' in images_folder:
        labels_folder = images_folder.replace('images', 'labels')
    else:
        labels_folder = os.path.join(os.path.dirname(images_folder), 'labels', os.path.basename(images_folder))
    
    print(f"   Images folder: {images_folder}")
    print(f"   Labels folder: {labels_folder}")
    
    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_folder, ext)))
    
    # Remove duplicates (Windows glob is case-insensitive)
    image_paths = list(set(image_paths))
    
    if not image_paths:
        print(f"\n❌ No images found in: {images_folder}")
        return
    
    print(f"\n📊 Found {len(image_paths)} images")
    
    # Create output folder
    if SAVE_RESULTS:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # IoU thresholds (0.5 to 0.95 in steps of 0.05 - COCO standard)
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    
    # Storage for all results
    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_target_cls = []
    
    total_gt = 0
    total_pred = 0
    images_processed = 0
    
    print("\n🔍 Processing images...")
    print("-" * 70)
    
    for img_path in image_paths:
        img_name = Path(img_path).stem
        label_path = os.path.join(labels_folder, f"{img_name}.txt")
        
        # Read image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        
        # Load ground truth labels
        ground_truths = load_labels(label_path, img_width, img_height)
        
        if not ground_truths and not os.path.exists(label_path):
            continue  # Skip images without labels
        
        # Run inference
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]
        
        # Extract predictions
        predictions = []
        if result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                xyxy = result.boxes.xyxy[i].cpu().numpy()
                predictions.append({
                    'class': int(result.boxes.cls[i].item()),
                    'confidence': float(result.boxes.conf[i].item()),
                    'bbox': xyxy.tolist()
                })
        
        # Match predictions to ground truth
        tp = match_predictions(predictions, ground_truths, iou_thresholds)
        
        # Store results
        if len(predictions) > 0:
            all_tp.append(tp)
            all_conf.append(np.array([p['confidence'] for p in predictions]))
            all_pred_cls.append(np.array([p['class'] for p in predictions]))
        
        all_target_cls.append(np.array([gt['class'] for gt in ground_truths]))
        
        total_gt += len(ground_truths)
        total_pred += len(predictions)
        images_processed += 1
        
        # Print progress
        if images_processed % 10 == 0 or images_processed == len(image_paths):
            print(f"   Processed {images_processed}/{len(image_paths)} images")
        
        # Save annotated image
        if SAVE_RESULTS:
            annotated = result.plot()
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"result_{img_name}.jpg"), annotated)
    
    # ========================================================================
    # CALCULATE FINAL METRICS
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("CALCULATING METRICS (Using Ultralytics Formulas)")
    print("=" * 70)
    
    # Calculate metrics
    metrics = calculate_metrics(
        all_tp, all_conf, all_pred_cls, all_target_cls, num_classes
    )
    
    # Calculate fitness
    fitness_score = fitness(metrics)
    
    # Display results
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    ACCURACY RESULTS                               ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Images Tested:      {images_processed:>6}                                      ║
    ║  Total Ground Truth: {total_gt:>6}                                      ║
    ║  Total Predictions:  {total_pred:>6}                                      ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                       METRICS                                     ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Precision (P):      {metrics['precision']:.4f}  ({metrics['precision']*100:.1f}%)                          ║
    ║  Recall (R):         {metrics['recall']:.4f}  ({metrics['recall']*100:.1f}%)                          ║
    ║  F1-Score:           {metrics['f1']:.4f}  ({metrics['f1']*100:.1f}%)                          ║
    ║  mAP@0.5:            {metrics['map50']:.4f}  ({metrics['map50']*100:.1f}%)                          ║
    ║  mAP@0.5:0.95:       {metrics['map']:.4f}  ({metrics['map']*100:.1f}%)                          ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  ⭐ FITNESS SCORE:   {fitness_score:.4f}  ({fitness_score*100:.1f}%)                          ║
    ║  (This is THE final accuracy metric used by YOLO)                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Per-class results
    if len(metrics['unique_classes']) > 1:
        print("\n📊 Per-Class Results:")
        print("-" * 50)
        for i, cls_id in enumerate(metrics['unique_classes']):
            cls_name = class_names.get(int(cls_id), f"Class {cls_id}")
            print(f"   {cls_name}: P={metrics['per_class_p'][i]:.3f}, R={metrics['per_class_r'][i]:.3f}, AP@0.5={metrics['ap50'][i]:.3f}")
    
    # Print formulas used
    print("\n" + "=" * 70)
    print("📐 MATHEMATICAL FORMULAS USED (from ultralytics/utils/metrics.py)")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │ IoU = Intersection / Union                        (line 56-76) │
    │                                                                 │
    │ Precision = TP / (TP + FP)                       (line 805)    │
    │                                                                 │
    │ Recall = TP / (TP + FN)                          (line 802)    │
    │                                                                 │
    │ F1 = 2 × (P × R) / (P + R)                       (line 817)    │
    │                                                                 │
    │ AP = ∫ Precision(Recall) dRecall                 (line 711-737)│
    │     (101-point interpolation, trapezoidal rule)                │
    │                                                                 │
    │ mAP = (1/C) × Σ AP_c                             (line 936)    │
    │                                                                 │
    │ Fitness = 0×P + 0×R + 0×mAP@0.5 + 1×mAP@0.5:0.95 (line 955-957)│
    │         = mAP@0.5:0.95                                         │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    # Save results to file
    results_file = os.path.join(OUTPUT_FOLDER, "accuracy_report.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("YOLO Model Accuracy Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Data: {DATA_YAML}\n")
        f.write(f"Images Tested: {images_processed}\n")
        f.write(f"Total Ground Truth: {total_gt}\n")
        f.write(f"Total Predictions: {total_pred}\n\n")
        f.write("METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Precision:     {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)\n")
        f.write(f"Recall:        {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)\n")
        f.write(f"F1-Score:      {metrics['f1']:.4f} ({metrics['f1']*100:.1f}%)\n")
        f.write(f"mAP@0.5:       {metrics['map50']:.4f} ({metrics['map50']*100:.1f}%)\n")
        f.write(f"mAP@0.5:0.95:  {metrics['map']:.4f} ({metrics['map']*100:.1f}%)\n")
        f.write(f"\nFITNESS:       {fitness_score:.4f} ({fitness_score*100:.1f}%)\n")
    
    print(f"\n📁 Results saved to: {OUTPUT_FOLDER}/")
    print("=" * 70)
    
    return metrics, fitness_score


if __name__ == "__main__":
    test_accuracy()

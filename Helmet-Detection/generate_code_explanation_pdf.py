"""
Generate PDF explaining each line of the YOLO Accuracy Calculator code.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

def create_pdf():
    doc = SimpleDocTemplate(
        "YOLO_Accuracy_Code_Explanation.pdf",
        pagesize=A4,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontSize=13,
        spaceAfter=8,
        spaceBefore=15,
        textColor=colors.darkgreen
    )
    
    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Heading3'],
        fontSize=11,
        spaceAfter=6,
        spaceBefore=10,
        textColor=colors.brown
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=8,
        fontName='Courier',
        backColor=colors.Color(0.95, 0.95, 0.95),
        leftIndent=10,
        spaceAfter=5,
        spaceBefore=5
    )
    
    explain_style = ParagraphStyle(
        'Explain',
        parent=styles['Normal'],
        fontSize=9,
        leftIndent=20,
        textColor=colors.darkslategray,
        spaceAfter=10
    )
    
    story = []
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("YOLO Accuracy Calculator", title_style))
    story.append(Paragraph("Complete Line-by-Line Code Explanation", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Using Ultralytics Mathematical Formulas", body_style))
    story.append(Spacer(1, 1*inch))
    
    info_data = [
        ["File:", "test_accuracy_with_labels.py"],
        ["Purpose:", "Calculate model accuracy using exact YOLO formulas"],
        ["Source:", "ultralytics/utils/metrics.py"],
        ["Author:", "Generated for IEEE Project"],
    ]
    info_table = Table(info_data, colWidths=[1.5*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", h1_style))
    toc = [
        "1. Import Statements",
        "2. Configuration Section",
        "3. IoU (Intersection over Union) Function",
        "4. Average Precision (AP) Calculation",
        "5. Metrics Calculation Function",
        "6. Fitness Score Function",
        "7. Helper Functions",
        "8. Main Accuracy Test Function",
        "9. YOLO Built-in Validation",
        "10. Main Entry Point"
    ]
    for item in toc:
        story.append(Paragraph(f"• {item}", body_style))
    story.append(PageBreak())
    
    # ==========================================================================
    # SECTION 1: IMPORTS
    # ==========================================================================
    story.append(Paragraph("1. Import Statements", h1_style))
    story.append(Paragraph("These are the libraries required to run the accuracy calculator.", body_style))
    
    imports = [
        ('from ultralytics import YOLO', 
         'Imports the YOLO class from Ultralytics library. This is the main class used to load and run YOLO models for object detection.'),
        ('import numpy as np', 
         'NumPy is used for numerical computations - arrays, mathematical operations like mean, sum, cumsum, etc. Essential for metric calculations.'),
        ('import os', 
         'Provides functions for interacting with the operating system - checking if files exist, joining paths, creating directories.'),
        ('import glob', 
         'Used to find all files matching a pattern (e.g., all .jpg files in a folder).'),
        ('import yaml', 
         'YAML parser to read the data.yaml configuration file that contains dataset paths and class names.'),
        ('from pathlib import Path', 
         'Modern way to handle file paths. Used to extract filename without extension.'),
        ('from collections import defaultdict', 
         'Dictionary that provides default values. Useful for counting and grouping.'),
        ('import cv2', 
         'OpenCV library for reading images, getting dimensions, and saving annotated results.')
    ]
    
    for code, explanation in imports:
        story.append(Paragraph(f"<font face='Courier' size='9' color='blue'>{code}</font>", body_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    story.append(PageBreak())
    
    # ==========================================================================
    # SECTION 2: CONFIGURATION
    # ==========================================================================
    story.append(Paragraph("2. Configuration Section", h1_style))
    story.append(Paragraph("User-editable settings at the top of the file.", body_style))
    
    configs = [
        ('MODEL_PATH = "best.pt"',
         'Path to your trained YOLO model file. "best.pt" is the model with best fitness during training.'),
        ('DATA_YAML = "data.yaml"',
         'Path to your dataset configuration file. Contains paths to images/labels and class names.'),
        ('CONF_THRESHOLD = 0.001',
         'Confidence threshold for detections. Set very low (0.001) to get ALL possible detections during validation. This matches YOLO\'s default validation behavior.'),
        ('IOU_THRESHOLD = 0.5',
         'IoU threshold for matching predictions to ground truth. A prediction is "correct" if IoU ≥ 0.5 with a ground truth box.'),
        ('SAVE_RESULTS = True',
         'Whether to save annotated images showing detections. Useful for visual verification.'),
        ('OUTPUT_FOLDER = "accuracy_results"',
         'Folder where results (annotated images, report) will be saved.')
    ]
    
    for code, explanation in configs:
        story.append(Paragraph(f"<font face='Courier' size='9' color='darkred'>{code}</font>", body_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    story.append(PageBreak())
    
    # ==========================================================================
    # SECTION 3: IoU FUNCTION
    # ==========================================================================
    story.append(Paragraph("3. IoU (Intersection over Union) Function", h1_style))
    story.append(Paragraph("This function calculates how much two bounding boxes overlap. It's the fundamental metric for object detection.", body_style))
    
    story.append(Paragraph("Mathematical Formula:", h2_style))
    story.append(Paragraph("<b>IoU = Area of Intersection / Area of Union</b>", body_style))
    story.append(Paragraph("Source: ultralytics/utils/metrics.py, lines 56-76", body_style))
    
    iou_lines = [
        ('def box_iou(box1, box2, eps=1e-7):',
         'Function definition. Takes two boxes and epsilon (small number to prevent division by zero).'),
        ('# Intersection area',
         'Comment indicating we\'re calculating where the boxes overlap.'),
        ('inter_x1 = max(box1[0], box2[0])',
         'Left edge of intersection = rightmost of the two left edges.'),
        ('inter_y1 = max(box1[1], box2[1])',
         'Top edge of intersection = bottommost of the two top edges.'),
        ('inter_x2 = min(box1[2], box2[2])',
         'Right edge of intersection = leftmost of the two right edges.'),
        ('inter_y2 = min(box1[3], box2[3])',
         'Bottom edge of intersection = topmost of the two bottom edges.'),
        ('inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)',
         'Intersection area = width × height. max(0, ...) ensures no negative values if boxes don\'t overlap.'),
        ('# Union area',
         'Comment indicating we\'re calculating total area covered by both boxes.'),
        ('box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])',
         'Area of first box = width × height.'),
        ('box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])',
         'Area of second box = width × height.'),
        ('union_area = box1_area + box2_area - inter_area',
         'Union = sum of areas minus intersection (to avoid counting overlap twice).'),
        ('return inter_area / (union_area + eps)',
         'Final IoU = intersection / union. Add eps to prevent division by zero.')
    ]
    
    for code, explanation in iou_lines:
        story.append(Paragraph(f"<font face='Courier' size='8'>{code}</font>", code_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    # Visual diagram
    story.append(Paragraph("Visual Representation:", h3_style))
    diagram = """
    ┌─────────────────┐
    │     Box 1       │
    │   ┌─────────┼───┼──┐
    │   │ INTER-  │   │  │
    │   │ SECTION │   │  │
    └───┼─────────┘   │  │
        │     Box 2   │  │
        └─────────────┘
    
    IoU = Intersection Area / (Box1 + Box2 - Intersection)
    """
    story.append(Preformatted(diagram, code_style))
    
    story.append(PageBreak())
    
    # ==========================================================================
    # SECTION 4: COMPUTE AP
    # ==========================================================================
    story.append(Paragraph("4. Average Precision (AP) Calculation", h1_style))
    story.append(Paragraph("Calculates the Area Under the Precision-Recall Curve using 101-point interpolation (COCO method).", body_style))
    story.append(Paragraph("Source: ultralytics/utils/metrics.py, lines 711-737", body_style))
    
    ap_lines = [
        ('def compute_ap(recall, precision):',
         'Function takes arrays of recall and precision values at different confidence thresholds.'),
        ('mrec = np.concatenate(([0.0], recall, [1.0]))',
         'Add sentinel values: recall starts at 0 and ends at 1. This ensures the curve covers full range.'),
        ('mpre = np.concatenate(([1.0], precision, [0.0]))',
         'Add sentinel values: precision starts at 1 (perfect at no detections) and ends at 0.'),
        ('mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))',
         'Make precision monotonically decreasing from right to left. This is the "envelope" of the PR curve.'),
        ('x = np.linspace(0, 1, 101)',
         '101 evenly spaced points from 0 to 1. This is the COCO-style interpolation (at recall points 0, 0.01, 0.02, ..., 1.0).'),
        ('ap = np.trapezoid(np.interp(x, mrec, mpre), x)',
         'Interpolate precision at 101 recall points, then calculate area using trapezoidal rule. This IS the AP.')
    ]
    
    for code, explanation in ap_lines:
        story.append(Paragraph(f"<font face='Courier' size='8'>{code}</font>", code_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    story.append(Paragraph("Why 101-Point Interpolation?", h3_style))
    story.append(Paragraph("COCO benchmark uses 101 points for smoother, more accurate AP calculation. Points are at recall = 0.00, 0.01, 0.02, ..., 0.99, 1.00.", body_style))
    
    story.append(PageBreak())
    
    # ==========================================================================
    # SECTION 5: CALCULATE METRICS
    # ==========================================================================
    story.append(Paragraph("5. Metrics Calculation Function", h1_style))
    story.append(Paragraph("This is the core function that calculates Precision, Recall, F1, and mAP.", body_style))
    story.append(Paragraph("Source: ultralytics/utils/metrics.py, ap_per_class function, lines 743-823", body_style))
    
    story.append(Paragraph("5.1 Data Concatenation", h2_style))
    concat_lines = [
        ('tp = np.concatenate(tp_list, axis=0)',
         'Combine True Positive arrays from all images into one array.'),
        ('conf = np.concatenate(conf_list)',
         'Combine confidence scores from all predictions.'),
        ('pred_cls = np.concatenate(pred_cls_list)',
         'Combine predicted class IDs from all predictions.'),
        ('target_cls = np.concatenate(target_cls_list)',
         'Combine ground truth class IDs from all labels.')
    ]
    for code, explanation in concat_lines:
        story.append(Paragraph(f"<font face='Courier' size='8'>{code}</font>", code_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    story.append(Paragraph("5.2 Sorting by Confidence", h2_style))
    story.append(Paragraph("<font face='Courier' size='8'>i = np.argsort(-conf)</font>", code_style))
    story.append(Paragraph("↳ Sort all predictions by confidence (highest first). The negative sign makes it descending order.", explain_style))
    story.append(Paragraph("<font face='Courier' size='8'>tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]</font>", code_style))
    story.append(Paragraph("↳ Reorder all arrays according to sorted confidence. High-confidence predictions are processed first.", explain_style))
    
    story.append(Paragraph("5.3 Per-Class Calculation Loop", h2_style))
    loop_lines = [
        ('for ci, c in enumerate(unique_classes):',
         'Loop through each unique class found in ground truth.'),
        ('i = pred_cls == c',
         'Boolean mask: True where prediction class matches current class.'),
        ('n_l = nt[ci]  # number of labels',
         'Count of ground truth objects for this class.'),
        ('n_p = i.sum()  # number of predictions',
         'Count of predictions for this class.'),
        ('tpc = tp[i].cumsum(axis=0)',
         'Cumulative sum of True Positives. At each position, how many TPs so far.'),
        ('fpc = (1 - tp[i]).cumsum(axis=0)',
         'Cumulative sum of False Positives. (1 - TP) = FP at each position.')
    ]
    for code, explanation in loop_lines:
        story.append(Paragraph(f"<font face='Courier' size='8'>{code}</font>", code_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    story.append(Paragraph("5.4 Precision and Recall Formulas", h2_style))
    pr_lines = [
        ('recall = tpc / (n_l + eps)',
         '<b>Recall = TP / Total Ground Truth</b>. How many actual objects did we find?'),
        ('precision = tpc / (tpc + fpc + eps)',
         '<b>Precision = TP / (TP + FP)</b>. Of all our predictions, how many were correct?')
    ]
    for code, explanation in pr_lines:
        story.append(Paragraph(f"<font face='Courier' size='8'>{code}</font>", code_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph("5.5 AP Calculation for Each IoU Threshold", h2_style))
    story.append(Paragraph("YOLO calculates AP at 10 different IoU thresholds: 0.50, 0.55, 0.60, ..., 0.95", body_style))
    story.append(Paragraph("<font face='Courier' size='8'>for j in range(tp.shape[1]): ap[ci, j] = compute_ap(rec, prec)</font>", code_style))
    story.append(Paragraph("↳ For each IoU threshold, calculate AP using the 101-point interpolation method.", explain_style))
    
    story.append(Paragraph("5.6 Final Metrics", h2_style))
    final_lines = [
        ('mp = p_values.mean()',
         'Mean Precision across all classes.'),
        ('mr = r_values.mean()',
         'Mean Recall across all classes.'),
        ('map50 = ap[:, 0].mean()',
         'mAP at IoU=0.5 (first threshold). Average AP across all classes at IoU 0.5.'),
        ('map_val = ap.mean()',
         'mAP at IoU=0.5:0.95. Average of all APs across all classes and all 10 IoU thresholds.'),
        ('f1 = 2 * mp * mr / (mp + mr + eps)',
         '<b>F1 Score = 2 × P × R / (P + R)</b>. Harmonic mean of precision and recall.')
    ]
    for code, explanation in final_lines:
        story.append(Paragraph(f"<font face='Courier' size='8'>{code}</font>", code_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    story.append(PageBreak())
    
    # ==========================================================================
    # SECTION 6: FITNESS
    # ==========================================================================
    story.append(Paragraph("6. Fitness Score Function", h1_style))
    story.append(Paragraph("The FITNESS score is THE final accuracy metric used by YOLO to select the best model.", body_style))
    story.append(Paragraph("Source: ultralytics/utils/metrics.py, lines 955-957", body_style))
    
    fitness_lines = [
        ('def fitness(metrics):',
         'Function to calculate the overall model fitness/accuracy.'),
        ('w = np.array([0.0, 0.0, 0.0, 1.0])',
         'Weights for [Precision, Recall, mAP@0.5, mAP@0.5:0.95]. Only mAP@0.5:0.95 has weight 1.0!'),
        ('values = np.array([metrics["precision"], metrics["recall"], metrics["map50"], metrics["map"]])',
         'Array of the four main metrics.'),
        ('return (values * w).sum()',
         'Weighted sum. With these weights: Fitness = 0×P + 0×R + 0×mAP@0.5 + 1×mAP@0.5:0.95 = mAP@0.5:0.95')
    ]
    for code, explanation in fitness_lines:
        story.append(Paragraph(f"<font face='Courier' size='8'>{code}</font>", code_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    story.append(Spacer(1, 0.3*inch))
    key_insight = """
    ╔════════════════════════════════════════════════════════════════════════╗
    ║  KEY INSIGHT: FITNESS = mAP@0.5:0.95                                   ║
    ║                                                                         ║
    ║  YOLO uses ONLY mAP@0.5:0.95 to determine the "best" model.            ║
    ║  Precision, Recall, and mAP@0.5 have ZERO weight in this calculation.  ║
    ║                                                                         ║
    ║  This is because mAP@0.5:0.95 is the strictest metric, requiring       ║
    ║  accurate detection across ALL IoU thresholds from 0.5 to 0.95.        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    story.append(Preformatted(key_insight, code_style))
    
    story.append(PageBreak())
    
    # ==========================================================================
    # SECTION 7: HELPER FUNCTIONS
    # ==========================================================================
    story.append(Paragraph("7. Helper Functions", h1_style))
    
    story.append(Paragraph("7.1 load_yaml()", h2_style))
    yaml_lines = [
        ('def load_yaml(yaml_path):',
         'Function to load a YAML configuration file.'),
        ('with open(yaml_path, "r") as f:',
         'Open the file in read mode.'),
        ('return yaml.safe_load(f)',
         'Parse YAML content into a Python dictionary. safe_load prevents code execution.')
    ]
    for code, explanation in yaml_lines:
        story.append(Paragraph(f"<font face='Courier' size='8'>{code}</font>", code_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    story.append(Paragraph("7.2 load_labels()", h2_style))
    story.append(Paragraph("Converts YOLO format labels to absolute coordinates.", body_style))
    label_lines = [
        ('parts = line.strip().split()',
         'Split label line into parts: [class_id, x_center, y_center, width, height]'),
        ('cls_id = int(parts[0])',
         'First value is the class ID (0, 1, 2, etc.)'),
        ('x_center = float(parts[1]) * img_width',
         'Convert normalized x_center (0-1) to pixel coordinates.'),
        ('x1 = x_center - width / 2',
         'Calculate left edge: center minus half width.'),
        ('x2 = x_center + width / 2',
         'Calculate right edge: center plus half width.')
    ]
    for code, explanation in label_lines:
        story.append(Paragraph(f"<font face='Courier' size='8'>{code}</font>", code_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    story.append(Paragraph("YOLO Label Format:", h3_style))
    label_format = """
    Each line in a .txt label file:
    <class_id> <x_center> <y_center> <width> <height>
    
    Example: 0 0.5 0.5 0.2 0.3
    - Class 0 (e.g., "helmet")
    - Center at 50% width, 50% height
    - Box is 20% of image width, 30% of image height
    
    All values are NORMALIZED (0 to 1), not pixels!
    """
    story.append(Preformatted(label_format, code_style))
    
    story.append(Paragraph("7.3 match_predictions()", h2_style))
    story.append(Paragraph("Matches each prediction to ground truth boxes using IoU.", body_style))
    match_lines = [
        ('sorted_indices = sorted(range(num_preds), key=lambda i: predictions[i]["confidence"], reverse=True)',
         'Sort predictions by confidence (highest first). High-confidence predictions get first chance to match.'),
        ('for pred_idx in sorted_indices:',
         'Process each prediction in order of confidence.'),
        ('iou = box_iou(pred["bbox"], gt["bbox"])',
         'Calculate IoU between prediction and each ground truth box.'),
        ('if iou > best_iou:',
         'Keep track of the best matching ground truth (highest IoU).'),
        ('for t_idx, threshold in enumerate(iou_thresholds):',
         'Check if match is valid at each IoU threshold (0.5, 0.55, ..., 0.95).'),
        ('if best_iou >= threshold: tp[pred_idx, t_idx] = True',
         'If IoU meets threshold, mark as True Positive for that threshold.'),
        ('matched_gt.add(best_gt_idx)',
         'Mark ground truth as matched so it can\'t be matched again (prevents double counting).')
    ]
    for code, explanation in match_lines:
        story.append(Paragraph(f"<font face='Courier' size='8'>{code}</font>", code_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    story.append(PageBreak())
    
    # ==========================================================================
    # SECTION 8: MAIN FUNCTION
    # ==========================================================================
    story.append(Paragraph("8. Main Accuracy Test Function", h1_style))
    story.append(Paragraph("The test_accuracy() function orchestrates the entire accuracy calculation.", body_style))
    
    story.append(Paragraph("8.1 Model Loading", h2_style))
    story.append(Paragraph("<font face='Courier' size='8'>model = YOLO(MODEL_PATH)</font>", code_style))
    story.append(Paragraph("↳ Load your trained model from the .pt file.", explain_style))
    story.append(Paragraph("<font face='Courier' size='8'>class_names = model.names</font>", code_style))
    story.append(Paragraph("↳ Get class names dictionary {0: 'helmet', 1: 'no_helmet', ...}", explain_style))
    
    story.append(Paragraph("8.2 IoU Thresholds Setup", h2_style))
    story.append(Paragraph("<font face='Courier' size='8'>iou_thresholds = np.linspace(0.5, 0.95, 10)</font>", code_style))
    story.append(Paragraph("↳ Creates array [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]. These are the 10 COCO IoU thresholds.", explain_style))
    
    story.append(Paragraph("8.3 Image Processing Loop", h2_style))
    loop_code = [
        ('for img_path in image_paths:',
         'Loop through all test images.'),
        ('img = cv2.imread(img_path)',
         'Read image to get dimensions.'),
        ('ground_truths = load_labels(label_path, img_width, img_height)',
         'Load ground truth boxes from corresponding .txt file.'),
        ('results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)',
         'Run model inference. conf=0.001 gets all possible detections.'),
        ('tp = match_predictions(predictions, ground_truths, iou_thresholds)',
         'Match predictions to ground truth, get True Positive array.')
    ]
    for code, explanation in loop_code:
        story.append(Paragraph(f"<font face='Courier' size='8'>{code}</font>", code_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    story.append(Paragraph("8.4 Final Calculation", h2_style))
    story.append(Paragraph("<font face='Courier' size='8'>metrics = calculate_metrics(all_tp, all_conf, all_pred_cls, all_target_cls, num_classes)</font>", code_style))
    story.append(Paragraph("↳ Calculate all metrics from accumulated results.", explain_style))
    story.append(Paragraph("<font face='Courier' size='8'>fitness_score = fitness(metrics)</font>", code_style))
    story.append(Paragraph("↳ Calculate final fitness score (= mAP@0.5:0.95).", explain_style))
    
    story.append(PageBreak())
    
    # ==========================================================================
    # SECTION 9: YOLO VAL
    # ==========================================================================
    story.append(Paragraph("9. YOLO Built-in Validation", h1_style))
    story.append(Paragraph("The simplest and most accurate method - uses YOLO's own validation code.", body_style))
    
    val_lines = [
        ('model = YOLO(MODEL_PATH)',
         'Load your trained model.'),
        ('results = model.val(data=DATA_YAML, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)',
         'Run validation using YOLO\'s built-in validator. This is IDENTICAL to what runs during training.'),
        ('results.box.mp',
         'Mean Precision across all classes.'),
        ('results.box.mr',
         'Mean Recall across all classes.'),
        ('results.box.map50',
         'mAP at IoU=0.5.'),
        ('results.box.map',
         'mAP at IoU=0.5:0.95 (the FITNESS score).')
    ]
    for code, explanation in val_lines:
        story.append(Paragraph(f"<font face='Courier' size='8'>{code}</font>", code_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    story.append(Spacer(1, 0.3*inch))
    recommendation = """
    ╔════════════════════════════════════════════════════════════════════════╗
    ║  RECOMMENDATION: Use model.val() for production                        ║
    ║                                                                         ║
    ║  The custom implementation is for LEARNING how metrics are calculated. ║
    ║  For actual accuracy testing, model.val() is:                          ║
    ║  • More accurate (handles edge cases)                                  ║
    ║  • Faster (optimized code)                                             ║
    ║  • Same results as training validation                                 ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    story.append(Preformatted(recommendation, code_style))
    
    story.append(PageBreak())
    
    # ==========================================================================
    # SECTION 10: ENTRY POINT
    # ==========================================================================
    story.append(Paragraph("10. Main Entry Point", h1_style))
    
    entry_lines = [
        ('if __name__ == "__main__":',
         'This code only runs when the script is executed directly (not imported as a module).'),
        ('choice = input("Enter 1 or 2: ")',
         'Ask user to choose between custom calculation or YOLO validation.'),
        ('if choice == "1": test_accuracy()',
         'Run custom calculation that shows all formulas.'),
        ('else: test_with_yolo_val()',
         'Run YOLO\'s built-in validation (recommended).')
    ]
    for code, explanation in entry_lines:
        story.append(Paragraph(f"<font face='Courier' size='8'>{code}</font>", code_style))
        story.append(Paragraph(f"↳ {explanation}", explain_style))
    
    # ==========================================================================
    # SUMMARY PAGE
    # ==========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Summary: All Formulas at a Glance", h1_style))
    
    formulas_data = [
        ["Metric", "Formula", "Code Location"],
        ["IoU", "Intersection / Union", "box_iou()"],
        ["Precision", "TP / (TP + FP)", "calculate_metrics()"],
        ["Recall", "TP / (TP + FN)", "calculate_metrics()"],
        ["F1 Score", "2 × P × R / (P + R)", "calculate_metrics()"],
        ["AP", "Area under PR curve (101-point)", "compute_ap()"],
        ["mAP@0.5", "Mean AP at IoU=0.5", "calculate_metrics()"],
        ["mAP@0.5:0.95", "Mean AP at IoU=0.5 to 0.95", "calculate_metrics()"],
        ["Fitness", "1.0 × mAP@0.5:0.95", "fitness()"]
    ]
    
    formula_table = Table(formulas_data, colWidths=[1.5*inch, 2.5*inch, 2*inch])
    formula_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 1.0)])
    ]))
    story.append(formula_table)
    
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Quick Usage Guide", h2_style))
    
    usage = """
    1. Edit MODEL_PATH and DATA_YAML at top of script
    2. Ensure your data.yaml points to correct image/label folders
    3. Run the script: python test_accuracy_with_labels.py
    4. Choose option 1 (custom) or 2 (YOLO built-in)
    5. View results in console and accuracy_results/ folder
    """
    story.append(Preformatted(usage, code_style))
    
    # Build PDF
    doc.build(story)
    print("✅ PDF created: YOLO_Accuracy_Code_Explanation.pdf")


if __name__ == "__main__":
    create_pdf()

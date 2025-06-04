import os
import argparse
import json
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import torch


def draw_predictions(image, results, confidence_threshold=0.25):
    """Draw bounding boxes and labels on the image"""
    # Make a copy of the image
    img_with_boxes = image.copy()

    # Define colors for each class (standing, sitting, lying, throwing)
    colors = [
        (0, 255, 0),    # Green for standing
        (0, 0, 255),    # Blue for sitting
        (255, 0, 0),    # Red for lying
        (255, 255, 0),  # Yellow for throwing
    ]

    # Class names
    class_names = ["standing", "sitting", "lying", "throwing"]

    try:
        # Get boxes, confidence scores, and class IDs
        if hasattr(results[0].boxes, 'xyxy'):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        else:
            # Handle case where boxes might be in a different format
            print("Warning: Expected box format not found, trying alternative format")
            return img_with_boxes

        # Draw each box if above threshold
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            if conf < confidence_threshold:
                continue

            # Get box coordinates
            x1, y1, x2, y2 = map(int, box)

            # Get class color and name
            color = colors[cls_id % len(colors)]
            class_name = class_names[cls_id] if cls_id < len(
                class_names) else f"Class {cls_id}"

            # Draw rectangle and label
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"

            # Calculate text size for better positioning
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw label background
            cv2.rectangle(
                img_with_boxes,
                (x1, y1 - label_height - 5),
                (x1 + label_width, y1),
                color,
                -1  # Fill
            )

            # Draw label text
            cv2.putText(
                img_with_boxes,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
    except Exception as e:
        print(f"Error processing detection results: {e}")

    return img_with_boxes


def load_gt_labels(label_path):
    """Load ground truth labels from JSON file in the dataset format"""
    try:
        with open(label_path, 'r') as f:
            data = json.load(f)

        boxes = []
        classes = []

        # Class mapping
        class_to_idx = {
            "standing": 0,
            "sitting": 1,
            "lying": 2,
            "throwing": 3
        }

        # Process each shape/bounding box
        for shape in data.get('shapes', []):
            label = shape.get('label')
            points = shape.get('points')

            if label in class_to_idx and points and len(points) == 2:
                # Get class index
                class_idx = class_to_idx[label]

                # Extract coordinates (points are in [x1,y1], [x2,y2] format)
                x1, y1 = points[0]
                x2, y2 = points[1]

                # Ensure x1 < x2 and y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                boxes.append([x1, y1, x2, y2])
                classes.append(class_idx)

        return np.array(boxes), np.array(classes)

    except Exception as e:
        print(f"Error loading ground truth label {label_path}: {e}")
        return np.array([]), np.array([])


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    # Calculate IoU
    iou = intersection / union if union > 0 else 0

    return iou


def evaluate_predictions(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes,
                         iou_threshold=0.5, conf_threshold=0.25):
    """
    Evaluate predictions against ground truth
    Returns true positives, false positives, false negatives for each class
    """
    # Initialize counters for each class
    class_names = ["standing", "sitting", "lying", "throwing"]
    num_classes = len(class_names)

    true_positives = {i: 0 for i in range(num_classes)}
    false_positives = {i: 0 for i in range(num_classes)}
    false_negatives = {i: 0 for i in range(num_classes)}

    # Track which ground truth boxes have been matched
    matched_gt = []

    # Filter predictions by confidence
    valid_indices = np.where(pred_scores >= conf_threshold)[0]
    valid_pred_boxes = pred_boxes[valid_indices] if len(
        valid_indices) > 0 else []
    valid_pred_classes = pred_classes[valid_indices] if len(
        valid_indices) > 0 else []
    valid_pred_scores = pred_scores[valid_indices] if len(
        valid_indices) > 0 else []

    # Sort predictions by confidence (highest first)
    if len(valid_pred_scores) > 0:
        sort_indices = np.argsort(-valid_pred_scores)
        valid_pred_boxes = valid_pred_boxes[sort_indices]
        valid_pred_classes = valid_pred_classes[sort_indices]

    # Match predictions to ground truth
    for i, (box, cls) in enumerate(zip(valid_pred_boxes, valid_pred_classes)):
        # Initialize variables for best match
        best_iou = iou_threshold
        best_gt_idx = -1

        # Check each ground truth box
        for j, gt_box in enumerate(gt_boxes):
            # Skip if not the same class or already matched
            if gt_classes[j] != cls or j in matched_gt:
                continue

            # Calculate IoU
            iou = calculate_iou(box, gt_box)

            # Update best match if better
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # Process result
        if best_gt_idx >= 0:
            # True positive
            true_positives[cls] += 1
            matched_gt.append(best_gt_idx)
        else:
            # False positive
            false_positives[cls] += 1

    # Count false negatives (unmatched ground truths)
    for j, gt_cls in enumerate(gt_classes):
        if j not in matched_gt:
            false_negatives[gt_cls] += 1

    return true_positives, false_positives, false_negatives


def draw_predictions_with_gt(image, pred_results, gt_boxes=None, gt_classes=None, confidence_threshold=0.25):
    """Draw both predictions and ground truth on the image"""
    # Make a copy of the image
    img_with_boxes = image.copy()

    # Define colors for each class (standing, sitting, lying, throwing)
    colors = [
        (0, 255, 0),    # Green for standing
        (0, 0, 255),    # Blue for sitting
        (255, 0, 0),    # Red for lying
        (255, 255, 0),  # Yellow for throwing
    ]

    # Class names
    class_names = ["standing", "sitting", "lying", "throwing"]

    try:
        # Draw ground truth boxes with dashed lines (if provided)
        if gt_boxes is not None and gt_classes is not None and len(gt_boxes) > 0:
            for box, cls_id in zip(gt_boxes, gt_classes):
                if cls_id >= len(colors):
                    continue

                # Get box coordinates
                x1, y1, x2, y2 = map(int, box)

                # Get class color and name
                color = colors[cls_id]
                class_name = class_names[cls_id] if cls_id < len(
                    class_names) else f"Class {cls_id}"

                # Draw dashed rectangle for ground truth
                # Draw dashed lines by alternating small segments
                dash_length = 10

                # Draw top
                for i in range(x1, x2, dash_length * 2):
                    x_end = min(i + dash_length, x2)
                    cv2.line(img_with_boxes, (i, y1), (x_end, y1), color, 2)

                # Draw bottom
                for i in range(x1, x2, dash_length * 2):
                    x_end = min(i + dash_length, x2)
                    cv2.line(img_with_boxes, (i, y2), (x_end, y2), color, 2)

                # Draw left
                for i in range(y1, y2, dash_length * 2):
                    y_end = min(i + dash_length, y2)
                    cv2.line(img_with_boxes, (x1, i), (x1, y_end), color, 2)

                # Draw right
                for i in range(y1, y2, dash_length * 2):
                    y_end = min(i + dash_length, y2)
                    cv2.line(img_with_boxes, (x2, i), (x2, y_end), color, 2)

                # Draw label
                label = f"GT: {class_name}"
                cv2.putText(
                    img_with_boxes,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Draw prediction boxes with solid lines
        if hasattr(pred_results[0].boxes, 'xyxy'):
            boxes = pred_results[0].boxes.xyxy.cpu().numpy()
            confs = pred_results[0].boxes.conf.cpu().numpy()
            cls_ids = pred_results[0].boxes.cls.cpu().numpy().astype(int)
        else:
            return img_with_boxes

        # Draw each prediction if above threshold
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            if conf < confidence_threshold:
                continue

            # Get box coordinates
            x1, y1, x2, y2 = map(int, box)

            # Get class color and name
            color = colors[cls_id % len(colors)]
            class_name = class_names[cls_id] if cls_id < len(
                class_names) else f"Class {cls_id}"

            # Draw solid rectangle for predictions
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"

            # Calculate text size for better positioning
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw label background
            cv2.rectangle(
                img_with_boxes,
                (x1, y1 - label_height - 5),
                (x1 + label_width, y1),
                color,
                -1  # Fill
            )

            # Draw label text
            cv2.putText(
                img_with_boxes,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
    except Exception as e:
        print(f"Error drawing predictions with ground truth: {e}")

    return img_with_boxes


def evaluate_model(model_path, test_dir, output_dir, confidence_threshold=0.25, device='', gt_dir=None, limit=0):
    """Evaluate the YOLO model on test images"""
    # Load the model
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all test images
    image_extensions = ['.jpg', '.jpeg', '.png']
    test_images = [
        os.path.join(test_dir, f) for f in os.listdir(test_dir)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]

    if not test_images:
        print(f"No images found in {test_dir}")
        return None

    # Limit the number of images if specified
    if limit > 0 and limit < len(test_images):
        test_images = test_images[:limit]
        print(f"Limiting evaluation to {limit} images")

    print(f"Found {len(test_images)} test images")

    # Track performance metrics
    inference_times = []
    detection_counts = []

    # Initialize metrics for ground truth evaluation
    if gt_dir:
        class_names = ["standing", "sitting", "lying", "throwing"]
        num_classes = len(class_names)

        all_tp = {i: 0 for i in range(num_classes)}
        all_fp = {i: 0 for i in range(num_classes)}
        all_fn = {i: 0 for i in range(num_classes)}

        print(f"Ground truth directory: {gt_dir}")

        has_gt_data = os.path.exists(gt_dir) and os.path.isdir(gt_dir)
        if not has_gt_data:
            print(f"Warning: Ground truth directory not found: {gt_dir}")
            gt_dir = None

    # Process each test image
    for i, img_path in enumerate(test_images):
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        # RGB conversion for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get base name for ground truth lookup
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Load ground truth if available
        gt_boxes = None
        gt_classes = None
        if gt_dir:
            gt_label_path = os.path.join(gt_dir, f"{base_name}.json")
            if os.path.exists(gt_label_path):
                gt_boxes, gt_classes = load_gt_labels(gt_label_path)

        # Measure inference time
        start_time = time.time()
        try:
            # Use task='detect' to explicitly set object detection
            results = model.predict(
                img_rgb, conf=confidence_threshold, device=device, verbose=False)
            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time)

            # Count detections
            if len(results) > 0 and hasattr(results[0].boxes, 'cls'):
                detection_count = len(results[0].boxes.cls)
                detection_counts.append(detection_count)

                # If ground truth is available, evaluate predictions
                if gt_dir and len(gt_boxes) > 0:
                    pred_boxes = results[0].boxes.xyxy.cpu().numpy()
                    pred_classes = results[0].boxes.cls.cpu(
                    ).numpy().astype(int)
                    pred_scores = results[0].boxes.conf.cpu().numpy()

                    # Calculate metrics
                    tp, fp, fn = evaluate_predictions(
                        pred_boxes, pred_classes, pred_scores,
                        gt_boxes, gt_classes,
                        iou_threshold=0.5, conf_threshold=confidence_threshold
                    )

                    # Accumulate metrics
                    for cls in tp:
                        all_tp[cls] += tp[cls]
                        all_fp[cls] += fp[cls]
                        all_fn[cls] += fn[cls]
            else:
                detection_counts.append(0)
                # If ground truth has objects but no detections, count as all false negatives
                if gt_dir and len(gt_boxes) > 0:
                    for cls in np.unique(gt_classes):
                        all_fn[cls] += np.sum(gt_classes == cls)

            # Print progress every 10 images
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(test_images)} images")

            # Draw predictions and ground truth boxes on image
            if gt_dir and len(gt_boxes) > 0:
                img_with_boxes = draw_predictions_with_gt(
                    img, results, gt_boxes, gt_classes, confidence_threshold)
            else:
                img_with_boxes = draw_predictions(
                    img, results, confidence_threshold)

            # Create output path
            base_name_full = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"pred_{base_name_full}")

            # Save output image
            cv2.imwrite(output_path, img_with_boxes)

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

    # Calculate and print statistics
    if inference_times:
        avg_inference_time = np.mean(inference_times)
        avg_detections = np.mean(detection_counts) if detection_counts else 0

        print("Evaluation Summary:")
        print("------------------")
        print(f"Evaluated {len(test_images)} images")
        print(f"Average inference time: {avg_inference_time:.2f} ms")
        print(f"Average detections per image: {avg_detections:.2f}")
        print(f"Total detections: {sum(detection_counts)}")

        # Create a summary file
        summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Evaluation Summary for {model_path}\n")
            f.write(f"Evaluated {len(test_images)} images\n")
            f.write(f"Average inference time: {avg_inference_time:.2f} ms\n")
            f.write(f"Average detections per image: {avg_detections:.2f}\n")
            f.write(f"Total detections: {sum(detection_counts)}\n")
            f.write(f"Confidence threshold: {confidence_threshold}\n")
            f.write(f"Device: {device if device else 'auto'}\n")

            # Add ground truth evaluation results if available
            if gt_dir:
                f.write("\nGround Truth Evaluation:\n")
                f.write("------------------------\n")

                # Calculate precision and recall for each class
                class_names = ["standing", "sitting", "lying", "throwing"]
                total_precision = 0
                total_recall = 0
                classes_with_data = 0

                for cls_id, name in enumerate(class_names):
                    tp = all_tp.get(cls_id, 0)
                    fp = all_fp.get(cls_id, 0)
                    fn = all_fn.get(cls_id, 0)

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / \
                        (precision + recall) if (precision + recall) > 0 else 0

                    if (tp + fp + fn) > 0:
                        total_precision += precision
                        total_recall += recall
                        classes_with_data += 1

                    f.write(f"Class: {name}\n")
                    f.write(f"  True Positives: {tp}\n")
                    f.write(f"  False Positives: {fp}\n")
                    f.write(f"  False Negatives: {fn}\n")
                    f.write(f"  Precision: {precision:.4f}\n")
                    f.write(f"  Recall: {recall:.4f}\n")
                    f.write(f"  F1 Score: {f1:.4f}\n\n")

                    # Also print to console
                    print(f"\nClass: {name}")
                    print(f"  Precision: {precision:.4f}")
                    print(f"  Recall: {recall:.4f}")
                    print(f"  F1 Score: {f1:.4f}")

                # Calculate mAP (mean average precision)
                if classes_with_data > 0:
                    mean_precision = total_precision / classes_with_data
                    mean_recall = total_recall / classes_with_data

                    f.write(f"Mean Precision: {mean_precision:.4f}\n")
                    f.write(f"Mean Recall: {mean_recall:.4f}\n")

                    print(f"\nMean Precision: {mean_precision:.4f}")
                    print(f"Mean Recall: {mean_recall:.4f}")

        # Plot inference time histogram
        plt.figure(figsize=(10, 6))
        plt.hist(inference_times, bins=20)
        plt.axvline(x=50, color='r', linestyle='--', label='50ms threshold')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Frequency')
        plt.title(
            f'Inference Time Distribution (Avg: {avg_inference_time:.2f} ms)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inference_time_hist.png'))

        # Plot detections histogram if we have data
        if detection_counts:
            plt.figure(figsize=(10, 6))
            plt.hist(detection_counts, bins=range(max(detection_counts)+2))
            plt.xlabel('Number of Detections')
            plt.ylabel('Frequency')
            plt.title(f'Detections per Image (Avg: {avg_detections:.2f})')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'detections_hist.png'))

        # Return metrics
        metrics_result = {
            'avg_inference_time': avg_inference_time,
            'avg_detections': avg_detections,
            'total_detections': sum(detection_counts),
            'num_images': len(test_images),
            'summary_path': summary_path
        }

        # Add ground truth metrics if available
        if gt_dir:
            metrics_result['ground_truth_evaluation'] = True
            metrics_result['precision'] = total_precision / \
                classes_with_data if classes_with_data > 0 else 0
            metrics_result['recall'] = total_recall / \
                classes_with_data if classes_with_data > 0 else 0

        return metrics_result
    else:
        print("No images were successfully processed")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained YOLO model")
    parser.add_argument("--test-dir", type=str, default="CV_Test/Images",
                        help="Directory containing test images")
    parser.add_argument("--output-dir", type=str, default="test_outputs/yolo",
                        help="Directory to save outputs")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--device", type=str, default='',
                        help="Device to use for inference (empty for auto)")
    parser.add_argument("--gt-dir", type=str, default="",
                        help="Optional path to ground truth labels for evaluation")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit evaluation to this many images (0 for all)")

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model not found at: {args.model}")
        return

    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"Test directory not found: {args.test_dir}")
        return

    # Customize output directory with model name if not explicit
    if args.output_dir == "test_outputs/yolo":
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"test_outputs/yolo_{model_name}_{timestamp}"

    # Run evaluation
    print(f"Starting evaluation on {args.test_dir}...")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Output directory: {args.output_dir}")

    metrics = evaluate_model(
        args.model,
        args.test_dir,
        args.output_dir,
        args.confidence,
        args.device,
        args.gt_dir,
        args.limit
    )

    if metrics:
        print("\nEvaluation complete!")
        print(f"Results saved to {args.output_dir}")
        print(
            f"Average inference time: {metrics['avg_inference_time']:.2f} ms")
        print(f"Average detections per image: {metrics['avg_detections']:.2f}")
        print(f"Total detections: {metrics['total_detections']}")
        print(f"Summary saved to: {metrics.get('summary_path', 'N/A')}")

        if 'ground_truth_evaluation' in metrics:
            print("\nGround Truth Evaluation:")
            print(f"Mean Precision: {metrics.get('precision', 0):.4f}")
            print(f"Mean Recall: {metrics.get('recall', 0):.4f}")


if __name__ == "__main__":
    main()
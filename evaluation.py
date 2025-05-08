import torch
import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models import ActionModel
from sklearn.metrics import confusion_matrix, classification_report

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_IMG_DIR = "CV_Test/Images"  # Path to test images
TEST_LABEL_DIR = "CV_Test/Labels"  # Path to ground truth JSON labels
DEFAULT_MODEL_PATH = "action_detection_model.pth"  # Default model path
CHECKPOINT_DIR = "checkpoints"  # Directory containing checkpoint models
OUTPUT_DIR = "test_outputs"
NUM_CLASSES = 4  # standing, sitting, lying, throwing
CLASS_NAMES = {0: "standing", 1: "sitting", 2: "lying", 3: "throwing"}
CLASS_LIST = ["standing", "sitting", "lying", "throwing"]

# Image transformation should match what was used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path, num_classes):
    """Load model from file - handles both full checkpoints and state_dict only files"""
    model = ActionModel(num_classes=num_classes)

    try:
        # First try loading as a complete checkpoint dictionary
        checkpoint = torch.load(model_path, map_location=DEVICE)

        # If the checkpoint is a dict with 'model_state_dict' key
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(
                f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        # If the file contains just the state_dict directly
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights directly")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def predict(model, image_path):
    """Make a prediction on a single image and measure inference time"""
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None, None

    # Get original image for visualization
    orig_img = img.copy()

    # Transform for model input
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    # Measure inference time
    start = time.time()
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    elapsed = (time.time() - start) * 1000

    # Get prediction and probability
    prob, pred = probs.max(1)
    class_name = CLASS_NAMES[int(pred)]
    confidence = float(prob)

    return {
        'label': class_name,
        'confidence': confidence,
        'time_ms': elapsed,
        'pred_idx': int(pred),
        'probs': probs.cpu().numpy()[0],
        'orig_img': orig_img
    }


def visualize_prediction(img, prediction, gt_label=None):
    """Create a visualization of the prediction with confidence scores"""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot image with prediction
    ax1.imshow(img)
    title = f"Prediction: {prediction['label']} ({prediction['confidence']:.2%})"
    if gt_label:
        correct = prediction['label'] == gt_label
        title += f"\nGround Truth: {gt_label} ({'✓' if correct else '✗'})"
    ax1.set_title(title)
    ax1.axis('off')

    # Plot confidence bars for each class
    probs = prediction['probs']
    y_pos = np.arange(len(CLASS_LIST))
    ax2.barh(y_pos, probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(CLASS_LIST)
    ax2.set_xlabel('Confidence')
    ax2.set_title('Class Probabilities')

    # Highlight the predicted class
    pred_idx = prediction['pred_idx']
    ax2.get_children()[pred_idx].set_color('red')

    plt.tight_layout()
    return fig


def save_results(predictions, metrics, output_dir):
    """Save results to files and create visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    # Save textual results
    results_file = os.path.join(output_dir, "results.txt")
    with open(results_file, 'w') as f:
        # Write overall metrics
        if metrics:
            f.write("=== EVALUATION METRICS ===\n")
            f.write(f"Total images: {metrics['total']}\n")
            if 'accuracy' in metrics:
                f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
            f.write("\n=== CLASSIFICATION REPORT ===\n")
            if 'report' in metrics:
                f.write(metrics['report'])
            f.write("\n\n")

        # Write individual predictions
        f.write("=== INDIVIDUAL PREDICTIONS ===\n")
        for img_name, pred in predictions.items():
            f.write(
                f"{img_name}: {pred['label']} ({pred['confidence']:.2%}, {pred['time_ms']:.1f} ms)")
            if 'gt_label' in pred:
                f.write(f" | GT: {pred['gt_label']}")
            f.write("\n")

    # Create and save confusion matrix if we have ground truth
    if metrics and 'confusion_matrix' in metrics:
        plt.figure(figsize=(10, 8))
        cm = metrics['confusion_matrix']
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(CLASS_LIST))
        plt.xticks(tick_marks, CLASS_LIST, rotation=45)
        plt.yticks(tick_marks, CLASS_LIST)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

    # Save visualizations for sample predictions (limit to 20)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    sample_count = min(20, len(predictions))
    sample_keys = list(predictions.keys())[:sample_count]

    for img_name in sample_keys:
        pred = predictions[img_name]
        gt_label = pred.get('gt_label', None)
        fig = visualize_prediction(pred['orig_img'], pred, gt_label)
        fig.savefig(os.path.join(
            vis_dir, f"{os.path.splitext(img_name)[0]}_pred.png"))
        plt.close(fig)

    print(f"Results saved to {output_dir}")
    print(f"- Text results: {results_file}")
    print(f"- Visualizations: {vis_dir}")
    if 'confusion_matrix' in metrics:
        print(
            f"- Confusion matrix: {os.path.join(output_dir, 'confusion_matrix.png')}")


def list_available_checkpoints():
    """List all available model checkpoints"""
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Checkpoint directory {CHECKPOINT_DIR} not found")
        return []

    checkpoints = []
    for file in os.listdir(CHECKPOINT_DIR):
        if file.endswith('.pth'):
            checkpoints.append(os.path.join(CHECKPOINT_DIR, file))

    if os.path.exists(DEFAULT_MODEL_PATH):
        checkpoints.append(DEFAULT_MODEL_PATH)

    return checkpoints


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate action detection model")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Path to model weights (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--best', action='store_true',
                        help='Use best checkpoint model from training')
    parser.add_argument('--list-checkpoints', action='store_true',
                        help='List available checkpoint models')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help=f'Directory to save results (default: {OUTPUT_DIR})')
    parser.add_argument('--visualize-all', action='store_true',
                        help='Create visualizations for all test images (otherwise limit to 20)')
    args = parser.parse_args()

    # List checkpoints if requested
    if args.list_checkpoints:
        checkpoints = list_available_checkpoints()
        if checkpoints:
            print("Available checkpoint models:")
            for i, ckpt in enumerate(checkpoints):
                print(f"  [{i}] {ckpt}")
        else:
            print("No checkpoint models found")
        return

    # Determine which model to use
    model_path = args.model
    if args.best and os.path.exists(os.path.join(CHECKPOINT_DIR, "best_model.pth")):
        model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        print(f"Using best model checkpoint: {model_path}")

    print(f"Using device: {DEVICE}")

    # Check for ground truth labels
    use_gt = os.path.isdir(TEST_LABEL_DIR)
    if use_gt:
        print(
            f"Found ground-truth labels in {TEST_LABEL_DIR}, computing accuracy metrics")

    # Verify test images directory
    if not os.path.isdir(TEST_IMG_DIR):
        print(f"Error: Test image directory not found at {TEST_IMG_DIR}")
        return

    # Get test images
    test_images = sorted([f for f in os.listdir(TEST_IMG_DIR)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not test_images:
        print(f"No images found in {TEST_IMG_DIR}")
        return

    print(f"Found {len(test_images)} test images")

    # Load model
    model = load_model(model_path, NUM_CLASSES)
    if model is None:
        return

    # Process all images
    predictions = {}
    all_gt_labels = []
    all_pred_labels = []

    for img_name in test_images:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        prediction = predict(model, img_path)

        if prediction['label'] is None:
            continue

        print(
            f"{img_name}: {prediction['label']} ({prediction['confidence']:.2%}, {prediction['time_ms']:.1f} ms)")

        # Check ground truth if available
        if use_gt:
            json_name = os.path.splitext(img_name)[0] + ".json"
            gt_path = os.path.join(TEST_LABEL_DIR, json_name)
            if os.path.isfile(gt_path):
                with open(gt_path, 'r') as jf:
                    data = json.load(jf)
                shapes = data.get('shapes', [])
                if shapes:
                    gt_label = shapes[0].get('label')
                    prediction['gt_label'] = gt_label
                    all_gt_labels.append(gt_label)
                    all_pred_labels.append(prediction['label'])

                    # Print ✓ or ✗ to indicate correct/incorrect
                    correct = gt_label == prediction['label']
                    print(
                        f"  Ground truth: {gt_label} {'✓' if correct else '✗'}")
            else:
                print(f"  Warning: GT label file not found for {img_name}")

        predictions[img_name] = prediction

    # Calculate metrics
    metrics = {'total': len(predictions)}

    if use_gt and all_gt_labels:
        # Only include images where we had ground truth
        correct = sum(1 for gt, pred in zip(
            all_gt_labels, all_pred_labels) if gt == pred)
        accuracy = 100.0 * correct / len(all_gt_labels)

        # Create confusion matrix and classification report
        cm = confusion_matrix(
            all_gt_labels, all_pred_labels, labels=CLASS_LIST)
        report = classification_report(
            all_gt_labels, all_pred_labels, labels=CLASS_LIST)

        # Add to metrics
        metrics.update({
            'correct': correct,
            'total_with_gt': len(all_gt_labels),
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'report': report
        })

        print(f"\nAccuracy: {correct}/{len(all_gt_labels)} = {accuracy:.2f}%")
        print("\nClassification Report:")
        print(report)

    # Save results
    save_results(predictions, metrics, args.output)


if __name__ == '__main__':
    main()

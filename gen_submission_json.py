#!/usr/bin/env python
# filepath: /home/ahrilab/Desktop/CV/Computer_vision/generate_json_labels.py

import os
import json
import argparse
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
import torch
import timm
from torchvision import transforms
from PIL import Image
import time

from PIL import ImageOps

class PadToSquare:
    def __call__(self, image):
        w, h = image.size
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        return ImageOps.expand(image, padding, fill=(0, 0, 0))
# Action class mapping
idx_to_class = {
    0: "standing",
    1: "sitting",
    2: "lying",
    3: "throwing"
}

# Transform for classifier input
classifier_transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def create_json_label(image_path, predictions, classifier, device, confidence_threshold=0.25):
    """Create a JSON label using YOLO for detection and MobilenetV3 for classification."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None

    image_height, image_width = img.shape[:2]
    image_name = os.path.basename(image_path)

    label_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    if len(predictions) > 0 and hasattr(predictions[0].boxes, 'xyxy'):
        boxes = predictions[0].boxes.xyxy.cpu().numpy()
        confs = predictions[0].boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            if conf < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]

            if crop_img.size == 0:
                continue  # skip invalid crops

            # Convert BGR crop to RGB PIL image
            crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            input_tensor = classifier_transform(crop_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = classifier(input_tensor).argmax(1).item()

            class_name = idx_to_class.get(pred, f"class_{pred}")

            shape = {
                "label": class_name,
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                # "description": f"confidence: {float(conf):.2f}",
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            }

            label_data["shapes"].append(shape)

    return label_data


def generate_labels(model_path, classifier_path, image_dir, output_dir, confidence_threshold=0.25, device=''):
    """Generate JSON label files for all images in a directory using YOLO + classifier."""
    # Load YOLO detection model
    try:
        model = YOLO(model_path)
        print(f"✅ Loaded YOLO model from {model_path}")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return False

    # Load MobilenetV3 classifier
    try:
        classifier = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=4)
        classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
        classifier.eval()
        classifier.to(device)
        print(f"✅ Loaded classifier from {classifier_path}")
    except Exception as e:
        print(f"❌ Error loading classifier: {e}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]

    if not image_files:
        print(f"No images found in {image_dir}")
        return False

    print(f"📸 Found {len(image_files)} images in {image_dir}")

    total_time = 0
    count = 0

    for image_path in tqdm(image_files, desc="🔍 Processing images"):
        try:
            start_time = time.time()

            results = model.predict(image_path, conf=confidence_threshold, device=device, verbose=False)

            label_data = create_json_label(image_path, results, classifier, device, confidence_threshold)

            end_time = time.time()
            total_time += (end_time - start_time)
            count += 1

            if label_data:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}.json")
                with open(output_path, 'w') as f:
                    json.dump(label_data, f, indent=4)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    avg_time = total_time / count * 1000  # milliseconds
    print(f"\n  Average inference time per image: {avg_time:.2f} ms")

    print(f"✅ Label generation complete. JSONs saved to: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate JSON labels using YOLO + classifier")
    parser.add_argument("--model", type=str, default="/root/dev/Computer_vision_new/yolo_runs/run1/weights/best.pt", help="YOLO model path")
    parser.add_argument("--classifier", type=str, default="mobilenetv3_action_cls_best.pth", help="Classifier weights")
    parser.add_argument("--image-dir", type=str, default="dataset/CV_Test/Images", help="Input image directory")
    parser.add_argument("--output-dir", type=str, default="submission_json", help="Output JSON label directory")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default='cuda:3', help="Inference device (e.g., 'cuda:0' or 'cpu')")

    args = parser.parse_args()

    # Check paths
    if not os.path.exists(args.model):
        print(f"❌ YOLO model not found: {args.model}")
        return
    if not os.path.exists(args.classifier):
        print(f"❌ Classifier model not found: {args.classifier}")
        return
    if not os.path.exists(args.image_dir):
        print(f"❌ Image directory not found: {args.image_dir}")
        return

    print(f"▶ Running label generation with:")
    print(f" - YOLO model: {args.model}")
    print(f" - Classifier: {args.classifier}")
    print(f" - Images: {args.image_dir}")
    print(f" - Output: {args.output_dir}")
    print(f" - Device: {args.device or 'auto'}")

    generate_labels(
        model_path=args.model,
        classifier_path=args.classifier,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
        device=args.device
    )


if __name__ == "__main__":
    main()

    # python generate_json_labels.py \
    # --model yolo_runs/run12/weights/best.pt \
    # --classifier mobilenetv3_action_cls_best.pth \
    # --image-dir CV_Test/Images \
    # --output-dir CV_Test/Labels \
    # --confidence 0.3 \
    # --device cuda:0


# YOLO Action Detection

This project implements action detection (standing, sitting, lying, throwing) using YOLOv8 and the Ultralytics library.

## Setup

1. Install the required packages:
   ```
   pip install ultralytics opencv-python matplotlib torch wandb
   ```

2. Convert the dataset to YOLO format:
   ```
   python convert_to_yolo.py
   ```

3. Train the YOLO model:
   ```
   python train_yolo.py --model yolov8n.pt --epochs 50
   ```

4. Evaluate the trained model:
   ```
   python evaluate_yolo.py --model yolo_runs/run1/weights/best.pt
   ```

## Automated Pipeline

Run the entire training and evaluation pipeline automatically:
```
./run_yolo_pipeline.sh
```

## Scripts Description

- **convert_to_yolo.py**: Converts the dataset from JSON format to YOLO format
- **train_yolo.py**: Trains a YOLO model on the converted dataset
- **evaluate_yolo.py**: Evaluates the trained model on test images
- **run_yolo_pipeline.sh**: Automates the entire workflow

## Dataset Structure

The dataset contains images with bounding boxes for four action classes:
- standing
- sitting
- lying 
- throwing

## Model Options

You can choose from various YOLOv8 model sizes:
- yolov8n.pt (nano) - fastest but less accurate
- yolov8s.pt (small)
- yolov8m.pt (medium)
- yolov8l.pt (large)
- yolov8x.pt (xlarge) - slowest but most accurate
# YOLO Action Detection

This project implements action detection (standing, sitting, lying, throwing) using YOLOv8 and the Ultralytics library.

Adjust script arguments (e.g., `--epochs`, `--model`, `--device`, other path... ) as needed based on your setup and performance requirements!

## Setup

1. Install the required packages:
   ```
   pip install ultralytics opencv-python matplotlib torch timm wandb
   ```

2. Convert LabelMe JSON to YOLO Format:
   ```
   python convert_label_to_yolo.py
   ```

3. Generate Action Classification Crops:
   ```
   python gen_action_crops.py
   ```

4. Train YOLOv8 Detection Model:
   ```
   python train_yolo.py
   ```

5. Train MobileNetV3 Action Classifier:
   ```
   python train_mobilenetv3.py
   ```

6. Run Inference and Generate Submission JSONs:
   ```
   python gen_submission_json.py
   ```

7. Create zip file for Submission
   ```
   zip -r submission_json.zip submission_json
   ```

## Automated Pipeline

Run the entire training and evaluation pipeline automatically:
```
./run_all.sh
```

## Scripts Description


- **convert_label_to_yolo.py**  
  Converts LabelMe-style JSON annotations to YOLO format.

- **gen_action_crops.py**  
  Generates cropped human action images for classification.

- **train_yolo.py**  
  Trains a YOLO model on the person-detection dataset.

- **train_mobilenetv3.py**  
  Trains a MobileNetV3 model for action classification.

- **gen_submission_json.py**  
  Runs YOLO detection + classification and generates submission JSONs.

- **run_all.sh**  
  Automates the entire pipeline from preprocessing to final submission.

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

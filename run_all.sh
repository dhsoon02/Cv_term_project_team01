#!/bin/bash

echo "Start full pipeline..."

# 1.
echo "🔧 Converting LabelMe JSON to YOLO format..."
python convert_label_to_yolo.py

# 2.
echo "Generating action classification crops..."
python gen_action_crops.py

# 3.
echo "Training YOLO model..."
python train_yolo.py

# 4.
echo "Training MobileNetV3 classifier..."
python train_mobilenetv3.py

# 5. Inference + submission json generation
echo "Generating submission JSONs..."
python gen_submission_json.py

# 6. Zip
echo "Zipping final submission..."
zip -r submission_json.zip submission_json

echo "All done!"

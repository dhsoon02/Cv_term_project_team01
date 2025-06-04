import os
import sys
from ultralytics import YOLO
from pathlib import Path
import wandb
import argparse


def main(args):
    """
    Train a YOLO model using Ultralytics on the converted dataset
    """
    # Set up WandB logging if enabled
    if args.use_wandb:
        wandb.init(
            project="action-detection-yolo",
            config={
                "model": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "image_size": args.img_size,
                "dataset": "Custom Action Dataset",
                # Augmentation parameters
                "degrees": args.degrees,
                "translate": args.translate,
                "scale": args.scale,
                "shear": args.shear,
                "perspective": args.perspective,
                "flipud": args.flipud,
                "fliplr": args.fliplr,
                "mosaic": args.mosaic,
                "mixup": args.mixup,
                "copy_paste": args.copy_paste
            }
        )

    # Path to the dataset configuration
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dataset_yolo_only_person",
        "human_action.yaml"
    )

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset configuration not found at: {dataset_path}")
        print("Please run convert_to_yolo.py first to prepare the dataset.")
        sys.exit(1)

    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    try:
        # Configure training arguments
        train_args = {
            "data": dataset_path,
            "epochs": args.epochs,
            "batch": args.batch_size,
            "imgsz": args.img_size,
            "patience": args.patience,
            "device": args.device,
            "project": args.project_name,
            "name": args.run_name,
            "save": True,
            "save_period": args.save_period,
            "pretrained": True,
            "verbose": args.verbose,
            "val": False,  # Disable validation during training

            # Data augmentation parameters
            "degrees": args.degrees,           # Rotation degrees
            "translate": args.translate,       # Translation
            "scale": args.scale,               # Scale
            "shear": args.shear,               # Shear
            "perspective": args.perspective,   # Perspective
            "flipud": args.flipud,             # Flip up-down
            "fliplr": args.fliplr,             # Flip left-right
            "mosaic": args.mosaic,             # Mosaic augmentation
            "mixup": args.mixup,               # Mixup augmentation
            "copy_paste": args.copy_paste      # Copy-paste augmentation
        }

        # Optional arguments based on Ultralytics version compatibility
        try:
            # First attempt with newer parameters
            results = model.train(
                **train_args,
                optimizer=args.optimizer,
                lr0=args.learning_rate,
                lrf=args.final_learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                warmup_momentum=args.warmup_momentum,
                warmup_bias_lr=args.warmup_bias_lr
            )
        except TypeError as e:
            # Fallback to simpler parameter set if there's a compatibility issue
            print(
                f"Warning: Incompatible training parameters, using simplified configuration: {str(e)}")
            results = model.train(**train_args)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

    # Skip validation after training
    print("\nTraining complete!")
    print(f"Model saved to: {os.path.join(args.project_name, args.run_name)}")

    # Close WandB run if enabled
    if args.use_wandb:
        wandb.finish()

    return results, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on action detection dataset")
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                        help='Model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (empty string for auto, or cuda:0, cpu)')
    parser.add_argument('--project-name', type=str, default='yolo_runs',
                        help='Project name for saving outputs')
    parser.add_argument('--run-name', type=str, default='run1',
                        help='Run name for this training session')
    parser.add_argument('--save-period', type=int, default=10,
                        help='Save checkpoint every X epochs')
    parser.add_argument('--optimizer', type=str, default='auto',
                        help='Optimizer: SGD, Adam, AdamW, RMSProp, auto')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--final-learning-rate', type=float, default=0.001,
                        help='Final learning rate')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='Optimizer momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Optimizer weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='Warmup epochs')
    parser.add_argument('--warmup-momentum', type=float, default=0.8,
                        help='Warmup momentum')
    parser.add_argument('--warmup-bias-lr', type=float, default=0.1,
                        help='Warmup bias learning rate')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    # Data augmentation parameters
    parser.add_argument('--degrees', type=float, default=10.0,
                        help='Rotation augmentation degrees (±)')
    parser.add_argument('--translate', type=float, default=0.1,
                        help='Translation augmentation fraction (±)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Scale augmentation fraction (±)')
    parser.add_argument('--shear', type=float, default=0.0,
                        help='Shear augmentation degrees (±)')
    parser.add_argument('--perspective', type=float, default=0.0,
                        help='Perspective augmentation distortion')
    parser.add_argument('--flipud', type=float, default=0.0,
                        help='Flip up-down augmentation probability')
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='Flip left-right augmentation probability')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, default=0.1,
                        help='Mixup augmentation probability')
    parser.add_argument('--copy-paste', type=float, default=0.1,
                        help='Copy-paste augmentation probability')

    args = parser.parse_args()
    main(args)
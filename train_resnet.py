import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
import wandb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime
import time
from dataset import ActionDataset, create_train_test_split
from models import ResNetActionModel

# Define constants
NUM_CLASSES = 4  # standing, sitting, lying, throwing
BATCH_SIZE = 128  # Adjust based on your GPU memory
EPOCHS = 500  # Adjust as needed
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_IMG_DIR = "CV_Train/Images"
TRAIN_LABEL_DIR = "CV_Train/Labels"
CHECKPOINT_DIR = "checkpoints"  # Directory to save checkpoints
SAVE_INTERVAL = 100  # Save a checkpoint every N epochs
RESNET_TYPE = 34  # 18, 34, 50, 101, or 152
MODEL_NAME = f"resnet{RESNET_TYPE}_action"  # Used for saving models

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Custom noise augmentation class that preserves bounding box positions


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.transform = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, img):
        return self.transform(img)


def visualize_batch(images, labels, predictions=None, num_samples=4):
    """Visualize a batch of images with labels and predictions"""
    class_names = ["standing", "sitting", "lying", "throwing"]
    plt.figure(figsize=(12, 3*min(num_samples, len(images))))

    for i in range(min(num_samples, len(images))):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        # Denormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.subplot(min(num_samples, len(images)), 2, 2*i+1)
        plt.imshow(img)
        title = f"True: {class_names[labels[i].item()]}"
        if predictions is not None:
            title += f" | Pred: {class_names[predictions[i].item()]}"
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    return plt


def main():
    # Access global constants
    global TRAIN_IMG_DIR, TRAIN_LABEL_DIR

    # Initialize wandb
    wandb.init(
        project="action-detection-resnet",
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "model": f"ResNet{RESNET_TYPE}",
            "dataset": "Custom Action Dataset",
            "num_classes": NUM_CLASSES,
            "data_augmentation": "Noise + ColorJitter",
        }
    )

    print(f"Using device: {DEVICE}")

    # --- 1. Split Data and Load ---
    # Find the source data directory by looking at parents of CV_Train
    base_dir = os.path.dirname(os.path.dirname(TRAIN_IMG_DIR))
    source_img_dir = os.path.join(base_dir, "Images")
    source_label_dir = os.path.join(base_dir, "Labels")

    # Check if we need to create the split (if source directories exist)
    if os.path.exists(source_img_dir) and os.path.exists(source_label_dir):
        print(f"Found source data directories. Creating 80/20 train/test split...")
        train_img_dir, train_label_dir, test_img_dir, test_label_dir = create_train_test_split(
            source_img_dir, source_label_dir, test_size=0.2, seed=42
        )
        # Update paths to use the output from create_train_test_split
        TRAIN_IMG_DIR = train_img_dir
        TRAIN_LABEL_DIR = train_label_dir
        print(f"Using train data from {TRAIN_IMG_DIR}")
    else:
        print(f"Using predefined train/test split directories")

    # Define transforms for training with position-preserving augmentations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # Adding small noise that won't affect position
        AddGaussianNoise(0., 0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Define transforms for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Set up a validation split
    full_dataset = ActionDataset(
        img_dir=TRAIN_IMG_DIR, label_dir=TRAIN_LABEL_DIR, transform=train_transform)
    dataset_size = len(full_dataset)
    val_split = 0.2  # 20% for validation
    train_size = int((1 - val_split) * dataset_size)
    val_size = dataset_size - train_size

    # Create train and validation datasets with different transforms
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    val_dataset_with_augment = torch.utils.data.Subset(
        full_dataset, range(train_size, dataset_size))

    # Create a separate validation dataset with no augmentation
    val_dataset = ActionDataset(
        img_dir=TRAIN_IMG_DIR, label_dir=TRAIN_LABEL_DIR, transform=val_transform)
    val_dataset = torch.utils.data.Subset(
        val_dataset, range(train_size, dataset_size))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")

    # --- 2. Define Model ---
    model = ResNetActionModel(
        num_classes=NUM_CLASSES, pretrained=True, resnet_type=RESNET_TYPE).to(DEVICE)
    wandb.watch(model, log="all")  # Log model gradients and parameters

    # --- 3. Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # --- 4. Training Loop ---
    best_val_acc = 0.0
    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # Log batch metrics
            if batch_idx % 5 == 0:  # Log every 5 batches
                wandb.log({
                    "batch_train_loss": loss.item(),
                    "batch": batch_idx + epoch * len(train_loader)
                })

        # Validate after each epoch
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        inference_times = []

        with torch.no_grad():
            for images, labels in val_loader:
                start_time = time.time()
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                inference_time = (time.time() - start_time) * \
                    1000 / images.size(0)  # ms per image
                inference_times.append(inference_time)

                loss = criterion(outputs, labels)

                # Statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # For confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

                # Visualize last batch
                if images.size(0) >= 4:  # Only save if we have at least 4 images
                    last_batch_images = images
                    last_batch_labels = labels
                    last_batch_preds = predicted

        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        avg_inference_time = sum(inference_times) / len(inference_times)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Generate and log confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        class_names = ["standing", "sitting", "lying", "throwing"]

        # Create confusion matrix figure
        fig_cm, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        # Make sure the tick counts match the labels
        n_classes = len(class_names)
        tick_marks = np.arange(n_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45,
                 ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.tight_layout()

        # Create inference time histogram
        fig_time, ax_time = plt.subplots(figsize=(10, 6))
        ax_time.hist(inference_times, bins=20)
        ax_time.axvline(x=50, color='r', linestyle='--',
                        label='50ms threshold')
        ax_time.set_xlabel('Inference Time (ms)')
        ax_time.set_ylabel('Frequency')
        ax_time.set_title(
            f'Inference Time Distribution (Avg: {avg_inference_time:.2f} ms)')
        ax_time.legend()

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "avg_inference_time": avg_inference_time,
            "confusion_matrix": wandb.Image(fig_cm),
            "inference_time_hist": wandb.Image(fig_time),
            "sample_predictions": wandb.Image(visualize_batch(
                last_batch_images, last_batch_labels, last_batch_preds))
        })

        plt.close('all')  # Close all figures to prevent memory leaks

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{EPOCHS}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"Inference Time: {avg_inference_time:.2f} ms/image")

        # Save checkpoints
        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"{MODEL_NAME}_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(
                CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(
                f"New best model saved with validation accuracy: {best_val_acc:.2f}%")

    print("Training complete.")

    # --- 5. Save Final Model ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(
        CHECKPOINT_DIR, f"{MODEL_NAME}_final_{timestamp}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Also save to standard path for compatibility with existing code
    std_model_path = f"{MODEL_NAME}.pth"
    torch.save(model.state_dict(), std_model_path)
    print(f"Model also saved to {std_model_path}")

    # Finish the wandb run
    wandb.finish()


if __name__ == '__main__':
    main()

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import numpy as np
import random
import shutil


class ActionDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        # Map action labels to indices
        self.class_to_idx = {"standing": 0,
                             "sitting": 1, "lying": 2, "throwing": 3}
        # List image files
        self.image_files = sorted([f for f in os.listdir(
            self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Load corresponding label JSON
        label_name = os.path.splitext(img_name)[0] + ".json"
        label_path = os.path.join(self.label_dir, label_name)
        with open(label_path, 'r') as f:
            data = json.load(f)
        shapes = data.get('shapes', [])
        # Get first shape label
        if shapes:
            label_str = shapes[0].get('label', '')
            label_idx = self.class_to_idx.get(label_str, -1)
        else:
            label_idx = -1
        return image, torch.tensor(label_idx, dtype=torch.long)


def create_train_test_split(img_dir, label_dir, test_size=0.2, seed=42):
    """
    Creates or updates train/test split by physically separating files into 
    CV_Train and CV_Test directories with an 80/20 split

    Args:
        img_dir: Source directory containing all images
        label_dir: Source directory containing all labels
        test_size: Proportion of the dataset to include in the test split (default: 0.2)
        seed: Random seed for reproducibility

    Returns:
        train_img_dir, train_label_dir, test_img_dir, test_label_dir paths
    """
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Create train and test directories if they don't exist
    train_img_dir = os.path.join(
        os.path.dirname(img_dir), "CV_Train", "Images")
    train_label_dir = os.path.join(
        os.path.dirname(img_dir), "CV_Train", "Labels")
    test_img_dir = os.path.join(os.path.dirname(img_dir), "CV_Test", "Images")
    test_label_dir = os.path.join(
        os.path.dirname(img_dir), "CV_Test", "Labels")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # Get all image files
    img_files = sorted([f for f in os.listdir(img_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not img_files:
        print(f"No image files found in {img_dir}")
        return train_img_dir, train_label_dir, test_img_dir, test_label_dir

    # Shuffle and split
    random.shuffle(img_files)
    split_idx = int(len(img_files) * (1 - test_size))
    train_files = img_files[:split_idx]
    test_files = img_files[split_idx:]

    print(
        f"Splitting dataset: {len(train_files)} training samples, {len(test_files)} test samples")

    # Move files to respective directories
    for img_file in train_files:
        base_name = os.path.splitext(img_file)[0]

        # Copy image
        src_img = os.path.join(img_dir, img_file)
        dst_img = os.path.join(train_img_dir, img_file)
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)

        # Copy label if exists
        label_file = f"{base_name}.json"
        src_label = os.path.join(label_dir, label_file)
        dst_label = os.path.join(train_label_dir, label_file)
        if os.path.exists(src_label) and not os.path.exists(dst_label):
            shutil.copy2(src_label, dst_label)

    for img_file in test_files:
        base_name = os.path.splitext(img_file)[0]

        # Copy image
        src_img = os.path.join(img_dir, img_file)
        dst_img = os.path.join(test_img_dir, img_file)
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)

        # Copy label if exists
        label_file = f"{base_name}.json"
        src_label = os.path.join(label_dir, label_file)
        dst_label = os.path.join(test_label_dir, label_file)
        if os.path.exists(src_label) and not os.path.exists(dst_label):
            shutil.copy2(src_label, dst_label)

    print(
        f"Train data: {len(os.listdir(train_img_dir))} images, {len(os.listdir(train_label_dir))} labels")
    print(
        f"Test data: {len(os.listdir(test_img_dir))} images, {len(os.listdir(test_label_dir))} labels")

    return train_img_dir, train_label_dir, test_img_dir, test_label_dir


if __name__ == '__main__':
    # Example usage:
    transform = transforms.Compose([
        transforms.Resize((345, 640)),  # Or your model's expected input size
        transforms.ToTensor(),
    ])

    # Update IMG_DIR and LABEL_DIR to absolute paths or correct relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "Images")
    label_dir = os.path.join(current_dir, "Labels")

    print(f"Looking for images in: {img_dir}")
    print(f"Looking for labels in: {label_dir}")

    # Check if directories exist
    if not os.path.isdir(img_dir):
        print(f"Image directory not found: {img_dir}")
    if not os.path.isdir(label_dir):
        print(f"Label directory not found: {label_dir}")

    try:
        train_img_dir, train_label_dir, test_img_dir, test_label_dir = create_train_test_split(
            img_dir=img_dir, label_dir=label_dir, test_size=0.2, seed=42)

        train_dataset = ActionDataset(
            img_dir=train_img_dir, label_dir=train_label_dir, transform=transform)
        test_dataset = ActionDataset(
            img_dir=test_img_dir, label_dir=test_label_dir, transform=transform)

        if len(train_dataset) > 0 and len(test_dataset) > 0:
            print(f"Train dataset loaded with {len(train_dataset)} samples.")
            print(f"Test dataset loaded with {len(test_dataset)} samples.")
            img, label = train_dataset[0]
            print("Sample train image shape:", img.shape)
            print("Sample train label:", label)

        else:
            print("Train or test dataset is empty. Check paths and data.")

    except Exception as e:
        print(f"Error initializing or using dataset: {e}")
        import traceback
        traceback.print_exc()

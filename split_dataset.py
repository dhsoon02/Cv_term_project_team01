#!/usr/bin/env python3
"""
Script to split a dataset of images and LabelMe JSON files into training and test sets.
Uses an 80/20 split by default.
"""

import os
import argparse
from dataset import create_train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into training and test sets (80/20 split by default)"
    )
    parser.add_argument(
        "--source-img-dir",
        type=str,
        default="Images",
        help="Source directory containing all images"
    )
    parser.add_argument(
        "--source-label-dir",
        type=str,
        default="Labels",
        help="Source directory containing all label files"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing files in target directories"
    )

    args = parser.parse_args()

    # Check if source directories exist
    if not os.path.isdir(args.source_img_dir):
        print(
            f"Error: Source image directory not found at {args.source_img_dir}")
        return

    if not os.path.isdir(args.source_label_dir):
        print(
            f"Error: Source label directory not found at {args.source_label_dir}")
        return

    # Run the split function
    train_img_dir, train_label_dir, test_img_dir, test_label_dir = create_train_test_split(
        img_dir=args.source_img_dir,
        label_dir=args.source_label_dir,
        test_size=args.test_size,
        seed=args.seed
    )

    print("\nData split summary:")
    print(
        f"- Training: {len(os.listdir(train_img_dir))} images with {len(os.listdir(train_label_dir))} labels")
    print(
        f"- Testing:  {len(os.listdir(test_img_dir))} images with {len(os.listdir(test_label_dir))} labels")
    print("\nSplit complete!")
    print(f"Train images: {train_img_dir}")
    print(f"Train labels: {train_label_dir}")
    print(f"Test images:  {test_img_dir}")
    print(f"Test labels:  {test_label_dir}")


if __name__ == "__main__":
    main()

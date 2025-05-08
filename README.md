# Action Classification CNN

A simple computer vision project for classifying human actions in images using PyTorch and Convolutional Neural Networks. This project provides functionality for training, evaluating, and visualizing action classification models with built-in checkpointing and Weights & Biases integration.

## Features

- Simple CNN architecture for image classification
- Automatic 80/20 train/test data splitting
- Training with visualization through Weights & Biases
- Checkpoint saving and loading system
- Detailed evaluation metrics and visualizations
- Support for LabelMe JSON format annotations

## Installation


   ```bash
   git clone https://github.com/Topasm/Computer_vision.git
   cd Computer_vision
   ```


## Usage

### 1. Split your dataset into train and test sets

```bash
python split_dataset.py --source-img-dir /path/to/images --source-label-dir /path/to/labels
```

This will create `CV_Train` and `CV_Test` directories with an 80/20 split.

### 2. Train the model

```bash
python train.py
```

This will:
- Create an 80/20 split if not already done
- Train the CNN model
- Save checkpoints to the `checkpoints/` directory
- Log training metrics to Weights & Biases

For the first time using Weights & Biases, you'll need to authenticate:
```bash
wandb login
```

### 3. Evaluate the model

```bash
# Use the default model path
python evaluation.py

# Use the best model from training
python evaluation.py --best

# List all available checkpoints
python evaluation.py --list-checkpoints

# Use a specific checkpoint
python evaluation.py --model checkpoints/model_epoch_25.pth
```

### 4. Explore the results

After evaluation, you can find:
- Text results in `test_outputs/results.txt`
- Visualizations in `test_outputs/visualizations/`
- Confusion matrix in `test_outputs/confusion_matrix.png`


You can modify the architecture in `models.py`.

### Using a different model architecture

Modify `models.py` to implement a different architecture. The `ActionModel` class should implement the PyTorch `nn.Module` interface.

### Adding data augmentation

Add transformations to the `transform` variable in `train.py` to include data augmentation techniques.


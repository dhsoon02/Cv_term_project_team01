import torch
import torch.nn as nn
import torchvision.models as models
import time


class ActionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ActionModel, self).__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNetActionModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, resnet_type=18):
        """
        Initialize ResNet model for action classification

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            resnet_type: ResNet architecture (18, 34, 50, 101, 152)
        """
        super(ResNetActionModel, self).__init__()

        # Select ResNet architecture
        if resnet_type == 18:
            base_model = models.resnet18(pretrained=pretrained)
        elif resnet_type == 34:
            base_model = models.resnet34(pretrained=pretrained)
        elif resnet_type == 50:
            base_model = models.resnet50(pretrained=pretrained)
        elif resnet_type == 101:
            base_model = models.resnet101(pretrained=pretrained)
        elif resnet_type == 152:
            base_model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])

        # Get the number of features from the last layer
        if resnet_type in [18, 34]:
            num_features = 512
        else:
            num_features = 2048

        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )

        # For inference speed measurement
        self.last_inference_time = 0

    def forward(self, x):
        start_time = time.time()
        x = self.features(x)
        x = self.classifier(x)
        self.last_inference_time = (time.time() - start_time) * 1000  # ms
        return x

    def get_inference_time(self):
        """Returns the last inference time in milliseconds"""
        return self.last_inference_time

"""
Helmet Detection WITHOUT Pretrained COCO Weights - With CBAM
=============================================================
This script trains a YOLOv8n model with CBAM (Convolutional Block Attention Module)
from SCRATCH (no pretrained weights) for helmet detection.
"""

import torch
import torch.nn as nn

# ============================================================
# STEP 1: Define CBAM Module
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Ensure reduced channels is at least 1
        reduced_channels = max(channels // reduction, 1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(
            self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module with lazy initialization.
    
    This version uses lazy initialization to determine channels from actual input,
    avoiding issues with width_multiple scaling.
    """
    def __init__(self, c1, *args):  # c1 = channel arg from YAML (may not match actual)
        super().__init__()
        self._channels = None
        self.ca = None
        self.sa = SpatialAttention()

    def _lazy_init(self, channels):
        """Initialize channel attention when we know actual input channels."""
        self.ca = ChannelAttention(channels)
        # Move to same device as spatial attention
        device = next(self.sa.parameters()).device
        self.ca = self.ca.to(device)
        self._channels = channels

    def forward(self, x):
        # Lazy init on first forward pass
        if self.ca is None or self._channels != x.shape[1]:
            self._lazy_init(x.shape[1])
        x = self.ca(x)
        return x * self.sa(x)


# ============================================================
# STEP 2: Register CBAM in Ultralytics' tasks.py globals
# This is the KEY fix - must be done BEFORE importing YOLO
# ============================================================

import ultralytics.nn.tasks as tasks
tasks.CBAM = CBAM  # Inject CBAM into tasks module's namespace (where globals() looks)

# Also register in modules for full compatibility
import ultralytics.nn.modules as modules
modules.CBAM = CBAM

print("✓ CBAM successfully registered in Ultralytics")


# ============================================================
# STEP 3: Load model FROM SCRATCH and train
# ============================================================

from ultralytics import YOLO

# Verify CBAM is registered
print("CBAM in tasks:", hasattr(tasks, "CBAM"))
print("CBAM in modules:", hasattr(modules, "CBAM"))

# Load the custom CBAM model architecture FROM SCRATCH (no pretrained weights)
# Using .yaml file directly means training from scratch with random initialization
model = YOLO("yolov8n_cbam.yaml")

print("✓ Model loaded successfully (training from scratch - no pretrained weights)!")

# Train the model from scratch
model.train(
    data=r"C:/Users/soumy/OneDrive/Desktop/Helmet Detection/Helmet-Detection/Helmet-Detection/dataset/data1.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    amp=True,
    cos_lr=True,
    close_mosaic=10,
    pretrained=False,  # Explicitly disable pretrained weights
    name="helmet_yolov8n_cbam_scratch"
)

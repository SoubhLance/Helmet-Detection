# Helmet Detection

A deep learning project for detecting helmets using YOLOv8 with various model configurations and CBAM (Convolutional Block Attention Module) integration.

## Overview

This project implements helmet detection using four different model configurations to compare the effectiveness of COCO pre-trained weights and CBAM attention mechanisms:

1. **COCO Pre-trained + CBAM**: YOLOv8 with COCO weights and CBAM attention
2. **COCO Pre-trained without CBAM**: YOLOv8 with COCO weights only
3. **No Pre-training + CBAM**: YOLOv8 trained from scratch with CBAM attention
4. **No Pre-training without CBAM**: YOLOv8 trained from scratch

## Features

- Multiple model configurations for performance comparison
- CBAM attention mechanism integration for improved feature extraction
- Transfer learning with COCO pre-trained weights
- Training from scratch option for specialized helmet detection
- Comprehensive evaluation metrics

## Project Structure

```
Helmet-Detection/
├── dataset/
│   ├── annotations/
│   ├── images/
│   └── labels/
├── runs/
│   └── detect/
├── data.yaml
├── model.ipynb
├── modelcbam.ipynb
├── scratchmodel.ipynb
├── scratchmodelcbam.ipynb
├── train_cbam_scratch.py
├── yolo11n.pt
├── yolov8n_cbam.yaml
└── .gitignore
```

## Requirements

```bash
pip install ultralytics torch torchvision
```

### Key Dependencies
- Python 3.12+
- PyTorch 2.5.1
- CUDA 13 (for GPU support)
- Ultralytics 8.3.235
- NVIDIA GeForce RTX 3050 Laptop GPU support (For Without CoCo Predtrained Weights)
- NVIDIA GeForce RTX 4060 Laptop GPU support (For With CoCo Predtrained Weights)

## Model Configurations

### 1. COCO Pre-trained with CBAM
- **File**: `modelcbam.ipynb`
- Uses COCO pre-trained weights
- Integrates CBAM attention modules
- Best for: Transfer learning with enhanced attention

### 2. COCO Pre-trained without CBAM
- **File**: `model.ipynb`
- Uses standard COCO pre-trained weights
- Baseline transfer learning approach
- Best for: Quick deployment with proven weights

### 3. Scratch Training with CBAM
- **File**: `scratchmodelcbam.ipynb`
- Trained from random initialization
- Includes CBAM attention modules
- Best for: Custom feature learning with attention

### 4. Scratch Training without CBAM
- **File**: `scratchmodel.ipynb`
- Trained from random initialization
- Standard architecture
- Best for: Pure custom training baseline

## Training Configuration

The models are trained with the following parameters:
- **Epochs**: 100
- **Batch size**: 16
- **Image size**: 640x640
- **Optimizer**: SGD
- **Device**: CUDA (GPU)
- **Cache**: Enabled for faster training
- **CFG**: Default YOLOv8 configuration

## Dataset

The project uses a custom helmet detection dataset with:
- Training images in `dataset/images/`
- Annotations in `dataset/annotations/`
- Labels in `dataset/labels/`
- Configuration in `data.yaml`

## Usage

### Training

#### Train with COCO Pre-trained + CBAM:
```python
# Run modelcbam.ipynb
from ultralytics import YOLO

model = YOLO('yolov8n_cbam.yaml')
model.load('yolov8n.pt')
results = model.train(data='data.yaml', epochs=100, batch=16, cache=True)
```

#### Train from Scratch + CBAM:
```python
# Run scratchmodelcbam.ipynb or train_cbam_scratch.py
python train_cbam_scratch.py
```

#### Train with Standard Configurations:
```python
# For COCO pre-trained: model.ipynb
# For scratch training: scratchmodel.ipynb
```

### Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('path/to/best.pt')

# Run inference
results = model.predict('path/to/image.jpg', save=True)
```

## CBAM Integration

CBAM (Convolutional Block Attention Module) enhances the model's ability to focus on relevant features through:
- **Channel Attention**: Emphasizes important feature channels
- **Spatial Attention**: Highlights important spatial locations
- Improved feature representation for helmet detection

## Evaluation Metrics

Models are evaluated using:
- **GPU Memory Usage**: GPU_mem (tracked during training)
- **Box Loss**: Bounding box regression loss
- **Class Loss**: Classification loss
- **DFL Loss**: Distribution Focal Loss
- **Instances**: Number of detected instances
- **Size**: Model size in MB

## Results

Training progress is logged with metrics including:
- Epoch-wise loss values
- GPU memory utilization
- Instance detection counts
- Model convergence tracking

Results are saved in `runs/detect/` directory.

## Contributors

- **SoubhLance** (Soubhik Sadhu)
- **Soumyajit-here** (Soumyajit Das)

## Languages

- Python
- Deep Learning
- YOLOv8
- CNN

## Suggested Workflows

To enhance your project, consider setting up:
- Continuous Integration (CI) workflows
- Automated testing pipelines
- Model deployment workflows

## License

Please add an appropriate license for your project.

## Acknowledgments

- Ultralytics YOLOv8 framework
- COCO dataset for pre-trained weights
- CBAM attention mechanism research

## Future Work

- Add comprehensive README with results comparison
- Implement model performance benchmarking
- Create deployment pipeline
- Add visualization tools for detection results
- Expand dataset with more diverse scenarios

## Contact

For questions or contributions, please open an issue or contact the contributors.

---

**Note**: This project is under active development. Make sure to check the latest commits for updates.

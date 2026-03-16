# Bubble Detection Project

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Approaches Explored](#approaches-explored)
  - [1. OpenCV Background Removal](#1-opencv-background-removal)
  - [2. SAM2 Object Segmentation + Tracking](#2-sam2-object-segmentation--tracking)
  - [3. YOLO World (Zero-Shot Tracker)](#3-yolo-world-zero-shot-tracker)
  - [4. SAM2 + Cotracker](#4-sam2--cotracker)
  - [5. Hybrid SAM2 + YOLO](#5-hybrid-sam2--yolo)
  - [6. FastSAM + YOLO](#6-fastsam--yolo-recommended)
- [Performance Comparison](#performance-comparison)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project focuses on developing an efficient and accurate system for detecting and tracking bubbles in video footage. Bubble detection presents unique challenges due to the transparent nature of bubbles, their dynamic behavior (merging, breaking, deformation), and the need for real-time or near-real-time processing capabilities.

The project explores multiple computer vision and deep learning approaches, ranging from traditional OpenCV methods to state-of-the-art segmentation models like SAM2 (Segment Anything Model 2) and object detection models like YOLO (You Only Look Once). Through iterative experimentation, we've identified optimal solutions that balance accuracy, speed, and resource requirements.

## Problem Statement

Detecting and tracking bubbles in video presents several unique challenges:

- **Transparency**: Bubbles are transparent objects, making traditional background subtraction and edge detection methods ineffective
- **Dynamic Behavior**: Bubbles constantly merge, split, and deform, requiring robust tracking mechanisms
- **Shape Variation**: Bubble shapes vary significantly based on size, pressure, and environmental conditions
- **Real-time Requirements**: For practical applications, processing speed is crucial
- **Resource Constraints**: Many deployment scenarios require solutions that can run on limited hardware

## Approaches Explored

### 1. OpenCV Background Removal

**Description**: Initial attempt using traditional computer vision techniques with OpenCV's background subtraction methods.

**Method**: Applied various background removal algorithms including:
- MOG2 (Mixture of Gaussians)
- KNN background subtractor
- Contour detection with edge filtering

**Results**: ❌ **Not Successful**

**Reason for Failure**: The fundamental transparency of bubbles makes them nearly invisible to background subtraction algorithms. These methods rely on detecting solid objects with distinct foreground-background separation, which doesn't apply to transparent or semi-transparent objects like bubbles.

**Key Learnings**:
- Traditional computer vision methods are insufficient for transparent object detection
- Need for deep learning-based approaches that can learn complex features
- Background subtraction works well for opaque objects but fails for transparent ones

---

### 2. SAM2 Object Segmentation + Tracking

**Description**: Implementation using Meta's Segment Anything Model 2 (SAM2) for object segmentation and tracking.

**Method**: 
- Used SAM2's automatic mask generation capabilities
- Applied temporal tracking across video frames
- Extracted bubble boundaries and shapes

**Notebook**: [SAM2 Implementation](https://colab.research.google.com/drive/1csvWGFAQG55H8PKMAyk64CFni8giuwFD?usp=sharing)

**Results**: ✅ **Successful** - Achieved ~85% bubble detection accuracy

#### Advantages:
- **High Accuracy**: Successfully detected approximately 85% of bubbles in test videos
- **Shape Detection**: Captures precise bubble boundaries, enabling surface area estimation
- **Robust Segmentation**: Handles various bubble sizes and shapes effectively
- **No Training Required**: Zero-shot capability works out of the box

#### Disadvantages:
- **Computationally Expensive**: Extremely resource-intensive
- **Very Slow Processing**: Took approximately **40 minutes to process a 15-second video** on Colab T4 GPU
- **Not Suitable for Real-time**: Processing speed makes it impractical for live applications
- **High Memory Usage**: Requires significant GPU memory

#### Video Comparison:

<table>
<tr>
<td width="50%">

**Original Video**
<!-- Insert original video here -->
```
[Original Video Placeholder]
```

</td>
<td width="50%">

**SAM2 Output**
<!-- Insert SAM2 processed video here -->
```
[SAM2 Output Video Placeholder]
```

</td>
</tr>
</table>

**Processing Stats**:
- Video Duration: 15 seconds
- Processing Time: ~40 minutes
- Hardware: Colab T4 GPU
- Frame Rate: ~0.006 FPS (processing)

---

### 3. YOLO World (Zero-Shot Tracker)

**Description**: Experimentation with YOLO World, a zero-shot object detection model that can detect objects without specific training.

**Method**:
- Applied YOLO World's zero-shot detection capabilities
- Attempted to detect bubbles using text prompts and generic object detection

**Results**: ❌ **Not Successful**

**Reason for Failure**: 
- The model failed to generalize to transparent bubbles
- YOLO World is optimized for solid, opaque objects with clear boundaries
- Lack of transparency understanding in the pre-trained model
- Text-based prompting couldn't adequately describe transparent bubble characteristics

**Key Learnings**:
- Zero-shot models trained on standard datasets struggle with transparent objects
- Specialized training or fine-tuning required for non-standard object types
- Text prompts alone insufficient for complex visual properties like transparency

---

### 4. SAM2 + Cotracker

**Description**: Attempted combination of SAM2's segmentation capabilities with Cotracker's point-based tracking.

**Method**:
- Used SAM2 for initial bubble detection
- Applied Cotracker for temporal tracking of detected bubbles

**Results**: ❌ **Not Suitable for This Application**

**Reason for Failure**:
- **Point-based Tracking Limitation**: Cotracker tracks individual points, not object boundaries
- **Cannot Handle Merging**: When bubbles merge, point-based tracking fails to recognize the merger
- **Cannot Handle Splitting**: When bubbles split, the tracker cannot properly assign new identities
- **Loss of Shape Information**: Point tracking doesn't preserve bubble boundary information

**Key Learnings**:
- Point-based tracking insufficient for objects with dynamic topology changes
- Need for instance-level tracking that handles object lifecycle events (creation, merging, splitting)
- Shape-aware tracking essential for bubble applications

---

### 5. Hybrid SAM2 + YOLO

**Description**: A hybrid approach combining SAM2's segmentation quality with YOLO's detection speed through transfer learning.

**Method**:
1. **Annotation Phase**: Use SAM2 to automatically annotate frames
2. **Training Phase**: Train YOLOv8/YOLOv11 on SAM2-annotated frames
3. **Detection Phase**: Use trained YOLO model for bubble detection and tracking

**Notebook**: [Hybrid SAM2 + YOLO Implementation](https://colab.research.google.com/drive/1yO9Rp5F02GJc1NxWgZ1uQPSume_zwMmC?usp=sharing)

**Results**: ✅ **Successful** - Good accuracy with improved speed

#### Advantages:
- **Faster than SAM2**: Significantly reduced processing time
- **Good Accuracy**: Maintains high detection quality
- **Leverages SAM2's Quality**: Benefits from SAM2's segmentation precision during training
- **YOLO's Speed**: Achieves real-time or near-real-time performance during inference

#### Limitations:
- **Training Data Requirements**: Requires **50+ annotated frames** for good performance
- **Environment-Specific**: Model doesn't generalize well to new environments
- **Failed Cross-Video Generalization**: Models trained on one video failed on different videos
- **Retraining Needed**: Each new environment requires retraining

#### Opportunity:
This approach can be deployed as an **environment-specific model** with an initialization phase:
- Before tracking bubbles in a new environment, spend initialization time training YOLO
- Works exceptionally well within the environment it's trained on
- Suitable for fixed-camera, controlled-environment applications

**Workflow**:
```
New Environment → SAM2 Annotation (50 frames) → YOLO Training → Deployment
```

#### Video Comparison:

<table>
<tr>
<td width="50%">

**Original Video**
<!-- Insert original video here -->
```
[Original Video Placeholder]
```

</td>
<td width="50%">

**Hybrid SAM2 + YOLO Output**
<!-- Insert processed video here -->
```
[Hybrid Output Video Placeholder]
```

</td>
</tr>
</table>

**Processing Stats**:
- Training Frames Required: 50
- Training Time: Variable (depends on GPU)
- Inference Speed: Real-time capable (30+ FPS)
- Generalization: Environment-specific only

---

### 6. FastSAM + YOLO (Recommended)

**Description**: Optimized pipeline replacing SAM2 with FastSAM for faster annotation while maintaining YOLO's tracking efficiency.

**Method**:
1. **Annotation Phase**: Use FastSAM for rapid object detection and annotation
2. **Training Phase**: Train YOLO on FastSAM-annotated frames (50 frames)
3. **Tracking Phase**: Deploy trained YOLO model for bubble tracking

**Notebook**: [FastSAM + YOLO Implementation](https://colab.research.google.com/drive/1_52Pdg0-7wtDDRclFezw4aVfo8iKczEw?usp=sharing)

**Results**: ✅ **Highly Successful** - Best speed-accuracy tradeoff

#### Key Observations:
- **FastSAM Segmentation**: Good quality object detection, faster than SAM2
- **Tracker Quality**: FastSAM's built-in tracker not as robust as SAM2
- **Solution**: Hybrid approach - FastSAM for annotation + YOLO for tracking

#### Performance:
- **Achieves 90% of Approach 5's accuracy**
- **Uses only 20% of computation time and resources**
- **Best speed-accuracy tradeoff** among all approaches

#### Advantages:
- ✅ **Significantly Faster**: ~5x faster than SAM2 + YOLO approach
- ✅ **Efficient Annotation**: Quick frame annotation process
- ✅ **Good Accuracy**: Minimal accuracy loss compared to SAM2
- ✅ **Resource Efficient**: Lower GPU memory requirements
- ✅ **Tunable**: Performance can be optimized through hyperparameter tuning

#### Workflow:
```
FastSAM Annotation (50 frames) → YOLO Training → Fast Inference
                ↓
         ~10-15 minutes
```

#### Hyperparameter Optimization:
With proper tuning of:
- YOLO training epochs
- Confidence thresholds
- NMS (Non-Maximum Suppression) parameters
- Augmentation strategies

The model can achieve near-SAM2 quality at a fraction of the cost.

#### Video Comparison:

<table>
<tr>
<td width="50%">

**Original Video**
<!-- Insert original video here -->
```
[Original Video Placeholder]
```

</td>
<td width="50%">

**FastSAM + YOLO Output**
<!-- Insert processed video here -->
```
[FastSAM Output Video Placeholder]
```

</td>
</tr>
</table>

**Processing Stats**:
- Annotation Time: ~10-15 minutes (50 frames)
- Training Time: ~5-10 minutes
- Total Setup Time: ~20-25 minutes
- Inference Speed: Real-time (30+ FPS)
- Accuracy: ~90% relative to SAM2 baseline

---

## Performance Comparison

| Approach | Accuracy | Speed | Resources | Generalization | Recommended |
|----------|----------|-------|-----------|----------------|-------------|
| OpenCV Background Removal | ❌ Failed | Fast | Low | N/A | ❌ |
| SAM2 + Tracking | ⭐⭐⭐⭐⭐ (85%) | ❌ Very Slow (40 min/15s) | Very High | ✅ Good | ❌ |
| YOLO World | ❌ Failed | Fast | Medium | ❌ Poor | ❌ |
| SAM2 + Cotracker | ❌ Unsuitable | Slow | High | N/A | ❌ |
| SAM2 + YOLO | ⭐⭐⭐⭐ | ⭐⭐⭐ Fast | Medium | ❌ Environment-specific | ⚠️ |
| **FastSAM + YOLO** | ⭐⭐⭐⭐ (90% of SAM2) | ⭐⭐⭐⭐⭐ Very Fast | Low | ⚠️ Environment-specific | ✅ **Best** |

### Recommended Approach

**FastSAM + YOLO** is the recommended solution because it offers:
- Excellent speed-accuracy tradeoff
- Practical processing times for real-world applications
- Resource efficiency suitable for various hardware configurations
- High accuracy (90% of SAM2 baseline) at 20% of the computational cost

### Use Cases

**Use SAM2 + Tracking when**:
- Absolute maximum accuracy is required
- Processing time is not a constraint
- High-end GPU resources are available
- Detailed shape analysis is critical

**Use FastSAM + YOLO when**:
- Fast processing is required
- Good accuracy is sufficient (not absolute maximum)
- Limited GPU resources
- Multiple videos need processing
- Near real-time performance desired

**Use SAM2 + YOLO when**:
- Slightly better accuracy than FastSAM needed
- Can afford longer initialization time
- Fixed environment with controlled conditions

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- Google Colab account (for notebooks)

### Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install ultralytics  # For YOLO
pip install opencv-python
pip install numpy pandas matplotlib

# For SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# For FastSAM
pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git

# Additional utilities
pip install tqdm
pip install scikit-learn
```

## Usage

### Quick Start with FastSAM + YOLO (Recommended)

```python
# 1. Annotate frames using FastSAM
from fastsam import FastSAM
from ultralytics import YOLO

# Initialize FastSAM
fastsam_model = FastSAM('FastSAM-x.pt')

# Annotate 50 frames
annotations = annotate_frames_with_fastsam(
    video_path='input_video.mp4',
    num_frames=50,
    model=fastsam_model
)

# 2. Train YOLO on annotated frames
yolo_model = YOLO('yolov8n.pt')
yolo_model.train(
    data='bubble_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# 3. Run inference on full video
results = yolo_model.predict(
    source='input_video.mp4',
    save=True,
    conf=0.25
)
```

### Detailed Workflow

#### 1. Frame Extraction
```python
import cv2

def extract_frames(video_path, num_frames=50):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames
```

#### 2. FastSAM Annotation
```python
def annotate_with_fastsam(frames, model):
    annotations = []
    for frame in frames:
        results = model(
            frame,
            device='cuda',
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9
        )
        annotations.append(results)
    return annotations
```

#### 3. YOLO Training
```python
# Create dataset in YOLO format
# dataset.yaml:
# train: path/to/train/images
# val: path/to/val/images
# nc: 1  # number of classes (bubble)
# names: ['bubble']

model = YOLO('yolov8n.pt')
model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=50,
    save=True,
    device='cuda'
)
```

#### 4. Inference
```python
# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Process video
results = model.predict(
    source='test_video.mp4',
    save=True,
    conf=0.25,
    iou=0.45,
    show_labels=True,
    show_conf=True
)
```

## Future Work

### Immediate Improvements
1. **Automated Hyperparameter Tuning**: Implement grid search or Bayesian optimization for YOLO training parameters
2. **Active Learning**: Develop system to identify and annotate challenging frames automatically
3. **Multi-Environment Training**: Create a diverse dataset spanning multiple environments to improve generalization
4. **Real-time Processing**: Optimize pipeline for true real-time performance on edge devices

### Advanced Features
1. **Bubble Size Estimation**: Implement surface area calculation using detected boundaries
2. **Trajectory Analysis**: Track individual bubble paths and analyze movement patterns
3. **Merger/Split Detection**: Explicitly detect and track bubble merging and splitting events
4. **Density Mapping**: Create heat maps showing bubble density across frame regions

### Model Enhancements
1. **Transfer Learning**: Experiment with domain adaptation techniques for better generalization
2. **Ensemble Methods**: Combine multiple detection methods for improved accuracy
3. **Attention Mechanisms**: Integrate attention layers to focus on bubble-specific features
4. **Synthetic Data Generation**: Create synthetic bubble datasets for data augmentation

### Production Deployment
1. **Model Optimization**: Quantization and pruning for edge device deployment
2. **API Development**: REST API for bubble detection as a service
3. **Web Interface**: User-friendly interface for video upload and analysis
4. **Batch Processing**: System for processing multiple videos in parallel

### Research Directions
1. **3D Bubble Reconstruction**: Estimate 3D bubble shapes from 2D video
2. **Physics-Informed Models**: Incorporate fluid dynamics constraints into detection
3. **Temporal Consistency**: Improve frame-to-frame consistency using temporal models
4. **Anomaly Detection**: Identify unusual bubble behavior automatically

## Contributing

We welcome contributions! Here's how you can help:

1. **Report Issues**: Found a bug or have a suggestion? Open an issue
2. **Improve Documentation**: Help us make this README and code more understandable
3. **Optimize Code**: Submit PRs for performance improvements
4. **Add Features**: Implement items from the Future Work section
5. **Share Results**: Share your bubble detection results and use cases

### Development Setup
```bash
git clone https://github.com/yourusername/bubble-detection.git
cd bubble-detection
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta AI for SAM2 (Segment Anything Model 2)
- Ultralytics for YOLO implementation
- CASIA-IVA-Lab for FastSAM
- Google Colab for providing GPU resources

## Contact

For questions, suggestions, or collaborations, please open an issue or contact [your email].

---

**Last Updated**: March 2026

**Project Status**: Active Development

**Version**: 1.0.0

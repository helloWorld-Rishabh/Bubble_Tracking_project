# 🫧 Bubble Detection and Tracking in Fluid Dynamics

## 📖 Introduction
Welcome to the Bubble Detection and Tracking project. Tracking chemical micro-bubbles in fluid environments is a notoriously difficult computer vision problem. Bubbles are highly transparent, constantly change shape, merge, break apart, and often blend into the background fluid. 

This repository documents our iterative approach to solving this problem. We experimented with a wide range of computer vision techniques—from classical pixel-math to state-of-the-art foundation models—to find the perfect balance between tracking accuracy, bounding-box/mask fidelity, and computational efficiency. 

Below is a chronological breakdown of the different methodologies we tested, along with their results.

---

## 🧪 Approaches & Experiments

### 1️⃣ OpenCV Background Removal (Classical CV)
Our initial approach relied on traditional computer vision techniques, specifically background subtraction and contour detection using OpenCV.
* **Result:** ❌ Unsuccessful.
* **Why it failed:** Because bubbles are essentially transparent, they do not have a solid pixel intensity that classical background removal can easily isolate. Glare, shadows, and the fluid itself caused too many false positives and lost tracks.

### 2️⃣ SAM 2 Object Segmentation + Tracking
We moved to Meta's Segment Anything Model 2 (SAM 2) to leverage its zero-shot video tracking and powerful masking capabilities.
* **Result:** ✅ Highly accurate (~85% detection rate).
* **Advantages:** Excellent accuracy. Because SAM 2 outputs pixel-perfect segmentation masks instead of just bounding boxes, we could capture the exact boundary and shape of the bubbles. This is highly valuable for estimating surface area and volume dynamics.
* **Disadvantages:** It is extremely resource-intensive and slow. Processing just a 15-second video took approximately 40 minutes on a Colab T4 GPU.
* **Notebook:** [View SAM 2 Implementation Here](https://colab.research.google.com/drive/1csvWGFAQG55H8PKMAyk64CFni8giuwFD?usp=sharing)

#### Results: SAM 2 Tracking
<div align="center">
  <video src="PATH_TO_YOUR_ORIGINAL_VIDEO.mp4" width="400" autoplay loop muted></video>
  <video src="PATH_TO_YOUR_OUTPUT_VIDEO.mp4" width="400" autoplay loop muted></video>
  <p><i>Left: Original Video | Right: SAM 2 Output</i></p>
</div>

### 3️⃣ YOLO-World (Zero-Shot Detection)
To solve the speed bottleneck of SAM 2, we tested YOLO-World, a real-time, zero-shot object detection model designed to find objects based on text prompts.
* **Result:** ❌ Unsuccessful.
* **Why it failed:** While extremely fast, the model failed to generalize to the concept of "transparent bubbles" in our specific fluid environment, leading to poor detection rates.

### 4️⃣ SAM 2 + CoTracker
We attempted to combine SAM 2's masking with CoTracker (a point-tracking model) to track specific points on the bubbles across time.
* **Result:** ❌ Unsuccessful.
* **Why it failed:** CoTracker is a point-based tracker. When bubbles naturally merge together or break apart into smaller micro-bubbles, the tracked points lose their geometric meaning, causing the tracking logic to fail entirely.

### 5️⃣ Hybrid Approach: SAM 2 Auto-Annotation + YOLO Training
Realizing that YOLO is incredibly fast but needs specific data to understand transparent bubbles, we built a hybrid pipeline. We used the highly accurate SAM 2 to automatically annotate ~50 diverse frames from our video, and then trained a custom YOLO model on those specific frames.
* **Result:** ✅ Very effective and significantly faster than pure SAM 2 tracking.
* **Limitations:** The custom YOLO model overfits to the specific lighting and fluid setup of the training video; it fails to generalize if applied instantly to a completely new video environment.
* **Opportunity:** This pipeline can be deployed as an "Environment-Specific Initialized Model." Before tracking a new experiment, the system takes a brief initialization period to auto-annotate 50 frames and train a localized YOLO model, achieving high-speed, high-accuracy tracking for the remainder of that specific environment.
* **Notebook:** [View SAM2 + YOLO Implementation Here](https://colab.research.google.com/drive/1yO9Rp5F02GJc1NxWgZ1uQPSume_zwMmC?usp=sharing)

#### Results: SAM 2 + YOLO Tracking
<div align="center">
  <video src="PATH_TO_YOUR_ORIGINAL_VIDEO.mp4" width="400" autoplay loop muted></video>
  <video src="PATH_TO_YOUR_OUTPUT_VIDEO.mp4" width="400" autoplay loop muted></video>
  <p><i>Left: Original Video | Right: SAM2 + YOLO Output</i></p>
</div>

### 6️⃣ Optimized Hybrid: FastSAM + YOLO (Final Pipeline)
To eliminate the massive computational bottleneck of generating the training data with SAM 2, we swapped it out for **FastSAM**. FastSAM provides similar zero-shot segmentation but is roughly 10x faster and requires lower memory. 
* **The Approach:** We use FastSAM to auto-annotate 50 frames of a new video. We train YOLO on those frames, and then use YOLO to track the bubbles for the rest of the video.
* **Result:** 🏆 **Best Overall Performance.** With properly tuned hyperparameters, this pipeline achieves 90% of the accuracy of Approach 5, but operates in just **20% of the computation time**. 
* **Notebook:** [View FastSAM + YOLO Implementation Here](https://colab.research.google.com/drive/1_52Pdg0-7wtDDRclFezw4aVfo8iKczEw?usp=sharing)

#### Results: FastSAM + YOLO Tracking
<div align="center">
  <video src="PATH_TO_YOUR_ORIGINAL_VIDEO.mp4" width="400" autoplay loop muted></video>
  <video src="PATH_TO_YOUR_OUTPUT_VIDEO.mp4" width="400" autoplay loop muted></video>
  <p><i>Left: Original Video | Right: FastSAM + YOLO Output</i></p>
</div>

---

## 🚀 How to Use
*(Add instructions here on how a user can clone the repo, install dependencies, and run your final FastSAM + YOLO pipeline script)*

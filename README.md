# VisDrone Object Detection & Tracking

A complete end-to-end deep learning pipeline for aerial drone imagery — covering dataset preparation, YOLOv8 model training, object detection, multi-object tracking, and full evaluation using the [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) benchmark dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tasks](#tasks)
  - [Task 1 — Dataset Understanding & Preprocessing](#task-1--dataset-preparation--eda)
  - [Task 2 — Model Training](#task-2--yolov8-model-training)
  - [Task 3 — Human & Car Detection with Human Counting](#task-3--object-detection--counting)
  - [Task 4 — Object Tracking](#task-4--multi-object-tracking-mot)
  - [Task 5 — Evaluation & Visualization](#task-5--evaluation--visualization)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Output Files](#output-files)

---

## Project Overview

This project applies YOLOv8 to aerial drone footage from the VisDrone2019 dataset to solve two real-world computer vision tasks:

1. **Object Detection** — Detect humans and cars in drone images with bounding boxes and class labels.
2. **Multi-Object Tracking** — Assign persistent IDs to detected objects across frames using the ByteTrack algorithm.

The pipeline is structured across 5 sequential tasks, each building on the previous, and is implemented as a Google Colab notebook with persistent state management across sessions via Google Drive.

---

## Dataset

**Source:** [VisDrone2019-DET Dataset](https://github.com/VisDrone/VisDrone-Dataset) via KaggleHub (`banuprasadb/visdrone-dataset`)

The original dataset contains 10 object classes captured from drone cameras at various altitudes and environments:

| Class ID | Original Class     |
|----------|--------------------|
| 0        | pedestrian         |
| 1        | people             |
| 2        | bicycle            |
| 3        | car                |
| 4        | van                |
| 5        | truck              |
| 6        | tricycle           |
| 7        | awning-tricycle    |
| 8        | bus                |
| 9        | motor              |

### Class Remapping

For this project, the 10 classes are remapped into 2 simplified classes:

| New Class ID | New Class Name | Original Classes Merged              |
|--------------|----------------|--------------------------------------|
| 0            | human          | pedestrian (0), people (1)           |
| 1            | car            | car (3), van (4)                     |

### Dataset Split

| Split      | Images | Annotation Instances |
|------------|--------|----------------------|
| Train      | 6,468  | 276,219              |
| Validation | 548    | 30,008               |

---

## Project Structure

```
VisDrone-Project/
├── dataset.yaml                        # YOLO dataset configuration
├── processed_dataset/
│   ├── images/
│   │   ├── train/                      # 6,468 training images
│   │   └── val/                        # 548 validation images
│   └── labels/
│       ├── train/                      # Remapped YOLO labels (train)
│       └── val/                        # Remapped YOLO labels (val)
├── runs/
│   └── visdrone_yolov8/
│       ├── weights/
│       │   ├── best.pt                 # Best model checkpoint
│       │   └── last.pt                 # Last epoch checkpoint
│       ├── results.png                 # Training curves
│       ├── confusion_matrix.png
│       └── confusion_matrix_normalized.png
├── tracking_outputs/                   # Per-image tracking results (50 frames)
├── tracking_runs/
│   └── video_tracking/                 # Video tracking output
├── videos/
│   ├── demo.mp4                        # Input demo video
│   └── compressed_video.mp4            # Compressed (640×360) version
├── project_state.pkl                   # Task 1 state
├── task3_complete_state.pkl            # Task 3 state
├── task4_complete_state.pkl            # Task 4 state
└── task5_complete_state.pkl            # Task 5 state
```

---

## Tasks

### Task 1 — Dataset Understanding & Preprocessing

**Goal:** Download, explore, and preprocess the VisDrone dataset for training.

**Steps:**
- Downloaded VisDrone2019-DET dataset via KaggleHub
- Explored dataset structure: confirmed 6,471 matched image-label pairs in train and 548 in validation
- Visualized a raw sample drone image and its original 10-class annotations with color-coded bounding boxes
- Remapped the 10-class labels to 2 classes (human, car), filtering out all other classes
- Processed and saved filtered images and labels into a clean `processed_dataset/` directory
- Generated a `dataset.yaml` YOLO configuration file
- Verified remapped labels and visualized the processed dataset

**Key Output:** `processed_dataset/` with 6,468 train images and 548 val images, `dataset.yaml`

**Output Images:**

| File | Description |
|------|-------------|
| `Outputs/Task-1/sample-drone-image.png` | Raw drone image without annotations |
| `Outputs/Task-1/annotated-drone-image.png` | Sample image with all 10 original class bounding boxes |
| `Outputs/Task-1/processed-dataset-visualization.png` | Sample image with remapped 2-class annotations |

---

### Task 2 — Model Training

**Goal:** Fine-tune a YOLOv8n model on the processed VisDrone dataset.

**Model:** `yolov8n.pt` (YOLOv8 Nano — pretrained on COCO)

**Training Configuration:**

| Parameter     | Value        |
|---------------|--------------|
| Epochs        | 30           |
| Image Size    | 640×640      |
| Batch Size    | 16           |
| Device        | CUDA (Tesla T4) |
| Optimizer     | SGD (lr=0.01) |
| Pretrained    | Yes (COCO)   |
| AMP           | Enabled      |
| Cache         | Enabled      |

**Framework:** Ultralytics 8.4.51 — PyTorch 2.10.0 + CUDA 12.8

**Output Images:**

| File | Description |
|------|-------------|
| `Outputs/Task-2/training-result.png` | Training curves: loss, precision, recall, mAP over 30 epochs |

---

### Task 3 — Human & Car Detection with Human Counting 

**Goal:** Run inference with the trained model on validation images and count detected humans and cars.

**Steps:**
- Loaded `best.pt` weights from the training run
- Ran prediction on 5 sample validation images at `conf=0.25`
- Visualized predictions with bounding boxes and class labels
- Counted detected humans and cars per image and overlaid counts on the annotated image

**Sample Detection Result (Image: `0000086_01443_d_0000004.jpg`):**
- Humans detected: **118**
- Cars detected: **0**
- Inference speed: **71.8ms** per image

**Output Images:**

| File | Description |
|------|-------------|
| `Outputs/Task-3/prediction.png` | YOLOv8 detection output with bounding boxes |
| `Outputs/Task-3/detection+counting.png` | Annotated image with human and car count overlaid |

---

### Task 4 — Object Tracking

**Goal:** Apply multi-object tracking with persistent IDs across frames using YOLOv8 + ByteTrack.

**Tracker:** ByteTrack (`bytetrack.yaml`)

**Steps:**
- Ran `model.track()` with `persist=True` on the first 50 validation images (frame-by-frame tracking simulation)
- Visualized tracked objects with unique track IDs per object
- Applied tracking to a full demo video (`demo.mp4`), compressed to 640×360 for faster processing
- Saved annotated tracking output images and an output video

**Sample Tracking Output (Frame 1):**
- 28 objects tracked with persistent IDs (Track ID: 1–28)
- Classes tracked: humans and cars

**Configuration:**

| Parameter    | Value           |
|--------------|-----------------|
| Tracker      | ByteTrack       |
| Confidence   | 0.25            |
| IoU          | 0.50            |
| Image Size   | 640             |
| Video Stride | 2 (every 2nd frame) |

**Output Files:**

| File | Description |
|------|-------------|
| `Outputs/Task-4/image-tracking-result.png` | Sample frame with ByteTrack IDs and bounding boxes |
| `Outputs/Task-4/video-tracking-result.avi` | Full annotated tracking video output |

---

### Task 5 — Evaluation & Visualization

**Goal:** Comprehensive model evaluation with metrics, confusion matrix, prediction samples, and performance summary.

**Steps:**
- Re-ran validation on the full val split (548 images, 30,008 instances)
- Computed overall and per-class metrics: Precision, Recall, mAP50, mAP50-95
- Displayed confusion matrix and normalized confusion matrix
- Ran predictions on 5 random validation images and visualized outputs
- Counted total detections across the sample set
- Measured inference speed (FPS)
- Generated a bar chart summarizing model performance

**Output Images:**

| File | Description |
|------|-------------|
| `Outputs/Task-5/confusion_matrix.png` | Raw confusion matrix across all classes |
| `Outputs/Task-5/confusion_matrix_normalized.png` | Normalized confusion matrix |
| `Outputs/Task-5/prediction-output.png` | Sample prediction visualizations |
| `Outputs/Task-5/model-performance-summary.png` | Bar chart: Precision, Recall, mAP50 |

---

## Results

### Overall Validation Metrics

| Metric      | Value  |
|-------------|--------|
| Precision   | 0.7051 |
| Recall      | 0.5295 |
| mAP@50      | 0.5791 |
| mAP@50-95   | 0.3194 |

### Per-Class Performance

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|--------|-----------|-----------|--------|--------|-----------|
| human | 531    | 13,969    | 0.638     | 0.347  | 0.397  | 0.147     |
| car   | 517    | 16,039    | 0.772     | 0.712  | 0.761  | 0.492     |

### Inference Speed

| Stage        | Time per Image |
|--------------|----------------|
| Preprocess   | ~1.0 ms        |
| Inference    | ~3.0 ms        |
| Postprocess  | ~2.6 ms        |
| **FPS**      | **~34.16**     |

### Detection Counts (5-Image Sample)

| Class  | Count |
|--------|-------|
| Humans | 142   |
| Cars   | 137   |

---

## Requirements

```
ultralytics>=8.4.0
opencv-python>=4.6.0
matplotlib
seaborn
tqdm
numpy
scipy
torch>=2.0.0
kagglehub
supervision
lap
filterpy
```

Install all dependencies:

```bash
pip install ultralytics opencv-python matplotlib seaborn tqdm kagglehub supervision lap filterpy
```

---

## How to Run

This project is designed to run on **Google Colab** with **Google Drive** for persistent storage.

### Step 1 — Mount Drive and Install Dependencies

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics kagglehub supervision lap filterpy
```

### Step 2 — Run Tasks Sequentially

Open `VisDrone_Project.ipynb` and execute tasks in order:

| Task   | Description                        |
|--------|------------------------------------|
| Task 1 | Dataset download & preprocessing  |
| Task 2 | YOLOv8 model training              |
| Task 3 | Inference & object counting        |
| Task 4 | ByteTrack multi-object tracking    |
| Task 5 | Evaluation & visualization         |

> Each task saves its state to a `.pkl` file on Google Drive. The next task loads this state automatically — you can safely restart the Colab runtime between tasks.

### Step 3 — Resume Training (if interrupted)

```python
from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/VisDrone-Project/runs/visdrone_yolov8/weights/last.pt")
results = model.train(resume=True)
```

---

## Output Files

All visual outputs are organized under `Outputs/`:

```
Outputs/
├── Task1_outputs/
│   ├── sample-drone-image.png
│   ├── annotated-drone-image.png
│   └── processed-dataset-visualization.png
├── Task2_outputs/
│   └── training-result.png
├── Task3_outputs/
│   ├── prediction.png
│   └── detection+counting.png
├── Task4_outputs/
│   ├── image-tracking-result.png
│   └── video-tracking-result.avi
└── Task5_outputs/
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── prediction-output.png
    └── model-performance-summary.png
```

---

## Notes

- The YOLOv8 **nano** (`yolov8n.pt`) variant was chosen for a balance of speed and accuracy suitable for real-time drone surveillance use cases.
- The **human** class shows lower recall (0.347) compared to **car** (0.712), likely due to the small pixel size of pedestrians in high-altitude drone imagery — a known challenge in the VisDrone benchmark.
- ByteTrack is used for tracking because it handles occlusions and re-entries well, which is common in aerial footage.
- Video stride of 2 was used during video tracking to reduce processing time without significant loss in tracking continuity.

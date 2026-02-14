# Hand Gesture Recognition

A research project on **sign and hand gesture recognition** using transfer learning across multiple datasets and model architectures. This repository implements static ASL letter recognition and device-control gesture recognition (Hagrid), with support for training, evaluation, and real-time demos.

---

## Overview

The project compares gesture recognition efficiency across:

1. **ASL (American Sign Language) letters** — Static 28×28 grayscale images (Sign Language MNIST), 24 classes (A–Y excluding J, Z), using a **CNN**.
2. **Hagrid** — FullHD RGB images, 18 gesture classes + “no_gesture”, for device control (e.g. call, like, peace, stop). Uses **ResNet18** (and related backbones) with optional leading-hand detection.
3. **Google GISLR** — Word-level sign language in videos (250 classes). Described in the report; implementation and experiments are referenced on Kaggle.

All experiments use transfer learning and are documented in the course report and presentation (see [Project info](#project-info)).

---

## What’s in This Repo

| Component | Description |
|-----------|-------------|
| **ASL** | CNN training and inference for Sign Language MNIST; real-time webcam demo with MediaPipe hand detection. |
| **Hagrid** | Dataset loading, preprocessing, training/eval pipeline, and live demo for gesture classification (ResNet18, etc.). |
| **models.py** | Shared two-headed ResNet/ResNeXt definition (gesture + leading hand). |
| **main.py** | Entry point that runs ASL training and display demo. |
| **Hagrid/run.py** | CLI to train or test the Hagrid model from a YAML config. |
| **Hagrid/demo.py** | Real-time webcam demo for Hagrid gestures (optional MediaPipe landmarks). |
| **FinalReport.pdf** | Full project report (datasets, methods, results). |

### Dataset summary

- **ASL (Sign Language MNIST):** ~1,000 images per letter, 24 letters, 28×28 grayscale.
- **Hagrid:** 550k+ FullHD images, 18 gesture classes + no_gesture, 34k+ subjects; train/test split by subject.

---

## Setup

### Requirements

- Python 3.8+ (tested with 3.x)
- PyTorch, torchvision, TensorBoard  
- Keras (for ASL CNN)  
- OpenCV, MediaPipe, NumPy, Pandas, scikit-learn  
- OmegaConf, tqdm, matplotlib, seaborn  

Install dependencies:

```bash
pip install -r requirements.txt
```

### Data

- **ASL:** Place Sign Language MNIST CSVs (`sign_mnist_train.csv`, `sign_mnist_test.csv`) in a known path and set that path in `ASL/t.py` (see [TODOs](TODOS_AND_FIXMES.md) for making paths configurable).
- **Hagrid:** Download the [HaGRID dataset](https://github.com/hukenovs/hagrid) and set `dataset.dataset` and `dataset.annotations` in `Hagrid/config/default.yaml` to your local paths.

---

## Usage

### ASL (train + demo)

Train the CNN and run the webcam demo (from repo root):

```bash
python main.py
```

This runs `ASL.t.train_model()` and `ASL.t.display()`. The demo uses the camera, crops the hand via MediaPipe, resizes to 28×28, and shows top-3 letter predictions. Press **Space** to capture a frame for prediction, **Esc** to exit.

### Hagrid (train / test)

Training:

```bash
python -m Hagrid.run -c train -p Hagrid/config/default.yaml
```

Testing:

```bash
python -m Hagrid.run -c test -p Hagrid/config/default.yaml
```

Ensure the config points to your Hagrid data and (if needed) a checkpoint path.

### Hagrid (live demo)

Run the real-time gesture demo (requires a trained checkpoint in config):

```bash
python -m Hagrid.demo -p Hagrid/config/default.yaml
```

Optional: add `-lm` to overlay MediaPipe hand landmarks. Press **q** to quit.

---

## Project structure (high level)

```
.
├── README.md              # This file
├── PROJECT_INFO.md        # Group, description, links
├── TODOS_AND_FIXMES.md    # TODOs and FIXMEs
├── requirements.txt
├── main.py                # ASL entry (train + display)
├── models.py              # Shared ResNet/ResNeXt (gesture + leading hand)
├── my_train.py            # Alternate Hagrid training entry (config-driven)
├── ASL/
│   └── t.py               # ASL CNN train + webcam display
├── Hagrid/
│   ├── run.py             # Train / test CLI
│   ├── demo.py            # Webcam demo
│   ├── dataset.py         # Gesture dataset
│   ├── train.py           # Training loop
│   ├── preprocess.py      # Transforms
│   ├── utils.py           # Model build, metrics, etc.
│   ├── constants.py       # Gesture targets
│   ├── config/
│   │   └── default.yaml   # Dataset paths, model, training params
│   └── Models/
│       ├── model.py       # TorchVision wrapper
│       ├── resnet.py      # ResNet backbone
│       └── fasterrcnn.py  # Faster R-CNN (if used)
└── FinalReport.pdf        # Project report
```

---

## Project info

- **Group:** Kristine N Umeh, Atharva Pandkar, Dharsan Krishnamoorthy  
- **Course:** PRCV Final Project  
- **Details, links, and full description:** see [PROJECT_INFO.md](PROJECT_INFO.md).  
- **TODOs and FIXMEs (e.g. user-facing app, path config):** see [TODOS_AND_FIXMES.md](TODOS_AND_FIXMES.md).

---

## License

See repository or course guidelines for license and academic use.

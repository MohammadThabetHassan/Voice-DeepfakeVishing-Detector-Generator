# Voice Deepfake Vishing Detector & Generator

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Detect and generate voice deepfakes, focusing on vishing (voice phishing) attacks using real and synthetic speech. Includes behavioral biometrics for enhanced detection.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Models & Datasets](#models--datasets)
- [Community & Support](#community--support)
- [License & Disclaimer](#license--disclaimer)

---

## Features

- **Deepfake Detection**: Classify `.wav` voice samples as real or fake using acoustic features (MFCCs, spectral centroid, jitter, shimmer) and a trained classifier.
- **Deepfake Generation**: Create synthetic voice samples by pitch-shifting and noise-injection (demo version; full TTS integration in future).
- **Behavioral Biometrics**: Profile caller speech patterns for anomaly detection.
- **VoIP Integration**: Real-time call analysis (Asterisk/FreeSWITCH) with Docker deployment.
- **Web Demo**: Upload and test audio via `index.html`.

---

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator.git
    cd Voice-Deepfake-Vishing-Detector-Generator
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **(Optional) Download or Train Models**
   - Ensure `deepfake_detector.pkl` is present (see documentation for training).

---

## Usage

**Python Pipeline (Recommended):**

```python
from pipeline import detect_deepfake, generate_deepfake

result = detect_deepfake("path/to/sample.wav")
print(f"Detection result: {result}")

generate_deepfake("path/to/input.wav", "path/to/output.wav", "optional text prompt")
```

**Command-Line (if provided):**

- See `scripts/` for batch and CLI usage examples.

**Web Demo:**

- Open `index.html` in your browser and follow instructions to upload a `.wav` file for detection.

---

## Project Structure

```
.
├── data/                # Voice datasets and samples
├── models/              # Pretrained/classifier models (deepfake_detector.pkl)
├── scripts/             # Utility scripts for training, evaluation
├── pipeline.py          # Main detection/generation pipeline
├── index.html           # Web demo interface
├── requirements.txt     # Python dependencies
├── README.md
└── project_spec.md      # Project specification and scope
```

---

## Requirements

- **Python 3.10+**
- Packages: `numpy`, `pandas`, `librosa`, `scikit-learn`, `xgboost`, `tensorflow`/`PyTorch`, `matplotlib`, `seaborn`, `joblib`
- **Jupyter Notebook** (optional, for exploration/training)
- **Docker** (for VoIP integration/deployment)
- **Asterisk/FreeSWITCH** (for real-time call analysis)
- (Optional) Speech synthesis tools for generating deepfakes

---

## Models & Datasets

- **Classifier:** Trained on MFCCs and other acoustic features; see `pipeline.py`
- **Datasets:** VoxCeleb (real), synthetic calls (TTS, Coqui, ElevenLabs), optional call-center data
- **Output:** Model as `deepfake_detector.pkl`, feature CSVs

---

## Community & Support

- Maintainer: [Mohammad Thabet Hassan](https://github.com/MohammadThabetHassan)
- Contributors: See [GitHub Contributors](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/graphs/contributors)
- For help, open an issue on GitHub

---

## License & Disclaimer

This project is licensed under the MIT License. See [LICENSE](LICENSE).

> **Disclaimer:**  
> Research and educational use only. Do not use for malicious purposes. Comply with all laws and obtain consent before analyzing or generating voice data.

---

**References & Further Reading:**
- [ASVspoof Challenge](https://www.asvspoof.org/)
- [Voice Deepfake Detection Research](https://arxiv.org/abs/1910.11916)

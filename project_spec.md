# Voice Deepfake Vishing Detection & Response System – Project Specification

## Problem Statement

Deepfake‐enabled **vishing** (voice phishing) attacks have evolved from fringe social‐engineering tricks into a powerful and scalable crime vector. Attackers now use sophisticated AI models to clone a target’s voice with only a few minutes of sample audio, enabling live phone calls that sound eerily authentic【308076305286644†L90-L101】. In early 2025, for example, criminals siphoned **US$25 million** from a European energy company after using a synthetic clone of the CFO’s voice to authorise a wire transfer【308076305286644†L90-L100】. Threat‑intelligence reports indicate that **deepfake vishing incidents surged by over 1 600 % in Q1 2025** versus the previous quarter【308076305286644†L111-L129】, with median losses around **US$1 400 per victim** and some sectors seeing 70 % of organisations targeted【308076305286644†L121-L141】. Traditional security awareness measures are struggling to keep up as attackers exploit human trust and well‑integrated VoIP systems.  

This project aims to design and implement a **Voice Deepfake Vishing Detection & Response System** that can identify synthetic voices in real time, perform secondary behavioural checks, and trigger secure response workflows. By doing so, it will help prevent financial losses and improve organisational resilience against modern social‑engineering attacks.

## Scope

1. **Data Collection & Feature Extraction** – Acquire a balanced dataset (~200 samples) of real human voices from open speech corpora (e.g., VoxCeleb) and generate an equivalent number of deepfake calls using text‑to‑speech engines (Coqui TTS, ElevenLabs). Extract acoustic features such as Mel Frequency Cepstral Coefficients (MFCCs), spectral centroids, jitter and shimmer.
2. **Model Training & Evaluation** – Train and compare multiple classifiers (e.g., XGBoost and a small deep neural network) using 5‑fold cross‑validation to distinguish real voices from deepfakes. Evaluate accuracy, precision, recall, F1‑score and confusion matrices. Save the best performing model.
3. **Behavioural Biometrics Module** – Develop a secondary component that learns typical speech profiles (pitch, cadence, lexical choices) for known callers and flags anomalies, enhancing detection even when the acoustic classifier is fooled.
4. **Real‑Time VoIP Integration** – Integrate the detection pipeline with an open‑source softphone (Asterisk or FreeSWITCH) so that incoming calls are intercepted, analysed and, if necessary, routed to a second‑factor verification. Provide a Dockerised deployment.
5. **Documentation & Delivery** – Produce an IEEE‑style report, code repository, configuration files and a Dockerfile. Package all artefacts for reproducibility in a UAE university lab.

## Requirements

### Hardware

* Standard computer or server with **multicore CPU** and **GPU** (e.g., NVIDIA GTX/RTX) to train models efficiently.  
* Microphone or VoIP headset for capturing test calls.  
* Network connectivity to run VoIP software (Asterisk/FreeSWITCH).  
* No specialised DSP or FPGA hardware is required.

### Software

* **Python 3.10+** environment with packages: `numpy`, `pandas`, `librosa`, `scikit‑learn`, `xgboost`, `tensorflow`/`PyTorch`, `matplotlib`, `seaborn`.  
* Jupyter Notebook for reproducible data exploration and model training.  
* VoIP server software (e.g., **Asterisk** or **FreeSWITCH**) for call interception.  
* **Docker** and `docker‑compose` for containerised deployment.  
* Optional: Speech‑synthesis tools (Coqui TTS, ElevenLabs API) for generating deepfake samples.

### Datasets

* **Real voice recordings** – Use publicly available speech corpora like **VoxCeleb**, which contains thousands of celebrity interview clips. (The dataset is widely used for speaker recognition tasks and provides diverse accents, noise conditions and utterances.)  
* **Synthetic voice samples** – Generate deepfake calls using open‑source TTS models. Ensure that the same speakers and phrases are matched across real and synthetic datasets.  
* Optionally, gather **call‑centre recordings** or publicly available vishing audio (where permitted) to enhance realism.

## Expected Outcomes

By the end of this project, we will deliver:

1. A reproducible **data and feature pipeline** (notebook and scripts) that downloads/produces voice samples and extracts acoustic features into a CSV.  
2. A trained **deepfake detection model** saved as a pickle file, along with evaluation metrics showing its performance.  
3. A **behavioural biometrics module** that profiles caller speech patterns and supplements the detector.  
4. A **real‑time integration** with a VoIP softphone (Asterisk/FreeSWITCH) and a Docker container for deployment.  
5. A comprehensive **IEEE‑style paper** detailing background, methodology, results, and future work, supported by credible statistics on the growth of deepfake vishing【308076305286644†L90-L102】【308076305286644†L111-L129】.


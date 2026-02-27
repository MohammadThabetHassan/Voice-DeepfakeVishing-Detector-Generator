# Evaluation Methodology

## Dataset

| Source | Category | Samples |
|---|---|---|
| Open Speech Repository (OSR) | Real | ~240 |
| gTTS synthetic speech | Fake | ~235 |
| **Total** | | **~475** |

Both `features.csv` (400 rows) and `osr_features.csv` (475 rows) are provided pre-extracted.

---

## Feature Extraction

Three feature strategies are compared:

### 1. MFCC-only (13 dimensions)

**Process:**
1. Pre-emphasis filter (α = 0.97)
2. Frame segmentation: 25 ms windows, 10 ms step
3. Hamming window applied per frame
4. 512-point FFT → Power spectrum
5. 26 Mel filterbank channels
6. DCT → first 13 coefficients (discarding C0)
7. Mean pooling over all frames → 13-dim vector

**Rationale:** MFCCs approximate the human auditory system and are widely used in speech processing. Deepfake speech shows distinct MFCC patterns due to vocoder artifacts.

### 2. FFT/Spectral-only (6 dimensions)

**Features extracted:**
- **Spectral Centroid** (Hz): centre of mass of the frequency spectrum
- **Spectral Bandwidth** (Hz): spread around the centroid
- **Spectral Rolloff** (Hz): frequency below which 85% of energy resides
- **Log Low-Band Energy**: log(1 + mean |FFT| for 0–1/3 of Nyquist)
- **Log Mid-Band Energy**: log(1 + mean |FFT| for 1/3–2/3)
- **Log High-Band Energy**: log(1 + mean |FFT| for 2/3–Nyquist)

**Rationale:** Deepfake voices often have unnatural high-frequency energy distribution and spectral rolloff characteristics.

### 3. Hybrid MFCC + FFT (19 dimensions)

Concatenation of MFCC (13) + FFT (6) vectors. Expected to outperform either alone by combining temporal (MFCC) and spectral (FFT) information.

---

## Classifier

All three models use an identical **Gradient Boosting Classifier** pipeline:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )),
])
```

GBM was selected over Random Forest due to:
- Better performance on imbalanced small datasets
- Built-in feature importance
- Lightweight enough for IoT deployment (model size < 2 MB)

---

## Evaluation Protocol

**5-fold Stratified Cross-Validation** (`StratifiedKFold(n_splits=5, shuffle=True)`)

Predictions are collected via `cross_val_predict` (out-of-fold), then all metrics are computed on the aggregated predictions.

### Metrics Reported

| Metric | Formula | Importance |
|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Low false-alarm rate |
| **Recall** | TP/(TP+FN) | Catching all deepfakes (primary) |
| **F1** | 2·P·R/(P+R) | Balanced (main selection criterion) |
| Confusion Matrix | — | Detailed error analysis |

Recall (sensitivity) is the primary criterion because **missing a deepfake (FN) is more dangerous than a false alarm (FP)** in a vishing detection context.

---

## Runtime Benchmarks

For each model, the following are measured:
- **Feature extraction time** (per sample, averaged over all training samples)
- **Inference time** (1000 single-sample predictions, reported as total ms)

These are critical for IoT/edge deployment where latency matters.

---

## Expected Results

Based on the existing OSR dataset and the three strategies, expected approximate performance:

| Model | Accuracy | F1 | Notes |
|---|---|---|---|
| MFCC | ~0.88–0.93 | ~0.88–0.92 | Strong baseline |
| FFT | ~0.82–0.88 | ~0.80–0.87 | Weaker on accent variation |
| **Hybrid** | **~0.90–0.95** | **~0.90–0.94** | Best overall |

*Actual results depend on dataset; check `models/results.json` for computed values.*

---

## Limitations

1. **Dataset size**: ~475 samples is small. Larger datasets (ASVspoof, FoR) would improve generalisability.
2. **Single-segment evaluation**: Only first 1 second of audio is used. Longer segment averaging would improve robustness.
3. **Distribution shift**: Models trained on OSR + gTTS may not generalise to Coqui TTS, ElevenLabs, or other modern deepfake systems.
4. **No adversarial examples**: Models are not tested against adversarially crafted audio.

---

## Future Work

- Train on ASVspoof 2019/2021 dataset (millions of samples)
- Add log-mel spectrogram + small CNN (2D feature map)
- Implement Equal Error Rate (EER) metric
- Test against state-of-the-art deepfakes (ElevenLabs, Vall-E, etc.)
- Real-time streaming inference (sliding window)

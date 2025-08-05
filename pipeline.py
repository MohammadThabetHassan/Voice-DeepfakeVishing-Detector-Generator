"""
Pipeline module for the Voice Deepfake Vishing Detection & Response System.

This module exposes two primary functions:

1. **detect_deepfake(audio_path)** – Given a path to a `.wav` file, the function
   reads the audio, extracts key speech features (MFCCs, spectral features,
   jitter, shimmer) and uses a pre‑trained classifier to determine whether
   the sample is likely a real or deepfake voice.
2. **generate_deepfake(input_audio_path, output_audio_path, text)** – Given an
   input recording and a text prompt, this function synthesizes a "deepfake"
   version of the recording by applying a pitch shift and noise. In a fully
   featured system, you would integrate a text‑to‑speech model here. For
   demonstration purposes, we modulate the pitch based on the text length and
   add minor noise to create a synthetic variant of the voice.

The pipeline assumes the classifier model (`deepfake_detector.pkl`) exists in
the same directory. If you retrain the model, make sure to overwrite the
pickle file.
"""

import numpy as np
import joblib
from scipy.io import wavfile
from scipy import signal
import scipy.fftpack as fftpack
import os
from gtts import gTTS
from pydub import AudioSegment
import traceback

"""
This module implements feature extraction, deepfake detection, and simple
voice‐modification for demonstration purposes. To improve generalisation,
audio is resampled to a fixed sample rate (16 kHz) prior to feature
extraction. When retraining the model on new data (e.g. Open Speech
Repository recordings), save the resulting pickle file alongside this
module and update ``MODEL_PATH`` accordingly.
"""

# Default model path. Update this constant if you train a new model
# (e.g. ``deepfake_detector_osr.pkl``) and wish to use it for inference.
MODEL_PATH = 'deepfake_detector_osr.pkl'


def load_model():
    """Load the pre‑trained deepfake detection model."""
    return joblib.load(MODEL_PATH)


def compute_features(segment: np.ndarray, sr: int):
    """Compute MFCCs, spectral features, jitter, and shimmer for a given audio segment.

    Parameters
    ----------
    segment : np.ndarray
        The audio signal segment (1D array).
    sr : int
        Sample rate of the audio.

    Returns
    -------
    np.ndarray
        A 1D feature vector containing 13 MFCCs followed by spectral centroid,
        spectral bandwidth, spectral rolloff, jitter and shimmer.
    """
    # Pre‑emphasis filter
    pre_emphasis = 0.97
    emphasized = np.append(segment[0], segment[1:] - pre_emphasis * segment[:-1])

    # Framing parameters
    frame_length = 0.025  # 25ms
    frame_step = 0.010    # 10ms
    frame_length_samples = int(round(frame_length * sr))
    frame_step_samples = int(round(frame_step * sr))
    signal_length = len(emphasized)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length_samples)) / frame_step_samples))
    pad_signal_length = num_frames * frame_step_samples + frame_length_samples
    pad_signal = np.append(emphasized, np.zeros(pad_signal_length - signal_length))

    # Create frames
    indices = (np.tile(np.arange(0, frame_length_samples), (num_frames, 1)) +
               np.tile(np.arange(0, num_frames * frame_step_samples, frame_step_samples), (frame_length_samples, 1)).T)
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length_samples)

    # FFT and power spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))

    # Mel filterbank parameters
    nfilt = 26
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_freqs = np.floor((NFFT + 1) * hz_points / sr).astype(int)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = bin_freqs[m - 1]
        f_m = bin_freqs[m]
        f_m_plus = bin_freqs[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_freqs[m - 1]) / (bin_freqs[m] - bin_freqs[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_freqs[m + 1] - k) / (bin_freqs[m + 1] - bin_freqs[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    mfcc = fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, :13]
    mfcc_mean = np.mean(mfcc, axis=0)

    # Spectral features
    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
    centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / np.sum(spectrum))
    cumulative_energy = np.cumsum(spectrum)
    rolloff_threshold = 0.85 * cumulative_energy[-1]
    rolloff_index = np.where(cumulative_energy >= rolloff_threshold)[0][0]
    rolloff = freqs[rolloff_index]

    # Jitter and shimmer approximations
    diff_signal = np.diff(segment)
    jitter = np.mean(np.abs(diff_signal))
    envelope = np.abs(segment)
    shimmer = np.std(np.diff(envelope))

    return np.concatenate([mfcc_mean, [centroid, bandwidth, rolloff, jitter, shimmer]])


def detect_deepfake(audio_path: str) -> str:
    """Classify an audio file as 'real' or 'fake'.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (.wav) to classify.

    Returns
    -------
    str
        Prediction label ('real' or 'fake').
    """
    sr, data = wavfile.read(audio_path)
    # Convert to mono if the audio has multiple channels
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Convert sample type to float and normalise
    data = data.astype(np.float32)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data /= max_val
    # Target sample rate for feature extraction
    target_sr = 16000
    if sr != target_sr:
        # Resample to the target sample rate
        new_length = int(len(data) * target_sr / sr)
        data = signal.resample(data, new_length)
        sr = target_sr
    # Use a one‑second segment; zero‑pad if shorter
    segment_length = sr  # one second in samples
    if len(data) >= segment_length:
        segment = data[:segment_length]
    else:
        segment = np.pad(data, (0, segment_length - len(data)), mode='constant')
    features = compute_features(segment, sr)
    # Ensure features are a DataFrame with correct column names
    import pandas as pd
    feature_names = [
        'MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9','MFCC10','MFCC11','MFCC12','MFCC13',
        'centroid','bandwidth','rolloff','jitter','shimmer'
    ]
    features_df = pd.DataFrame([features], columns=feature_names)
    # Load the classifier and predict
    model = load_model()
    pred = model.predict(features_df)[0]
    return pred


# Add Coqui TTS integration
try:
    from TTS.api import TTS
    coqui_tts_available = True
except ImportError:
    coqui_tts_available = False


def ensure_mono_16k(input_audio_path: str) -> str:
    """Convert input audio to mono, 16kHz WAV if needed. Returns new path if converted, else original."""
    audio = AudioSegment.from_file(input_audio_path)
    if audio.channels != 1 or audio.frame_rate != 16000:
        converted_path = input_audio_path.replace('.wav', '_mono16k.wav')
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(converted_path, format="wav")
        return converted_path
    return input_audio_path


def generate_deepfake(input_audio_path: str, output_audio_path: str, text: str) -> None:
    """Create a synthetic deepfake version of an input audio file using Coqui TTS if available."""
    print(f"[generate_deepfake] Input: {input_audio_path}, Output: {output_audio_path}, Text: {text}")
    used_fallback = False
    # Ensure input audio is mono/16kHz
    input_audio_path = ensure_mono_16k(input_audio_path)
    if coqui_tts_available:
        print("[generate_deepfake] Using Coqui TTS for voice cloning.")
        try:
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
            tts.tts_to_file(text=text, speaker_wav=input_audio_path, file_path=output_audio_path, language="en")
            if not os.path.exists(output_audio_path):
                raise RuntimeError(f"[!] Model file not found in the output path after synthesis. Check if input audio is clean (mono, 16kHz, single speaker) and ffmpeg is installed.")
        except Exception as e:
            print(f"[generate_deepfake] Coqui TTS failed: {e}. Falling back to gTTS.")
            traceback.print_exc()
            used_fallback = True
    else:
        used_fallback = True
    if used_fallback:
        print("[generate_deepfake] Using gTTS for text-to-speech.")
        tts = gTTS(text=text, lang='en')
        tts.save(output_audio_path)
    print(f"[generate_deepfake] Output file created: {output_audio_path}, size: {os.path.getsize(output_audio_path) if os.path.exists(output_audio_path) else 'N/A'} bytes")


def detect_deepfake_audio(audio_path: str) -> str:
    """Detect if the given audio file is a deepfake or real using the trained model."""
    sr, data = wavfile.read(audio_path)
    # Ensure mono
    if len(data.shape) > 1:
        data = data[:, 0]
    # Use the whole audio, regardless of length
    segment = data
    features = compute_features(segment, sr)
    model = load_model()
    pred = model.predict([features])[0]
    return 'deepfake' if pred == 1 else 'real'

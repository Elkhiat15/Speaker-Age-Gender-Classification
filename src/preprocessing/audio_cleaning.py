import librosa
import numpy as np

def preprocess_audio(file_path, target_sr=16000, top_db=30):
    """
    Load and preprocess an audio file:
    - Load with original sampling rate
    - Convert to mono
    - Remove silence
    - Resample to target sampling rate
    """
    # Load audio
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # Remove silence
    intervals = librosa.effects.split(y, top_db=top_db)
    y = np.concatenate([y[start:end] for start, end in intervals])

    # Resample
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    return y, target_sr

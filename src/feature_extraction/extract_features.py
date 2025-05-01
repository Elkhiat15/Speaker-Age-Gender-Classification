import librosa
import numpy as np
from constants.constants import N_FFT, N_MELS, N_MFCC

def extract_features(y, sr):
    """
    Extracts MFCC, delta MFCC, and pitch features from an audio signal using librosa.

    Args:
        y (np.ndarray): The audio signal as a NumPy array.
        sr (int): The sample rate of the audio signal.

    Returns:
        np.ndarray: A 1D array containing the concatenated features 
                    (mean MFCC, std MFCC, mean delta MFCC, std delta MFCC, pitch).
    """

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, n_mels=N_MELS)

    # Delta MFCCs
    mfcc_delta = librosa.feature.delta(mfcc)

    # Pitch (using piptrack)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=N_FFT)
    mask = magnitudes > np.median(magnitudes)
    pitch = np.mean(pitches[mask]) if np.any(mask) else 0.0

    # Feature vector
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.std(mfcc_delta, axis=1),
        [pitch]
    ])

    return features

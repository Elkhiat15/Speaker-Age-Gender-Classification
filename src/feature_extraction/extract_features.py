
import librosa
import torch
import numpy as np
import torchaudio.transforms as T

from constants.constants import N_FFT, N_MELS, N_MFCC


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features(y, sr):
    """
    Extracts MFCC, delta MFCC, and pitch features from an audio signal using GPU acceleration.

    Args:
        y (torch.Tensor): The audio signal as a tensor.
        sr (int): The sample rate of the audio signal.

    Returns:
        np.ndarray: A 1D array containing the concatenated features (mean MFCC, std MFCC, mean delta MFCC, std delta MFCC, pitch).
    """
    y = y.to(device) 

    # Extract MFCC features
    mfcc_transform = T.MFCC(sample_rate=sr, n_mfcc=N_MFCC, 
                            melkwargs={"n_fft": N_FFT,"n_mels": N_MELS}).to(device)
    mfcc = mfcc_transform(y).squeeze(0).cpu().numpy()

    # Extract delta features
    delta_transform = T.ComputeDeltas().to(device)
    mfcc_delta = delta_transform(torch.tensor(mfcc, device=device)).cpu().numpy()

    y_np = y.squeeze(0).cpu().numpy()  # Convert tensor to numpy array for librosa
    pitches, magnitudes = librosa.core.piptrack(y=y_np, sr=sr)
    pitch = np.mean(pitches[magnitudes > np.median(magnitudes)])

    # Concatenate features
    features = np.concatenate([
        np.mean(mfcc, axis=1),  
        np.std(mfcc, axis=1),  
        np.mean(mfcc_delta, axis=1),  
        np.std(mfcc_delta, axis=1),  
        [pitch]
    ])

    return features

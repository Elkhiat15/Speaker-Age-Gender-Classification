import os
import joblib
import natsort

def get_audio_paths(base_dir, start, end):
    """
    Retrieve all audio paths in the directory.
    """
    return [
        os.path.join(base_dir, fname)
        for fname in os.listdir(base_dir)[start:end]
    ]

def load_model(model_path):
    """
    Load the specified model from the models directory.
    """
    model = joblib.load(model_path)
    
    return model

def get_sorted_files(data_dir):
    """
    Get a list of audio files in natural sorted order (e.g., 1.mp3 before 10.mp3).
    Args:
        data_dir (str): Path to directory with audio files.
    Returns:
        list: Sorted file paths.
    """
    
    files = os.listdir(data_dir)
    return natsort.natsorted(files)

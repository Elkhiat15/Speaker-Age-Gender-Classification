import os

def get_audio_paths(base_dir, start, end):
    """
    Retrieve all audio paths in the directory.
    """
    return [
        os.path.join(base_dir, fname)
        for fname in os.listdir(base_dir)[start:end]
    ]

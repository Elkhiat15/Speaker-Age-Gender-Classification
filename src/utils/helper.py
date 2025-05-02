import os
import joblib
import natsort
import pandas as pd
from constants.constants import N_MFCC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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


def features_to_df(features):
     # Convert the batch to a DataFrame
    features_df = pd.DataFrame(
        features,
        columns=
        [f"mean_mfcc_{i}" for i in range(N_MFCC)] + 
        [f"std_mfcc_{i}" for i in range(N_MFCC)] + 
        [f"mean_delta_mfcc_{i}" for i in range(N_MFCC)] + 
        [f"std_delta_mfcc_{i}" for i in range(N_MFCC)] +  
        ["pitch"]
        )
    return features_df



def save_outputs(results, results_path="output/results.txt", type = 0):
    """
    Save prediction results to a text file.

    Args:
        results (Iterable): List or array of prediction results.
        results_path (str): Path to save the results file.
    """
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if type ==0:
        with open(results_path, "w") as f:
            for res in results:
                f.write(f"{res}\n")
    else:
        with open(results_path, "w") as f:
                f.write(f"{results}\n")


def get_splitted_data(X, y, df, target):
    """
    Splits the data into train, validation, and test sets.
    """
    X = df.drop(columns=['label', 'gender', 'age', 'voice_id'])
    y = df[target]
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_val_train, X_test, y_val_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_val_train, y_val_train, test_size=0.18, random_state=42, stratify=y_val_train)

    return X_train, X_val, y_train, y_val

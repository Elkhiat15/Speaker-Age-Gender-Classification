import os 
import time
from tqdm import tqdm
import pandas as pd
import torch

from extract_features import extract_features
from utils.helper import get_audio_paths
from preprocessing.audio_cleaning import preprocess_audio
from constants.constants import N_MFCC

def save_features_in_batches_as_csv(audio_paths, batch_size=32, output_csv='features.csv'):
    """
    Save features in batches to prevent loss of progress during long processing.
    Args:
        audio_paths (list): List of audio file paths.
        batch_size (int): Number of audio files to process per batch.
        output_csv (str): Path where features CSV will be saved.
    """
    # Open the CSV file and write the header if it's the first batch
    header_written = False

    # Process audio files in batches
    for start_idx in tqdm(range(0, len(audio_paths), batch_size), desc="Processing Batches"):
        end_idx = start_idx + batch_size
        if end_idx > 172158 :
            end_idx = None
        batch_paths = audio_paths[start_idx:end_idx]
        batch_features = []
        batch_voice_ids = []
        start = time.time()

        # Process each audio file in the batch
        for audio_path in batch_paths:
            # y, sr = torchaudio.load(audio_path)
            y, resampled_sr = preprocess_audio(audio_path)
            features = extract_features(y, resampled_sr)
            batch_features.append(features)
            voice_id = os.path.basename(audio_path)
            batch_voice_ids.append(voice_id)

        # Convert the batch to a DataFrame
        features_df = pd.DataFrame(batch_features, columns=[f"mean_mfcc_{i}" for i in range(N_MFCC)] +
                                                          [f"std_mfcc_{i}" for i in range(N_MFCC)] +
                                                          [f"mean_delta_mfcc_{i}" for i in range(N_MFCC)] +
                                                          [f"std_delta_mfcc_{i}" for i in range(N_MFCC)] + ["pitch"])
        features_df['voice_id'] = batch_voice_ids

        # Write to CSV (append if file already exists, otherwise create new file)
        mode = 'a' if header_written else 'w'
        header = not header_written
        features_df.to_csv(output_csv, mode=mode, header=header, index=False)

        # Set flag to indicate that header has been written
        header_written = True
        # print(f"Saved batch to {output_csv}.")
        end = time.time()
        print(f"Time to compute features for batch {start_idx//batch_size + 1}: ", end - start, "seconds")


# example usage

# base_dir = <the base directory for audio files>
# START, END, BATCH_SIZE = , , <put yours> 
# audio_paths = get_audio_paths(base_dir, start=START, end=END)
# # Save features in batches to prevent session timeout
# save_features_in_batches_as_csv(audio_paths, batch_size=BATCH_SIZE, output_csv='/kaggle/working/features2.csv')
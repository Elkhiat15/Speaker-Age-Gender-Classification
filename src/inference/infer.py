from src.preprocessing.audio_cleaning import preprocess_audio
from utils.helper import features_to_df, get_sorted_files, save_outputs, load_model
from feature_extraction.extract_features import extract_features
import argparse
import os
from constants.constants import (
    MODEL_PATH,
    AGE_MODEL_PATH,
    GENDER_MODEL_PATH, 
    HYBRID
)

def infer(audio_dir,output_dir,  model1, model2=None):
    sorted_audio_dir = get_sorted_files(audio_dir)
    print("Sorted audio files:", sorted_audio_dir)

    features_list = []

    for audio_file in sorted_audio_dir:
        # audio_path = os.path.join(audio_dir, audio_file)
        print("Processing audio file:", audio_file)

        y, resampled_sr = preprocess_audio(audio_file)
        features = extract_features(y, resampled_sr)
        features_list.append(features)

    features_df = features_to_df(features_list)
    if HYBRID:
        age_prediction = model1.predict(features_df)
        gender_prediction = model2.predict(features_df)
        prediction = 3 - (1*gender_prediction + 2*age_prediction)
    else:
        prediction = model1.predict(features_df)

    save_outputs(prediction, f'{output_dir}/results.txt', 0)

    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if HYBRID:
        print("Hybrid model is used")
        age_model = load_model(AGE_MODEL_PATH)
        gender_model = load_model(GENDER_MODEL_PATH)
        infer(args.audio_dir, args.output_dir, age_model, gender_model)
    else:
        print("Base model is used")
        model = load_model(MODEL_PATH)
        infer(args.audio_dir, args.output_dir, model)

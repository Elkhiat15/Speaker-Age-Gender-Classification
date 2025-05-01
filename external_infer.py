# external_infer.py

import subprocess
import time
from src.utils.helper import save_outputs

def external_infer(audio_dir = "test_data/data"):
    """
    Calls infer.py as a subprocess.
    """
    model_path = "src/selected_model/knn_base_40.pkl"

    print("[INFO] Starting external inference subprocess...")
    start = time.time()

    subprocess.run([
        "python", "src/inference/infer.py",
        "--model_path", model_path,
        "--audio_dir", audio_dir
    ])

    end = time.time()
    tot = end - start
    save_outputs(tot, 'output/time.txt', 1)
    print(f"[INFO] Inference completed in {tot:.2f} seconds.")


if __name__ == "__main__":
    # audio_dir = input("Enter the path to the audio directory: ")
    external_infer()

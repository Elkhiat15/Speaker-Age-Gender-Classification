import subprocess
import time
import argparse
from src.utils.helper import save_outputs

def external_infer(audio_dir, output_dir):
    """
    Calls infer.py as a subprocess.
    """
    # model_path = "src/selected_model/knn_base_40_.pkl"

    print("[INFO] Starting external inference subprocess...")
    start = time.time()

    subprocess.run([
        "python", "src/inference/infer.py",
        "--audio_dir", audio_dir,
        "--output_dir", output_dir
    ])

    end = time.time()
    tot = end - start
    save_outputs(tot, f'{output_dir}/time.txt', 1)
    print(f"[INFO] Inference completed in {tot:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to the audio directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()
    external_infer(args.audio_dir, args.output_dir)


# example usage from the terminal:  
# move to the src directory and run (cd src):
# python external_infer.py --audio_dir test_data/data  --output_dir outputs 
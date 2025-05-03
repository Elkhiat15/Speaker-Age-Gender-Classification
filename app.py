import streamlit as st
import os
import shutil
from zipfile import ZipFile, BadZipFile
from external_infer import external_infer
from src.constants.constants import OUTPUT_DIR
from src.utils.helper import get_sorted_files

# Constants
UPLOAD_FOLDER = "unzipped_data"
TEMP_ZIP = "temp.zip"

# Prepare clean workspace
if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="Speaker Recognition", layout="centered")
st.title("🎙️ Speaker Gender & Age Recognition")

# Upload ZIP
uploaded_zip = st.file_uploader("Upload a ZIP file of audio samples", type=["zip"])

if uploaded_zip:
    try:
        # Save and extract
        with open(TEMP_ZIP, "wb") as f:
            f.write(uploaded_zip.read())

        try:
            with ZipFile(TEMP_ZIP, 'r') as zip_ref:
                zip_ref.extractall(UPLOAD_FOLDER)
        except BadZipFile:
            st.error("❌ Uploaded file is not a valid ZIP.")
            st.stop()

        # Look for 'data/' subfolder
        data_dir = os.path.join(UPLOAD_FOLDER, "data")
        if not os.path.isdir(data_dir):
            st.error("❌ The ZIP file must contain a top-level folder named 'data'.")
            st.stop()

        # Check if there are any audio files inside
        audio_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.wav', '.mp3'))]
        if not audio_files:
            st.error("❌ No audio files found inside 'data/' folder.")
            st.stop()

        st.success("✅ Files uploaded and validated!")
        st.text(f"Found {len(audio_files)} audio files.")
        sorted_files = get_sorted_files(data_dir)
        st.text("Sorted files first 5:")
        for file in sorted_files[:5]:
            st.markdown(f"- {file}")

        if st.button("Run Inference"):
            with st.spinner("🧠 Running inference..."):
                try:
                    external_infer(data_dir, OUTPUT_DIR)
                except Exception as e:
                    st.error(f"❌ Inference failed: {e}")
                    st.stop()

            st.success("✅ Inference complete!")

            # Paths
            results_path = os.path.join(OUTPUT_DIR, "results.txt")
            time_path = os.path.join(OUTPUT_DIR, "time.txt")
            zip_path = os.path.join(OUTPUT_DIR, "output_team8.zip")

            # Create ZIP if both result files exist
            if os.path.exists(results_path) and os.path.exists(time_path):
                # Read and display the time value
                with open(time_path, "r") as f:
                    time_value = f.read().strip()
                    st.info(f"⏱️ Inference Time: {time_value}")

                with ZipFile(zip_path, 'w') as zipf:
                    zipf.write(results_path, arcname="results.txt")
                    zipf.write(time_path, arcname="time.txt")

                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="📦 Download All Outputs (ZIP)",
                        data=f,
                        file_name="output_team8.zip",
                        mime="application/zip"
                    )
            else:
                if not os.path.exists(results_path):
                    st.warning("⚠️ results.txt not found.")
                if not os.path.exists(time_path):
                    st.warning("⚠️ time.txt not found.")

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")

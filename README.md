# 🎙️ Speaker Gender and Age Recognition
This project is a classical machine learning-based system that predicts speaker gender and age from audio recordings. It uses audio preprocessing, feature extraction (MFCC, chroma, spectral features), and models like KNN, LightGBM, XGBoost, and MLP.

# 🚀 Features
- Mono audio conversion and silence removal

- Rich audio feature extraction using Librosa

- Model inference for speaker gender and age classification

- Two modes: single model and hybrid model inference

- Streamlit interface for user interaction

- Docker support for easy deployment

# 🗂️ Project Structure

```
.
├── src/
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── utils/
│   ├── inference/
│   ├── constants/
│   ├── selected_model/        <-- Downloaded models go here
│   └── ...
├── external_infer.py
├── app.py                     <-- Streamlit app
├── Dockerfile
├── requirements.txt
└── README.md

```

# 🧠 Model Files (Required)
You must download the pre-trained models before running inference.

📥 Download from: https://drive.google.com/drive/u/0/folders/15WNHl2cG5Obvo050YSEsh2PU3KH_3QRA

After downloading, place the folder `selected_model` inside the `src/` directory:

# 🔧 How to run
✅ Option 1: Run Locally via Git Clone
- Clone the Repository

  ```bash
  git clone https://github.com/your-username/speaker-age-gender-classification.git
  cd speaker-age-gender-classification
  ```

- Install Dependencies

  ```bash
  # For Linux/macOS:
  python3 -m venv venv
  source venv/bin/activate
  
  # For Windows:
  python -m venv venv
  venv\Scripts\activate

  pip install -r requirements.txt
  
  cd src/
  pip install -e . # this to package the project modules 
  ```
  

- Download the models from the Drive link above and place them into src/selected_model/

Put a `data` folder that contains the test audios inside `src` directory and then run `external_infer.py`   


# 🐳 Option 2: Run with Docker
- Create a folder “the base folder”.
  
- Change the directory to this folder.
  
- Put inside this folder the folder named “data” including the test audios.
  
- Open Docker Desktop.
  
- Run those two commands in the terminal “make sure to be in the base folder directory”.

  ```bash
  docker pull khiat/age-gender:latest
  ```

  ```bash
  docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/output:/app/output" \
    khiat/age-gender:latest
  ```
 
- Now you will find a directory named output the has the two text files `results.txt` and `time.txt`.

# Option 3: Run Streamlit App

- run the Streamlit App by:

  ```bash
  streamlit run app.py
  ```

- Prepare a ZIP file called `data.zip` and upload it  

- Click "Run Inference"  

- Download results from the provided download button



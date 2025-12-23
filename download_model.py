import os
import gdown

MODEL_PATH = "models/plant_disease_recog_model_pwp.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇ Downloading ML model...")
        os.makedirs("models", exist_ok=True)

        url = "https://drive.google.com/uc?id=1_C9sC-V9ku5X5OVATjQnh9EC2bfhDDdh"
        gdown.download(url, MODEL_PATH, quiet=False)

        print("✅ Model downloaded successfully")
    else:
        print("✅ Model already exists")

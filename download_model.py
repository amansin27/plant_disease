import os
import gdown

MODEL_PATH = "plant_disease.tflite"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇ Downloading TFLite model...")
        os.makedirs("models", exist_ok=True)

        url = "https://drive.google.com/uc?id=1m1yrCbYrIi0y00EM-dz0AW7Gd7_dtMSa"
        gdown.download(url, MODEL_PATH, quiet=False)

        print("✅ TFLite model downloaded successfully")
    else:
        print("✅ Model already exists")

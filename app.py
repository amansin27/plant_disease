from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import tensorflow as tf
import os

from download_model import download_model

app = Flask(__name__)

# =========================
# FOLDERS
# =========================
UPLOAD_FOLDER = "uploadimages"
MODEL_PATH = "plant_disease.tflite"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# DOWNLOAD MODEL IF NEEDED
# =========================
download_model()

# =========================
# LOAD TFLITE MODEL
# =========================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# LOAD LABELS
# =========================
with open("plant_disease.json", "r") as file:
    plant_disease = json.load(file)

# =========================
# ROUTES
# =========================
@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

# =========================
# IMAGE PREPROCESSING
# =========================
def extract_features(image_path):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(160, 160),
        color_mode="rgb"
    )
    image = tf.keras.utils.img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0).astype("float32")
    return image

# =========================
# PREDICTION
# =========================
def model_predict(image_path):
    img = extract_features(image_path)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = int(np.argmax(output))

    return plant_disease[predicted_index]

# =========================
# UPLOAD & PREDICT
# =========================
@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']

        temp_filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        image.save(temp_path)

        prediction = model_predict(temp_path)

        return render_template(
            'home.html',
            result=True,
            imagepath=f'/{temp_path}',
            prediction=prediction
        )
    else:
        return redirect('/')

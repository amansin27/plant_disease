import tensorflow as tf

# Load original Keras model
model = tf.keras.models.load_model(
    "models/plant_disease_recog_model_pwp.keras"
)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save TFLite model
with open("plant_disease.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ plant_disease.tflite model created successfully")

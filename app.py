from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("Blood Cell.h5")

# Class labels (change if your dataset order is different)
class_names = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # IMPORTANT: must match name="file" in HTML
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No file chosen"

    # Save uploaded file temporarily
    file_path = os.path.join("static", file.filename)
    file.save(file_path)

    # Preprocess image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template('result.html', 
                           prediction=predicted_class,
                           image_path=file_path)


if __name__ == "__main__":
    app.run(debug=True)

import os
from io import BytesIO
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, request
from flask import jsonify
from PIL import Image
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

MODEL_PATH = "models/model.h5"
app = Flask(__name__)


def predict_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0

    new_model = load_model(MODEL_PATH)

    predict = new_model.predict(np.array([image]))
    per = np.argmax(predict, axis=1)
    if per == 1:
        pred = 'No Diabetic Retinopathy'
    else:
        pred = 'Diabetic Retinopathy'

    return pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No image selected'})

    img_stream = BytesIO()
    image.save(img_stream)
    img_stream.seek(0)

    prediction = predict_image(Image.open(img_stream))
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
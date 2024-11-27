from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import tempfile
import json

app = Flask(__name__)

# Carregar o Modelo
model = tf.keras.models.load_model('keras/plant_disease_classifier.keras')

# Carregar os rótulos das classes
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Definindo os height e width das imagens
img_height, img_width = 150, 150

def predict_image(img_path):
    img = Image.open(img_path)
    img = img.resize((img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return class_labels[predicted_class[0]]

@app.route('/')
def index():
    if not class_labels:
        return "Class labels not found", 500
    return render_template('index.html', class_labels=class_labels)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    
    # Criar um arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = temp_file.name
        file.save(file_path)
    
    prediction = predict_image(file_path)
    
    # Remover o arquivo temporário
    os.remove(file_path)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
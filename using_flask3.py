from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Inisialisasi Flask
app = Flask(__name__)

# Memuat model
model_validasi = load_model('model_validasi_new.h5')
model_tbc = load_model('hasil_model_cnn.h5')


# Fungsi preprocessing untuk model validasi
def preprocess_image_validation(image_path, image_size=256):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype('float32') / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=-1)  # Tambahkan channel (grayscale -> 1 channel)
    image = np.expand_dims(image, axis=0)   # Tambahkan batch dimension
    return image

# Fungsi preprocessing untuk model prediksi TBC
def preprocess_image_tbc(image_path, image_size=256):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Gambar tidak ditemukan atau format tidak didukung.")
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype('float32') / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=-1)  # Tambahkan channel (grayscale -> 1 channel)
    image = np.expand_dims(image, axis=0)   # Tambahkan batch dimension
    return image

# Fungsi prediksi validasi
def predict_image_validation(image_path):
    """
    Memproses gambar dan memprediksi apakah itu gambar rontgen paru-paru atau bukan.
    """
    processed_image = preprocess_image_validation(image_path)
    if processed_image is None:
        return False, "Bukan gambar rontgen paru-paru (confidence: 0.00)"  # Penanganan jika gambar tidak valid

    prediction = model_validasi.predict(processed_image)
    confidence = prediction[0][0]

    # Interpretasi hasil prediksi validasi
    if confidence >= 0.5:
        return True, f"Gambar rontgen paru-paru (confidence: {confidence:.2f})"
    else:
        return False, f"Bukan gambar rontgen paru-paru (confidence: {confidence:.2f})"


# Fungsi prediksi TBC
def predict_tbc(image_path):
    """
    Memproses gambar rontgen paru-paru dan memprediksi apakah terkena TBC atau normal.
    """
    processed_image = preprocess_image_tbc(image_path)
    prediction = model_tbc.predict(processed_image)
    confidence = prediction[0][0]

    # Interpretasi hasil prediksi TBC
    if confidence >= 0.5:
        return f"TBC (confidence: {confidence:.2f})"
    else:
        return f"Normal (confidence: {confidence:.2f})"

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index2.html') 

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'result': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'result': 'No file selected'}), 400

    # Simpan file sementara
    file_path = os.path.join('temp', file.filename)
    os.makedirs('temp', exist_ok=True)
    file.save(file_path)

    try:
        # Validasi gambar
        is_rontgen, validation_result = predict_image_validation(file_path)
        if not is_rontgen:
            return jsonify({'result': validation_result})

        # Prediksi TBC
        tbc_result = predict_tbc(file_path)
        return jsonify({'result': tbc_result})

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)



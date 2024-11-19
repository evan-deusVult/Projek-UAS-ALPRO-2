import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Memuat model
model = load_model('hasil_model_cnn.h5')
print("Model berhasil dimuat!")

# 2. Fungsi untuk memproses gambar input
def preprocess_image(image_path, image_size=256):
    # Membaca gambar dalam mode grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Gambar tidak ditemukan atau format tidak didukung.")
    
    # Meresize gambar sesuai ukuran input model
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype('float32') / 255.0  # Normalisasi (0-1)
    image = np.expand_dims(image, axis=-1)   # Tambahkan channel grayscale
    image = np.expand_dims(image, axis=0)    # Tambahkan batch dimension
    return image

# 3. Fungsi untuk memprediksi gambar
def predict_tbc(image_path):
    try:
        # Proses gambar
        processed_image = preprocess_image(image_path)
        
        # Prediksi menggunakan model
        prediction = model.predict(processed_image)
        
        # Berikan kesimpulan berdasarkan output
        if prediction[0][0] > 0.5:
            result = "Hasil: Gambar rontgen mengindikasikan TBC."
        else:
            result = "Hasil: Gambar rontgen **tidak** mengindikasikan TBC."
        
        # Menampilkan gambar input
        plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
        plt.title(result)
        plt.axis('off')
        plt.show()
        
        return result
    except Exception as e:
        return str(e)

# 4. Menggunakan fungsi untuk memprediksi gambar
image_path = r"E:\Download\shutterstock_1659772552resizee.jpg" # Ganti dengan path gambar Anda
result = predict_tbc(image_path)
print(result)

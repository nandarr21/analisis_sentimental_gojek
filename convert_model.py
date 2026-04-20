# convert_model.py — jalankan di lokal
import tensorflow as tf

# Load model lama
model = tf.keras.models.load_model('model/lstm_model.h5')

# Simpan dalam format Keras native (lebih stabil)
model.save('model/lstm_model.keras')

print("Konversi selesai!")
import os
print(f"Ukuran .h5   : {os.path.getsize('model/lstm_model.h5')/1024/1024:.1f} MB")
print(f"Ukuran .keras: {os.path.getsize('model/lstm_model.keras')/1024/1024:.1f} MB")
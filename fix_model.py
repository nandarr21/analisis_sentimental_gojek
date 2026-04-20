# fix_model.py — update isinya
import tensorflow as tf
import os

# Load dari .keras (yang sudah ada)
model = tf.keras.models.load_model('model/lstm_model.keras')

# Simpan ulang dengan format bersih
model.save('model/lstm_model_fixed.keras')

print("Selesai!")
print(f"Ukuran: {os.path.getsize('model/lstm_model_fixed.keras')/1024/1024:.1f} MB")
print(f"TF version: {tf.__version__}")
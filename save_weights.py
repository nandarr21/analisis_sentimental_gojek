import tensorflow as tf
import numpy as np
import os

# Load dari .h5
model = tf.keras.models.load_model('model/lstm_model.h5')

# Simpan weights
weights = model.get_weights()
np.save('model/model_weights.npy', np.array(weights, dtype=object), allow_pickle=True)

print(f"Weights disimpan!")
print(f"Jumlah layer: {len(weights)}")
print(f"Ukuran: {os.path.getsize('model/model_weights.npy')/1024/1024:.1f} MB")
# save_all_json.py
import tensorflow as tf
import numpy as np
import pickle
import json
import os

# Load model
model = tf.keras.models.load_model('model/lstm_model.h5')

# Simpan weights sebagai JSON (list of lists)
weights_list = [w.tolist() for w in model.get_weights()]
with open('model/weights.json', 'w') as f:
    json.dump(weights_list, f)

# Load & simpan tokenizer sebagai JSON
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('model/tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer.to_json(), f, ensure_ascii=False)

# Load & simpan label encoder sebagai JSON
with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('model/label_encoder.json', 'w') as f:
    json.dump(le.classes_.tolist(), f)

print("Semua tersimpan sebagai JSON!")
print(f"weights.json : {os.path.getsize('model/weights.json')/1024/1024:.1f} MB")
print(f"tokenizer.json: {os.path.getsize('model/tokenizer.json')/1024:.1f} KB")
print(f"label_encoder.json: {os.path.getsize('model/label_encoder.json')} bytes")
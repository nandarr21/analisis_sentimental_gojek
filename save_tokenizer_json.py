# save_tokenizer_json.py
import pickle
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load tokenizer lama
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Simpan sebagai JSON
tokenizer_json = tokenizer.to_json()
with open('model/tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer_json, f, ensure_ascii=False)

print("Tokenizer tersimpan sebagai JSON!")
import os
print(f"Ukuran: {os.path.getsize('model/tokenizer.json')/1024:.1f} KB")
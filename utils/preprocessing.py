import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_input_json(text, tokenizer):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    return padded

# Alias untuk kompatibilitas
def preprocess_input(text, tokenizer):
    return preprocess_input_json(text, tokenizer)
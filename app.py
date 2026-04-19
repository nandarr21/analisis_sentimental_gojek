from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import os
from utils.preprocessing import preprocess_input
from utils.visualize import generate_all_visuals

app = Flask(__name__)

# ── Load model & tools ──────────────────────────────────────
model = tf.keras.models.load_model('model/lstm_model_fixed.keras')
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# ── Load dataset ─────────────────────────────────────────────
df_global = pd.read_csv('dataset/data_mentah.csv',
                        sep=';',
                        on_bad_lines='skip',
                        engine='python',
                        encoding='utf-8-sig',
                        usecols=['user', 'rating', 'review', 'at'])

# Konversi rating ke int
df_global['rating'] = pd.to_numeric(df_global['rating'], errors='coerce')
df_global = df_global.dropna(subset=['rating'])
df_global['rating'] = df_global['rating'].astype(int)
df_global = df_global.rename(columns={'at': 'timestamp'})

# Buat kolom sentimen dari rating
def rating_to_sentimen(r):
    if r >= 4:   return 'positif'
    elif r == 3: return 'netral'
    else:        return 'negatif'

df_global['sentimen'] = df_global['rating'].apply(rating_to_sentimen)
df_global['platform'] = 'Google Maps'

# Buat folder images kalau belum ada
os.makedirs('static/images', exist_ok=True) 

# ── Generate visualisasi saat startup ────────────────────────
if not os.path.exists('static/images/chart_distribusi.png'):
     print("Generating visualizations...")
     generate_all_visuals(df_global)
     print("Done!")
@app.route('/static/images/<filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)
# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    total = len(df_global)
    dist = df_global['sentimen'].value_counts().to_dict()
    return render_template('index.html', total=total, dist=dist)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    komentar = data.get('komentar', '')
    if not komentar.strip():
        return jsonify({'error': 'Komentar kosong'}), 400

    X = preprocess_input(komentar, tokenizer)
    pred = model.predict(X, verbose=0)
    label_idx = np.argmax(pred, axis=1)[0]
    label = le.classes_[label_idx]
    confidence = float(np.max(pred)) * 100
    proba = {le.classes_[i]: round(float(pred[0][i]) * 100, 1)
             for i in range(len(le.classes_))}

    return jsonify({
        'sentimen': label,
        'confidence': round(confidence, 1),
        'probabilitas': proba
    })

@app.route('/dashboard')
def dashboard():
    dist = df_global['sentimen'].value_counts().to_dict()
    platform_dist = df_global.groupby(
        ['platform', 'sentimen']).size().reset_index(
        name='count').to_dict('records')
    return render_template('dashboard.html',
                           dist=dist,
                           platform_dist=platform_dist,
                           total=len(df_global))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
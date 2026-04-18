import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

os.makedirs('model', exist_ok=True)

# ── 1. Load dataset berlabel ─────────────────────────────────
df = pd.read_csv('dataset/data_komentar.csv')
df = df.dropna(subset=['komentar', 'sentimen'])
df['komentar'] = df['komentar'].astype(str)

print("Distribusi sentimen:")
print(df['sentimen'].value_counts())

# ── 2. Balancing — oversample netral, undersample lainnya ────
target = 1500  # sampel per kelas

df_pos = df[df['sentimen']=='positif'].sample(target, random_state=42)
df_neg = df[df['sentimen']=='negatif'].sample(
    min(target, len(df[df['sentimen']=='negatif'])), random_state=42)

# Oversample netral (duplikasi karena datanya sedikit)
df_net_ori = df[df['sentimen']=='netral']
df_net = df_net_ori.sample(
    target, replace=True, random_state=42)  # replace=True untuk oversample

df_balanced = pd.concat([df_pos, df_neg, df_net]
    ).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nSetelah balancing: {len(df_balanced)} data")
print(df_balanced['sentimen'].value_counts())

# ── 3. Preprocessing ─────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_balanced['teks_bersih'] = df_balanced['komentar'].apply(clean_text)

# ── 4. Tokenisasi ────────────────────────────────────────────
MAX_WORDS = 10000
MAX_LEN   = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df_balanced['teks_bersih'])

X = pad_sequences(
    tokenizer.texts_to_sequences(df_balanced['teks_bersih']),
    maxlen=MAX_LEN, padding='post', truncating='post')

le = LabelEncoder()
y  = le.fit_transform(df_balanced['sentimen'])
print("\nLabel classes:", le.classes_)

# ── 5. Split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── 6. Class weights ─────────────────────────────────────────
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(cw))
print("Class weights:", cw_dict)

# ── 7. Model ─────────────────────────────────────────────────
model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.4),
    Bidirectional(LSTM(64)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ── 8. Training ──────────────────────────────────────────────
es = EarlyStopping(monitor='val_accuracy', patience=3,
                   restore_best_weights=True, verbose=1)

model.fit(X_train, y_train,
          epochs=15, batch_size=32,
          validation_data=(X_test, y_test),
          class_weight=cw_dict,
          callbacks=[es])

# ── 9. Evaluasi ──────────────────────────────────────────────
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\n========== HASIL EVALUASI ==========")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"{'':10}", end='')
for c in le.classes_: print(f"{c:10}", end='')
print()
for i, row in enumerate(cm):
    print(f"{le.classes_[i]:10}", end='')
    for val in row: print(f"{val:<10}", end='')
    print()

# ── 10. Simpan ───────────────────────────────────────────────
model.save('model/lstm_model.h5')
with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\nModel berhasil disimpan!")
print("Label classes:", le.classes_)
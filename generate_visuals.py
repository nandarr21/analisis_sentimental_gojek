import pandas as pd
import sys
sys.path.insert(0, '.')
from utils.visualize import generate_all_visuals
import os

df = pd.read_csv('dataset/data_mentah.csv',
                 sep=';',
                 on_bad_lines='skip',
                 engine='python',
                 encoding='utf-8-sig',
                 usecols=['user', 'rating', 'review', 'at'])

df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating'])
df['rating'] = df['rating'].astype(int)

def rating_to_sentimen(r):
    if r >= 4:   return 'positif'
    elif r == 3: return 'netral'
    else:        return 'negatif'

df['sentimen'] = df['rating'].apply(rating_to_sentimen)
df['platform'] = 'Google Maps'

# ← tambahkan baris ini:
df = df.rename(columns={'review': 'komentar', 'at': 'timestamp'})

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

print("Kolom:", df.columns.tolist())
print("Shape:", df.shape)

generate_all_visuals(df)

files = os.listdir('static/images/')
print(f"\nFile ter-generate ({len(files)}):")
for f in sorted(files):
    print(f"  ✓ {f}")
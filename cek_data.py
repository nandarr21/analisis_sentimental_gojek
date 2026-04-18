# cek_data.py
import pandas as pd

df = pd.read_csv('dataset/data_mentah.csv',
                 sep=';', on_bad_lines='skip',
                 engine='python', encoding='utf-8-sig',
                 usecols=['rating', 'review'])

df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna()

# Lihat contoh review rating 1-2 (harusnya negatif)
print("=== RATING 1-2 (Negatif) ===")
print(df[df['rating'] <= 2]['review'].head(10).to_string())

print("\n=== RATING 5 (Positif) ===")
print(df[df['rating'] == 5]['review'].head(10).to_string())

# Cari kata negatif di rating tinggi
kata_negatif = ['buruk','jelek','lambat','lama','kecewa','bohong',
                'tipu','mahal','rugi','batal','cancel','hilang']
mask = df['review'].str.lower().str.contains('|'.join(kata_negatif), na=False)
print(f"\nReview berisi kata negatif tapi rating 4-5: {len(df[mask & (df['rating']>=4)])}")
print(df[mask & (df['rating']>=4)][['rating','review']].head(5).to_string())
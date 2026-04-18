import pandas as pd
import re

df = pd.read_csv('dataset/data_mentah.csv',
                 sep=';', on_bad_lines='skip',
                 engine='python', encoding='utf-8-sig',
                 usecols=['user', 'rating', 'review', 'at'])

df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating', 'review'])
df['rating'] = df['rating'].astype(int)
df['review'] = df['review'].astype(str)

# ── Lexicon bahasa Indonesia ──────────────────────────────────
kata_positif = [
    'bagus','baik','mantap','ok','oke','memuaskan','puas','cepat',
    'ramah','nyaman','aman','josss','jos','top','keren','hebat',
    'recommended','rekomen','senang','suka','terbaik','best',
    'profesional','tepat','sesuai','mudah','membantu','murah',
    'canggih','kece','gampang','praktis','worth','mantab','oke banget',
    'terima kasih','makasih','alhamdulillah','bagus banget','luar biasa',
    'sangat membantu','sangat baik','sangat puas','terselamatkan',
    'good','great','awesome','helpful','excellent','perfect','fast'
]

kata_negatif = [
    'buruk','jelek','lambat','lama','kecewa','bohong','tipu',
    'mahal','rugi','batal','cancel','hilang','rusak','gagal',
    'error','tidak bisa','gak bisa','ga bisa','susah','ribet','parah',
    'nyebelin','kesal','marah','komplain','sedih','bingung','zonk',
    'scam','penipuan','aneh','curang','ngeluh','masalah','gangguan',
    'lelet','lemot','nunggu lama','gak jelas','ga jelas','kacau',
    'mengecewakan','tidak memuaskan','tidak sesuai','tidak aman',
    'gausah','potongan','keluhan','tolong','minta tolong','harap',
    'tidak responsif','lama banget','kelamaan','nungguin','nunggu',
    'dibatalkan','driver cancel','orderan hilang','uang hilang',
    'ditipu','tertipu','slow','worst','bad','terrible','awful'
]

kata_netral = [
    'biasa','lumayan','cukup','standar','ya begitu','gitu aja',
    'tidak terlalu','kurang lebih','so so','biasa saja'
]

def hitung_skor(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    skor_pos = sum(1 for k in kata_positif if k in text)
    skor_neg = sum(1 for k in kata_negatif if k in text)
    skor_net = sum(1 for k in kata_netral if k in text)
    
    return skor_pos, skor_neg, skor_net

def smart_label(row):
    text  = str(row['review'])
    rating = row['rating']
    
    skor_pos, skor_neg, skor_net = hitung_skor(text)
    selisih = skor_pos - skor_neg
    
    # Teks sangat jelas negatif
    if skor_neg >= 2 and skor_neg > skor_pos:
        return 'negatif'
    
    # Teks sangat jelas positif
    if skor_pos >= 2 and skor_pos > skor_neg:
        return 'positif'
    
    # Teks netral eksplisit
    if skor_net > 0 and skor_pos == 0 and skor_neg == 0:
        return 'netral'
    
    # Teks ambigu — gabungkan sinyal rating + teks
    if selisih > 0 and rating >= 3:
        return 'positif'
    elif selisih < 0 or rating <= 2:
        return 'negatif'
    elif rating == 3:
        return 'netral'
    elif rating >= 4:
        return 'positif'
    else:
        return 'negatif'

print("Melabel ulang dataset...")
df['sentimen'] = df.apply(smart_label, axis=1)

print("\nDistribusi sentimen (label baru):")
print(df['sentimen'].value_counts())

# Verifikasi sampel
print("\n--- Contoh POSITIF ---")
print(df[df['sentimen']=='positif'][['review','rating']].sample(5).to_string())
print("\n--- Contoh NEGATIF ---")
print(df[df['sentimen']=='negatif'][['review','rating']].sample(5).to_string())
print("\n--- Contoh NETRAL ---")
print(df[df['sentimen']=='netral'][['review','rating']].sample(5).to_string())

# Simpan dataset berlabel baru
df_out = df.rename(columns={'review':'komentar', 'at':'timestamp'})
df_out['platform'] = 'Google Maps'
df_out.to_csv('dataset/data_komentar.csv', index=False)
print(f"\nDataset tersimpan: dataset/data_komentar.csv ({len(df_out)} baris)")
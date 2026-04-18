import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import os

IMG_DIR = 'static/images'
os.makedirs(IMG_DIR, exist_ok=True)

def generate_all_visuals(df):
    print("  → chart distribusi...")
    _chart_distribusi(df)
    print("  → chart rating...")
    _chart_per_platform(df)
    print("  → chart per hari...")
    _chart_per_hari(df)
    print("  → chart per jam...")
    _chart_per_jam(df)
    print("  → wordcloud...")
    _wordcloud_all(df)
    print("  Semua selesai!")

def _chart_distribusi(df):
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = {'positif':'#4CAF50','negatif':'#F44336','netral':'#2196F3'}
        counts = df['sentimen'].value_counts()
        bars = ax.bar(counts.index, counts.values,
                      color=[colors.get(k,'gray') for k in counts.index],
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+50,
                    f'{val:,}', ha='center', va='bottom', fontsize=10)
        ax.set_title('Distribusi Sentimen Komentar', fontsize=13, pad=10)
        ax.set_xlabel('Sentimen')
        ax.set_ylabel('Jumlah Komentar')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'{IMG_DIR}/chart_distribusi.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ chart_distribusi.png")
    except Exception as e:
        print(f"    ✗ chart_distribusi ERROR: {e}")

def _chart_per_platform(df):
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        if 'rating' not in df.columns:
            print("    ✗ kolom rating tidak ada, skip")
            return
        rating_dist = df['rating'].value_counts().sort_index()
        colors = ['#F44336','#FF9800','#FFC107','#8BC34A','#4CAF50']
        bars = ax.bar(rating_dist.index.astype(str), rating_dist.values,
                      color=colors[:len(rating_dist)], edgecolor='white')
        for bar, val in zip(bars, rating_dist.values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+20,
                    f'{val:,}', ha='center', va='bottom', fontsize=10)
        ax.set_title('Distribusi Rating Pengguna', fontsize=13, pad=10)
        ax.set_xlabel('Rating (bintang)')
        ax.set_ylabel('Jumlah Komentar')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'{IMG_DIR}/chart_platform.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ chart_platform.png")
    except Exception as e:
        print(f"    ✗ chart_platform ERROR: {e}")

def _chart_per_hari(df):
    try:
        if 'timestamp' not in df.columns:
            print("    ✗ kolom timestamp tidak ada, skip")
            return
        df2 = df.copy()
        df2['timestamp'] = pd.to_datetime(df2['timestamp'], errors='coerce')
        df2 = df2.dropna(subset=['timestamp'])
        if len(df2) == 0:
            print("    ✗ timestamp semua NaT, skip")
            return
        df2['tanggal'] = df2['timestamp'].dt.date
        daily = df2.groupby(['tanggal','sentimen']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 4))
        colors_map = {'positif':'#4CAF50','negatif':'#F44336','netral':'#2196F3'}
        for col in daily.columns:
            ax.plot(daily.index, daily[col],
                    marker='o', linewidth=1.5,
                    label=col, color=colors_map.get(col,'gray'))
        ax.set_title('Aktivitas Komentar per Hari', fontsize=13, pad=10)
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Jumlah')
        ax.legend()
        ax.spines[['top','right']].set_visible(False)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{IMG_DIR}/chart_hari.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ chart_hari.png")
    except Exception as e:
        print(f"    ✗ chart_hari ERROR: {e}")

def _chart_per_jam(df):
    try:
        if 'timestamp' not in df.columns:
            print("    ✗ kolom timestamp tidak ada, skip")
            return
        df2 = df.copy()
        df2['timestamp'] = pd.to_datetime(df2['timestamp'], errors='coerce')
        df2 = df2.dropna(subset=['timestamp'])
        if len(df2) == 0:
            print("    ✗ timestamp semua NaT, skip")
            return
        df2['jam'] = df2['timestamp'].dt.hour
        hourly = df2.groupby('jam').size()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(hourly.index, hourly.values, alpha=0.3, color='#2E7D32')
        ax.plot(hourly.index, hourly.values, color='#2E7D32', linewidth=2)
        ax.set_title('Aktivitas Komentar per Jam', fontsize=13, pad=10)
        ax.set_xlabel('Jam')
        ax.set_ylabel('Jumlah')
        ax.set_xticks(range(0, 24))
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'{IMG_DIR}/chart_jam.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ chart_jam.png")
    except Exception as e:
        print(f"    ✗ chart_jam ERROR: {e}")

def _wordcloud_all(df):
    for sentimen, color in [
        ('positif', 'Greens'),
        ('negatif', 'Reds'),
        ('netral',  'Blues')
    ]:
        try:
            # Coba kolom 'komentar' dulu, fallback ke 'review'
            col = 'komentar' if 'komentar' in df.columns else 'review'
            subset = df[df['sentimen'] == sentimen][col].dropna()
            if len(subset) == 0:
                print(f"    ✗ wordcloud_{sentimen}: data kosong")
                continue
            teks = ' '.join(subset.astype(str).values)
            wc = WordCloud(width=800, height=400,
                           background_color='white',
                           colormap=color,
                           max_words=100,
                           collocations=False).generate(teks)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'WordCloud Komentar {sentimen.capitalize()}',
                      fontsize=13, pad=8)
            plt.tight_layout()
            plt.savefig(f'{IMG_DIR}/wordcloud_{sentimen}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    ✓ wordcloud_{sentimen}.png")
        except Exception as e:
            print(f"    ✗ wordcloud_{sentimen} ERROR: {e}")
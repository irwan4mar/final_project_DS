import pandas as pd
from datetime import datetime

# ===========================
# Modul Analisis & Rekomendasi AI
# ===========================

def generate_ai_recommendations(df: pd.DataFrame):
    """
    Menganalisis data mutasi barang dan memberikan rekomendasi berbasis pola historis.
    """
    recommendations = []

    if df.empty:
        return ["⚠️ Tidak ada data untuk dianalisis."]

    # Pastikan kolom-kolom utama ada
    required_columns = ["uraian_barang", "item_keluar", "tanggal_keluar"]
    if not all(col in df.columns for col in required_columns):
        return ["⚠️ Format data tidak lengkap. Kolom harus mencakup: uraian_barang, item_keluar, tanggal_keluar."]

    # Konversi tanggal
    df["tanggal_keluar"] = pd.to_datetime(df["tanggal_keluar"], errors="coerce")

    # 1️⃣ Analisis barang yang paling sering keluar
    top_items = df.groupby("uraian_barang")["item_keluar"].sum().sort_values(ascending=False)
    if not top_items.empty:
        top_name = top_items.index[0]
        top_value = top_items.iloc[0]
        recommendations.append(f"📦 Barang '{top_name}' memiliki jumlah keluar tertinggi ({top_value} item). Pastikan stok cadangan cukup.")

    # 2️⃣ Barang yang jarang keluar
    low_items = top_items[top_items < top_items.mean()]
    if not low_items.empty:
        low_list = ", ".join(low_items.index[:3])
        recommendations.append(f"🧊 Barang yang jarang keluar: {low_list}. Pertimbangkan untuk evaluasi stok agar tidak menumpuk.")

    # 3️⃣ Barang yang lama tidak keluar
    latest_dates = df.groupby("uraian_barang")["tanggal_keluar"].max().sort_values()
    old_items = latest_dates[latest_dates < datetime.now() - pd.Timedelta(days=60)]
    if not old_items.empty:
        old_list = ", ".join(old_items.index[:3])
        recommendations.append(f"⏰ Barang yang sudah >60 hari tidak keluar: {old_list}. Perlu pemeriksaan fisik di gudang.")

    # 4️⃣ Prediksi sederhana (trend)
    df["month"] = df["tanggal_keluar"].dt.to_period("M")
    trend = df.groupby(["month", "uraian_barang"])["item_keluar"].sum().reset_index()
    if not trend.empty:
        latest_month = trend["month"].max()
        last_month_data = trend[trend["month"] == latest_month]
        most_trending = last_month_data.sort_values("item_keluar", ascending=False).head(1)
        if not most_trending.empty:
            trending_item = most_trending["uraian_barang"].iloc[0]
            recommendations.append(f"📈 Tren terkini menunjukkan peningkatan permintaan untuk '{trending_item}'. Siapkan stok tambahan bulan depan.")

    # 5️⃣ Tambahan insight umum
    total_items = df["uraian_barang"].nunique()
    total_qty = df["item_keluar"].sum()
    recommendations.append(f"📊 Total {total_items} jenis barang tercatat keluar dengan total {total_qty} unit.")

    return recommendations

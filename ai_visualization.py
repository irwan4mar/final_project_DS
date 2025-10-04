import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ==============================
# Modul Visualisasi Tren Barang
# ==============================

def plot_monthly_trend(df: pd.DataFrame):
    """
    Menampilkan tren total barang keluar per bulan.
    """
    if df.empty:
        st.warning("Tidak ada data untuk divisualisasikan.")
        return

    df["tanggal_keluar"] = pd.to_datetime(df["tanggal_keluar"], errors="coerce")
    df["bulan"] = df["tanggal_keluar"].dt.to_period("M")

    trend = df.groupby("bulan")["item_keluar"].sum().reset_index()
    trend["bulan"] = trend["bulan"].astype(str)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trend["bulan"], trend["item_keluar"], marker="o")
    ax.set_title("📈 Tren Total Barang Keluar per Bulan", fontsize=12)
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Jumlah Barang Keluar")
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)


def plot_item_distribution(df: pd.DataFrame):
    """
    Menampilkan distribusi barang keluar berdasarkan total item.
    """
    if df.empty:
        st.warning("Tidak ada data untuk divisualisasikan.")
        return

    item_sum = df.groupby("uraian_barang")["item_keluar"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(item_sum.index, item_sum.values, color="skyblue")
    ax.set_title("📊 Distribusi Barang Keluar Berdasarkan Jenis Barang", fontsize=12)
    ax.set_ylabel("Jumlah Barang Keluar")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


def plot_trend_by_item(df: pd.DataFrame, selected_item: str):
    """
    Menampilkan tren per barang tertentu.
    """
    if df.empty:
        st.warning("Tidak ada data untuk divisualisasikan.")
        return

    df["tanggal_keluar"] = pd.to_datetime(df["tanggal_keluar"], errors="coerce")
    df["bulan"] = df["tanggal_keluar"].dt.to_period("M")

    df_item = df[df["uraian_barang"] == selected_item]
    if df_item.empty:
        st.warning(f"Tidak ditemukan data untuk '{selected_item}'.")
        return

    trend_item = df_item.groupby("bulan")["item_keluar"].sum().reset_index()
    trend_item["bulan"] = trend_item["bulan"].astype(str)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trend_item["bulan"], trend_item["item_keluar"], marker="o", color="orange")
    ax.set_title(f"📈 Tren Barang Keluar: {selected_item}", fontsize=12)
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Jumlah Barang Keluar")
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)

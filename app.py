import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# 1️⃣ KONFIGURASI HALAMAN
# ------------------------------
st.set_page_config(page_title="Dashboard E-Commerce Penjualan Handphone", layout="wide")

# ------------------------------
# 2️⃣ BACA DATA
# ------------------------------
data = pd.read_excel("handphone_clean_fixed.xlsx")
data["Tanggal"] = pd.to_datetime(data["Tanggal"], errors="coerce")
data = data.dropna(subset=["Tanggal"]).sort_values("Tanggal")

# Tambahkan kolom wilayah (simulasi)
import random
wilayah_list = ["Jakarta", "Bandung", "Surabaya", "Medan", "Yogyakarta"]
data["Wilayah"] = [random.choice(wilayah_list) for _ in range(len(data))]

# ------------------------------
# 3️⃣ SIDEBAR NAVIGASI
# ------------------------------
st.sidebar.title("📂 Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", 
                        ["Dashboard Penjualan", 
                         "Analisis Produk", 
                         "Prediksi Demand", 
                         "Rekomendasi Promosi",
                         "Insight Otomatis"])

# ------------------------------
# 4️⃣ DASHBOARD PENJUALAN
# ------------------------------
if menu == "Dashboard Penjualan":
    st.title("📊 Dashboard Penjualan Umum")
    st.markdown("Menampilkan tren penjualan keseluruhan dan distribusi penjualan berdasarkan wilayah.")

    # Grafik total penjualan per tanggal
    fig_total = px.line(data, x="Tanggal", y="Penjualan", color="Nama_Produk",
                        title="📆 Tren Penjualan Harian per Produk", markers=True)
    st.plotly_chart(fig_total, use_container_width=True)

    # Top 5 produk terlaris
    top_produk = data.groupby("Nama_Produk")["Penjualan"].sum().nlargest(5).reset_index()
    fig_top = px.bar(top_produk, x="Nama_Produk", y="Penjualan",
                     title="🏆 Top 5 Produk Terlaris", color="Nama_Produk", text="Penjualan")
    fig_top.update_traces(textposition="outside")
    st.plotly_chart(fig_top, use_container_width=True)

    # Distribusi penjualan per wilayah
    wilayah_penjualan = data.groupby("Wilayah")["Penjualan"].sum().reset_index()
    fig_wilayah = px.pie(wilayah_penjualan, names="Wilayah", values="Penjualan",
                         title="🌍 Distribusi Penjualan Berdasarkan Wilayah")
    st.plotly_chart(fig_wilayah, use_container_width=True)

# ------------------------------
# 5️⃣ ANALISIS PRODUK
# ------------------------------
elif menu == "Analisis Produk":
    st.title("🔍 Analisis Penjualan per Produk")
    produk = st.selectbox("Pilih Produk:", data["Nama_Produk"].unique())
    df_produk = data[data["Nama_Produk"] == produk]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Penjualan", f"{df_produk['Penjualan'].sum()} unit")
    col2.metric("Rata-rata Harian", f"{df_produk['Penjualan'].mean():.2f} unit/hari")
    col3.metric("Wilayah Terbanyak", df_produk["Wilayah"].mode()[0])

    fig_tren = px.line(df_produk, x="Tanggal", y="Penjualan", markers=True,
                       title=f"📈 Tren Penjualan {produk} per Hari")
    st.plotly_chart(fig_tren, use_container_width=True)

    st.subheader("📦 Data Penjualan Produk")
    st.dataframe(df_produk, use_container_width=True)

# ------------------------------
# 6️⃣ PREDIKSI DEMAND
# ------------------------------
elif menu == "Prediksi Demand":
    st.title("📈 Prediksi Demand (7 Hari ke Depan)")
    st.markdown("Prediksi dilakukan menggunakan model ARIMA berdasarkan data historis penjualan per produk.")

    produk = st.selectbox("Pilih Produk:", data["Nama_Produk"].unique())
    df_produk = data[data["Nama_Produk"] == produk]

    try:
        model = ARIMA(df_produk["Penjualan"], order=(1, 1, 1))
        fit = model.fit()
        forecast = fit.forecast(steps=7)
        forecast_df = pd.DataFrame({
            "Tanggal": pd.date_range(df_produk["Tanggal"].max() + pd.Timedelta(days=1), periods=7),
            "Prediksi_Penjualan": forecast
        })

        st.subheader(f"📅 Prediksi Penjualan 7 Hari ke Depan untuk {produk}")
        fig_forecast = px.line(forecast_df, x="Tanggal", y="Prediksi_Penjualan", markers=True,
                               title="📊 Hasil Prediksi Demand")
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"Model gagal dijalankan: {e}")

# ------------------------------
# 7️⃣ REKOMENDASI PROMOSI
# ------------------------------
elif menu == "Rekomendasi Promosi":
    st.title("💡 Rekomendasi Promosi Produk")
    st.markdown("Analisis efektivitas promosi terhadap peningkatan penjualan.")

    promo_avg = data[data["Promosi"] == 1]["Penjualan"].mean()
    non_promo_avg = data[data["Promosi"] == 0]["Penjualan"].mean()

    col1, col2 = st.columns(2)
    col1.metric("Rata-rata Saat Promosi", f"{promo_avg:.2f} unit")
    col2.metric("Rata-rata Tanpa Promosi", f"{non_promo_avg:.2f} unit")

    if promo_avg > non_promo_avg:
        st.success("✅ Promosi efektif! Pertahankan atau perluas strategi promosi.")
    else:
        st.warning("⚠️ Promosi belum berdampak signifikan, perlu evaluasi strategi.")

    promo_df = data.groupby(["Nama_Produk", "Promosi"])["Penjualan"].mean().reset_index()
    fig_promo = px.bar(promo_df, x="Nama_Produk", y="Penjualan", color="Promosi",
                       barmode="group", title="📊 Rata-rata Penjualan Saat Promosi vs Tidak Promosi")
    st.plotly_chart(fig_promo, use_container_width=True)

# ------------------------------
# 8️⃣ INSIGHT OTOMATIS (DITINGKATKAN)
# ------------------------------
elif menu == "Insight Otomatis":
    st.title("🤖 Insight Otomatis Mingguan")
    st.markdown("Sistem memberikan insight otomatis berdasarkan data 7 hari terakhir.")

    cutoff = data["Tanggal"].max() - timedelta(days=7)
    recent_data = data[data["Tanggal"] >= cutoff]

    if len(recent_data) > 0:
        st.markdown(f"📅 Data diambil dari periode **{cutoff.date()} hingga {data['Tanggal'].max().date()}**")

        # Insight utama
        top_produk = recent_data.groupby("Nama_Produk")["Penjualan"].sum().nlargest(1).index[0]
        total_penjualan = recent_data["Penjualan"].sum()
        wilayah_terbaik = recent_data.groupby("Wilayah")["Penjualan"].sum().idxmax()

        st.success(f"🔥 Dalam 7 hari terakhir, produk **{top_produk}** menjadi yang paling laris!")
        st.info(f"Total penjualan minggu ini mencapai **{total_penjualan} unit**, "
                f"dengan wilayah terbaik: **{wilayah_terbaik}**.")

        # Tambahan insight visual
        top5_recent = recent_data.groupby("Nama_Produk")["Penjualan"].sum().nlargest(5).reset_index()
        fig_bar = px.bar(top5_recent, x="Nama_Produk", y="Penjualan", color="Nama_Produk",
                         title="🏆 Top 5 Produk Terlaris Minggu Ini", text="Penjualan")
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Tren harian minggu terakhir
        fig_trend = px.line(recent_data, x="Tanggal", y="Penjualan", color="Nama_Produk",
                            title="📈 Tren Penjualan 7 Hari Terakhir", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("❌ Data 7 hari terakhir belum tersedia di dataset.")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("🛍️ Dibuat menggunakan Streamlit & ARIMA | © 2025 Kevin Saputra Rustian - 411222041")

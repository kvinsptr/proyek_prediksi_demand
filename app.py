# ====================================================
# 📱 DASHBOARD PENJUALAN SMARTPHONE - STREAMLIT APP (FINAL)
# ====================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")

# ====================================================
# 1️⃣ KONFIGURASI HALAMAN
# ====================================================
st.set_page_config(
    page_title="📱 Dashboard Penjualan Smartphone",
    layout="wide",
    page_icon="📊"
)

# ====================================================
# 2️⃣ MEMBACA DATA
# ====================================================
@st.cache_data
def load_data():
    data = pd.read_csv("handphone_smartphone_dataset.csv")
    data["Tanggal"] = pd.to_datetime(data["Tanggal"], errors="coerce")
    data = data.dropna(subset=["Tanggal"]).sort_values("Tanggal")
    return data

data = load_data()

# ====================================================
# 3️⃣ PALET WARNA CERAH
# ====================================================
COLOR_BRIGHT = px.colors.qualitative.Bold  # merah, biru, hijau, oranye, ungu, kuning

# ====================================================
# 4️⃣ SIDEBAR NAVIGASI
# ====================================================
st.sidebar.header("📂 Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "Dashboard Penjualan",
        "Analisis Produk",
        "Prediksi Demand",
        "Rekomendasi Promosi",
        "Insight Otomatis"
    ]
)

# ====================================================
# 5️⃣ DASHBOARD PENJUALAN
# ====================================================
if menu == "Dashboard Penjualan":
    st.title("📊 Dashboard Penjualan Smartphone")
    st.markdown("Menampilkan ringkasan penjualan, tren harian, dan distribusi berdasarkan wilayah dan produk.")

    # --- KPI SUMMARY ---
    total_penjualan = int(data["Penjualan"].sum())
    rata_harian = round(data.groupby("Tanggal")["Penjualan"].sum().mean(), 2)
    produk_teratas = data.groupby("Nama_Produk")["Penjualan"].sum().idxmax()
    wilayah_terbanyak = data.groupby("Wilayah")["Penjualan"].sum().idxmax()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Total Penjualan", f"{total_penjualan} unit")
    col2.metric("📅 Rata-rata Harian", f"{rata_harian} unit")
    col3.metric("🏆 Produk Terlaris", produk_teratas)
    col4.metric("🌍 Wilayah Tertinggi", wilayah_terbanyak)

    st.markdown("---")

    # --- Pilihan Produk ---
    produk_pilihan = st.multiselect(
        "Pilih produk untuk ditampilkan di grafik tren:",
        options=sorted(data["Nama_Produk"].unique()),
        default=sorted(data["Nama_Produk"].unique())[:3]
    )
    df_filtered = data[data["Nama_Produk"].isin(produk_pilihan)]

    # --- Tren Penjualan Harian ---
    fig_tren = px.line(
        df_filtered,
        x="Tanggal",
        y="Penjualan",
        color="Nama_Produk",
        markers=True,
        line_shape="spline",
        title="📆 Tren Penjualan Harian per Produk",
        color_discrete_sequence=COLOR_BRIGHT,
        template="plotly_white"
    )
    fig_tren.update_layout(legend_title="Produk", hovermode="x unified")
    st.plotly_chart(fig_tren, use_container_width=True)

    # --- Top 5 Produk Terlaris ---
    top_produk = data.groupby("Nama_Produk")["Penjualan"].sum().nlargest(5).reset_index()
    fig_top = px.bar(
        top_produk,
        x="Nama_Produk",
        y="Penjualan",
        color="Nama_Produk",
        text="Penjualan",
        title="🏅 Top 5 Produk Terlaris",
        color_discrete_sequence=COLOR_BRIGHT,
        template="plotly_white"
    )
    fig_top.update_traces(textposition="outside")
    st.plotly_chart(fig_top, use_container_width=True)

    # --- Distribusi Penjualan per Wilayah ---
    wilayah_df = data.groupby("Wilayah")["Penjualan"].sum().reset_index().sort_values("Penjualan", ascending=True)
    fig_wilayah = px.bar(
        wilayah_df,
        x="Penjualan",
        y="Wilayah",
        orientation="h",
        text="Penjualan",
        color="Wilayah",
        color_discrete_sequence=COLOR_BRIGHT,
        title="🌍 Distribusi Penjualan Berdasarkan Wilayah",
        template="plotly_white"
    )
    fig_wilayah.update_traces(textposition="outside")
    st.plotly_chart(fig_wilayah, use_container_width=True)

# ====================================================
# 6️⃣ ANALISIS PRODUK
# ====================================================
elif menu == "Analisis Produk":
    st.title("🔍 Analisis Penjualan per Produk")

    produk = st.selectbox("Pilih Produk:", sorted(data["Nama_Produk"].unique()))
    df_produk = data[data["Nama_Produk"] == produk]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Penjualan", f"{df_produk['Penjualan'].sum()} unit")
    col2.metric("Rata-rata Harian", f"{df_produk['Penjualan'].mean():.2f} unit/hari")
    col3.metric("Wilayah Terbanyak", df_produk["Wilayah"].mode()[0])

    fig_produk = px.area(
        df_produk,
        x="Tanggal",
        y="Penjualan",
        title=f"📈 Tren Penjualan {produk} per Hari",
        color_discrete_sequence=COLOR_BRIGHT,
        template="plotly_white"
    )
    fig_produk.update_traces(mode="lines+markers")
    st.plotly_chart(fig_produk, use_container_width=True)
    st.dataframe(df_produk, use_container_width=True)

# ====================================================
# 7️⃣ PREDIKSI DEMAND
# ====================================================
elif menu == "Prediksi Demand":
    st.title("📈 Prediksi Demand (7 Hari ke Depan)")
    st.markdown("Prediksi dilakukan menggunakan model **ARIMA** berdasarkan data historis penjualan per produk.")

    produk = st.selectbox("Pilih Produk:", sorted(data["Nama_Produk"].unique()))
    df_produk = data[data["Nama_Produk"] == produk].copy()

    try:
        model = ARIMA(df_produk["Penjualan"], order=(1, 1, 1))
        fit = model.fit()
        forecast = fit.forecast(steps=7)

        forecast_df = pd.DataFrame({
            "Tanggal": pd.date_range(df_produk["Tanggal"].max() + timedelta(days=1), periods=7),
            "Prediksi_Penjualan": forecast
        })

        fig_forecast = px.line(
            forecast_df,
            x="Tanggal",
            y="Prediksi_Penjualan",
            markers=True,
            line_shape="spline",
            title=f"📊 Prediksi Penjualan 7 Hari ke Depan untuk {produk}",
            color_discrete_sequence=["#E74C3C"],
            template="plotly_white"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"Gagal membuat prediksi: {e}")

# ====================================================
# 8️⃣ REKOMENDASI PROMOSI
# ====================================================
elif menu == "Rekomendasi Promosi":
    st.title("💡 Rekomendasi Promosi")
    st.markdown("Analisis efektivitas promosi terhadap peningkatan penjualan.")

    promo_avg = data[data["Promosi"] == 1]["Penjualan"].mean()
    non_promo_avg = data[data["Promosi"] == 0]["Penjualan"].mean()

    col1, col2 = st.columns(2)
    col1.metric("📈 Saat Promosi", f"{promo_avg:.2f} unit")
    col2.metric("📉 Tanpa Promosi", f"{non_promo_avg:.2f} unit")

    if promo_avg > non_promo_avg:
        st.success("✅ Promosi terbukti meningkatkan penjualan!")
    else:
        st.warning("⚠️ Promosi belum berdampak signifikan.")

    promo_df = data.groupby(["Nama_Produk", "Promosi"])["Penjualan"].mean().reset_index()
    fig_promo = px.bar(
        promo_df,
        x="Nama_Produk",
        y="Penjualan",
        color="Promosi",
        barmode="group",
        title="📊 Rata-rata Penjualan: Promosi vs Non-Promosi",
        color_discrete_sequence=COLOR_BRIGHT,
        template="plotly_white"
    )
    st.plotly_chart(fig_promo, use_container_width=True)

# ====================================================
# 9️⃣ INSIGHT OTOMATIS (DIPERKUAT)
# ====================================================
elif menu == "Insight Otomatis":
    st.title("🤖 Insight Otomatis Mingguan")

    cutoff = data["Tanggal"].max() - timedelta(days=7)
    prev_cutoff = cutoff - timedelta(days=7)
    recent = data[data["Tanggal"] >= cutoff]
    previous = data[(data["Tanggal"] < cutoff) & (data["Tanggal"] >= prev_cutoff)]

    if len(recent) > 0:
        top_produk = recent.groupby("Nama_Produk")["Penjualan"].sum().idxmax()
        wilayah_top = recent.groupby("Wilayah")["Penjualan"].sum().idxmax()
        total = recent["Penjualan"].sum()
        total_prev = previous["Penjualan"].sum() if len(previous) > 0 else 0

        delta = total - total_prev
        persentase = (delta / total_prev * 100) if total_prev > 0 else 0

        col1, col2 = st.columns(2)
        col1.success(f"🔥 Produk paling laris minggu ini: **{top_produk}**")
        col2.info(f"🌍 Wilayah penjualan tertinggi: **{wilayah_top}**")
        st.metric("📈 Total Penjualan Minggu Ini", f"{total} unit", f"{persentase:+.2f}% dibanding minggu lalu")

        # Grafik Top 5 Produk Minggu Ini
        top5 = recent.groupby("Nama_Produk")["Penjualan"].sum().nlargest(5).reset_index()
        fig_top5 = px.bar(
            top5,
            y="Nama_Produk",
            x="Penjualan",
            orientation="h",
            text="Penjualan",
            color="Nama_Produk",
            title="🏆 Top 5 Produk Terlaris Minggu Ini",
            color_discrete_sequence=COLOR_BRIGHT,
            template="plotly_white"
        )
        fig_top5.update_traces(textposition="outside")
        st.plotly_chart(fig_top5, use_container_width=True)

        # Tren 7 Hari Terakhir
        trend7 = recent.groupby("Tanggal")["Penjualan"].sum().reset_index()
        fig_trend = px.line(
            trend7,
            x="Tanggal",
            y="Penjualan",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["#3498DB"],
            title="📅 Tren Total Penjualan 7 Hari Terakhir (Semua Produk)",
            template="plotly_white"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("❌ Data 7 hari terakhir belum tersedia.")

# ====================================================
# 📋 FOOTER
# ====================================================
st.markdown("---")
st.caption("📱 Dashboard Penjualan Smartphone | © 2025 Kevin Saputra Rustian")

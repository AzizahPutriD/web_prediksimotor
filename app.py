import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import io
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
import warnings

# ===== ARIMA GRID SEARCH FUNCTION =====
def find_best_arima_model(ts_log, max_p=3, max_d=2, max_q=3):
    warnings.filterwarnings("ignore")
    best_aic = float("inf")
    best_order = None
    best_model = None

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(ts_log, order=(p, d, q)).fit()
                    aic = model.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = model
                except:
                    continue

    return best_model, best_order

# ===== DATABASE CONFIGURATION =====
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tipe_motor TEXT,
            bulan TEXT,
            prediksi_penjualan INTEGER,
            waktu_prediksi TEXT
        )
    ''')
    conn.commit()
    conn.close()

# ===== SIMPAN HASIL PREDIKSI =====
def save_predictions(predictions):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    for index, row in predictions.iterrows():
        c.execute("INSERT INTO predictions (tipe_motor, bulan, prediksi_penjualan, waktu_prediksi) VALUES (?, ?, ?, ?)",
                  (row['Tipe Motor'], row['Bulan'].strftime('%Y-%m'), row['Prediksi Penjualan'], str(pd.Timestamp.now())))
    conn.commit()
    conn.close()

# ===== HAPUS PREDIKSI =====
def delete_prediction(prediction_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM predictions WHERE id=?", (prediction_id,))
    conn.commit()
    conn.close()

# ===== TAMPILKAN HASIL UNTUK VIEWER =====
def display_predictions():
    conn = sqlite3.connect('users.db')
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    return df

# ===== LOGIN PAGE =====
def login():
    st.markdown("<h1 style='text-align: center;'>📈 Sistem Prediksi Penjualan Sepeda Motor Honda</h1>", unsafe_allow_html=True)
    with st.form("login_form"):
        st.markdown("<h2 style='text-align: center;'>🔐 Login</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Silakan masukkan username & password untuk mengakses aplikasi.</p>", unsafe_allow_html=True)
        st.markdown("---")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            admin_username = st.secrets["login"]["admin_username"]
            admin_password = st.secrets["login"]["admin_password"]
            viewer_username = st.secrets["login"]["viewer_username"]
            viewer_password = st.secrets["login"]["viewer_password"]
            
            if username == admin_username and password == admin_password:
                st.session_state.logged_in = True
                st.session_state.role = "admin"
            elif username == viewer_username and password == viewer_password:
                st.session_state.logged_in = True
                st.session_state.role = "viewer"
            else:
                st.error("❌ Username atau password salah.")

# ===== LOGOUT FUNCTION =====
def logout():
    st.session_state.logged_in = False
    st.session_state.role = None
    st.rerun()

# ===== KONVERSI BULAN =====
def ubah_bulan(bulan_str):
    mapping = {
        'Jan': 'Jan', 'Feb': 'Feb', 'Mar': 'Mar', 'Apr': 'Apr', 'Mei': 'May',
        'Jun': 'Jun', 'Jul': 'Jul', 'Agu': 'Aug', 'Sep': 'Sep',
        'Okt': 'Oct', 'Nov': 'Nov', 'Des': 'Dec'
    }
    for indo, eng in mapping.items():
        bulan_str = bulan_str.replace(indo, eng)
    return bulan_str

# ===== MAIN APP =====
def main():
    st.markdown("<h1 style='text-align: center;'>📈 Prediksi Penjualan Sepeda Motor Honda</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Gunakan aplikasi ini untuk memprediksi penjualan sepeda motor berdasarkan tipe dan periode waktu.</p>", unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.role == 'admin':
        st.sidebar.image("logo.png", width=300)
        st.sidebar.header("📂 Upload Data Penjualan")
        uploaded_file = st.sidebar.file_uploader("Unggah file CSV", type=["csv"])

        if uploaded_file is None:
            st.warning("Silakan unggah file data penjualan (CSV).")
            return

        try:
            df = pd.read_csv(uploaded_file, delimiter=';')
            df.columns = [col.lower().strip() for col in df.columns]

            required_columns = {'bulan', 'tipe_motor', 'jumlah'}
            if not required_columns.issubset(set(df.columns)):
                st.error("File harus memiliki kolom: 'bulan', 'tipe_motor', dan 'jumlah'")
                return

            if not df['bulan'].astype(str).str.match(r'^[A-Za-z]{3}-\d{2}$').all():
                st.error("Format kolom 'bulan' harus seperti 'Jan-19', 'Des-23', dll.")
                return

            df['jumlah'] = pd.to_numeric(df['jumlah'], errors='coerce')
            if df['jumlah'].isnull().any():
                st.error("Kolom 'jumlah' harus berisi angka.")
                return

            st.sidebar.success("✅ Data berhasil diunggah dan tervalidasi!")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
            return

        df['bulan'] = df['bulan'].apply(ubah_bulan)
        df['bulan'] = pd.to_datetime(df['bulan'], format='%b-%y')

        tipe_motor_options = sorted(list(df['tipe_motor'].unique()))
        tipe_motor_options.insert(0, 'All')

        st.markdown("### ⚙️ Konfigurasi Prediksi")
        tipe_motor_selected = st.multiselect("🔧 Pilih Tipe Motor:", tipe_motor_options, default=['All'])
        periode = st.slider("📆 Prediksi berapa bulan ke depan?", 1, 12, 6)
        st.markdown("---")

        if st.button("📊 Lihat Prediksi"):
            if 'All' in tipe_motor_selected:
                df_filtered = df.copy()
                selected_motors = tipe_motor_options[1:]
            else:
                df_filtered = df[df['tipe_motor'].isin(tipe_motor_selected)]
                selected_motors = tipe_motor_selected

            if df_filtered.empty:
                st.warning("Data kosong untuk tipe motor yang dipilih.")
                return

            all_predictions = pd.DataFrame()
            evaluation_results = []
            st.subheader("📉 Grafik Prediksi")

            for tipe_motor in selected_motors:
                df_motor = df_filtered[df_filtered['tipe_motor'] == tipe_motor]
                df_bulanan = df_motor.groupby('bulan')['jumlah'].sum().reset_index()
                df_bulanan.set_index('bulan', inplace=True)

                if len(df_bulanan) < 3:
                    st.info(f"Data untuk tipe motor '{tipe_motor}' terlalu sedikit untuk prediksi.")
                    continue

                df_bulanan['jumlah'] = df_bulanan['jumlah'].rolling(window=3, center=True).mean()
                df_bulanan['jumlah'] = df_bulanan['jumlah'].fillna(method='bfill').fillna(method='ffill')

                scaler = MinMaxScaler()
                jumlah_scaled = scaler.fit_transform(df_bulanan[['jumlah']])
                df_scaled = pd.DataFrame(jumlah_scaled, index=df_bulanan.index, columns=['jumlah'])
                df_log = np.log1p(df_scaled)

                model, order = find_best_arima_model(df_log)
                forecast_log = model.forecast(steps=periode)
                forecast_scaled = np.expm1(forecast_log.to_numpy().reshape(-1, 1))
                forecast = scaler.inverse_transform(forecast_scaled).flatten()

                future_dates = pd.date_range(df_bulanan.index[-1] + pd.offsets.MonthBegin(1), periods=periode, freq='MS')
                df_prediksi = pd.DataFrame({
                    'Tipe Motor': tipe_motor,
                    'Bulan': future_dates,
                    'Prediksi Penjualan': np.round(forecast).astype(int)
                })

                all_predictions = pd.concat([all_predictions, df_prediksi], ignore_index=True)

                fig, ax = plt.subplots()
                ax.plot(df_bulanan.index, df_bulanan['jumlah'], label=f'Data Historis {tipe_motor}', marker='o')
                ax.plot(df_prediksi['Bulan'], df_prediksi['Prediksi Penjualan'], label=f'Prediksi {tipe_motor}', marker='o')
                ax.set_xlabel("Bulan")
                ax.set_ylabel("Jumlah Terjual")
                ax.set_title(f"Prediksi Penjualan - {tipe_motor}")
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                save_predictions(df_prediksi)

                n_test = min(12, len(df_bulanan) // 3)
                if n_test >= 3:
                    train_log = df_log.iloc[:-n_test]
                    test_asli = df_bulanan.iloc[-n_test:]
                    model_eval, _ = find_best_arima_model(train_log)
                    pred_log_eval = model_eval.forecast(steps=n_test)
                    pred_scaled_eval = np.expm1(pred_log_eval.to_numpy().reshape(-1, 1))
                    pred_eval = scaler.inverse_transform(pred_scaled_eval).flatten()
                    mae = mean_absolute_error(test_asli['jumlah'], pred_eval)
                    mape = mean_absolute_percentage_error(test_asli['jumlah'], pred_eval) * 100
                    evaluation_results.append((tipe_motor, mae, mape))
                else:
                    evaluation_results.append((tipe_motor, None, None))
                    st.info(f"Data historis '{tipe_motor}' terlalu sedikit untuk evaluasi.")

            if all_predictions.empty:
                st.warning("Tidak ada prediksi yang bisa ditampilkan.")
                return

            st.subheader("📋 Tabel Hasil Prediksi")
            predictions_df = all_predictions.sort_values(['Tipe Motor', 'Bulan']).reset_index(drop=True)
            st.dataframe(predictions_df[['Tipe Motor', 'Bulan', 'Prediksi Penjualan']])

            st.subheader("🧪 Evaluasi Akurasi Model")
            for tipe_motor, mae, mape in evaluation_results:
                if mae is not None and mape is not None:
                    st.markdown(f"**🛵 {tipe_motor}**")
                    st.markdown(f"- MAE: {mae:.2f}")
                    st.markdown(f"- MAPE: {mape:.2f}%")
                else:
                    st.markdown(f"**🛵 {tipe_motor}** - Data terlalu sedikit untuk evaluasi.")

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                predictions_df.to_excel(writer, index=False, sheet_name='Prediksi')
            st.download_button(
                label="⬇️ Unduh Hasil Prediksi (Excel)",
                data=output.getvalue(),
                file_name="prediksi_gabungan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    elif st.session_state.role == 'viewer':
        st.subheader("📋 Hasil Prediksi Sebelumnya")
        past_predictions = display_predictions()
        past_predictions['tahun'] = pd.to_datetime(past_predictions['bulan']).dt.year
        grouped_predictions = past_predictions.groupby('tahun')

        for tahun, group in grouped_predictions:
            st.subheader(f"Hasil Prediksi untuk Tahun: {tahun}")
            st.dataframe(group[['tipe_motor', 'bulan', 'prediksi_penjualan']])

            if st.button(f"Hapus Semua untuk {tahun}"):
                for index, row in group.iterrows():
                    delete_prediction(row['id'])
                st.success(f"Semua prediksi untuk tahun {tahun} berhasil dihapus.")

# ===== START APP =====
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None

init_db()

if not st.session_state.logged_in:
    login()
else:
    if st.sidebar.button("Logout"):
        logout()
    main()

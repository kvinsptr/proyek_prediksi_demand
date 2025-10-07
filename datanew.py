import pandas as pd

# Baca data CSV
data = pd.read_csv("handphone_clean_fixed.csv")

# Pastikan kolom tanggal dikenali sebagai datetime
data['Tanggal'] = pd.to_datetime(data['Tanggal'])

import streamlit as st
import pandas as pd
import os
from model.ai_model import predict

# Veri dosyası yolu
DATA_PATH = "data/kullanicilar.csv"

# CSV dosyası yoksa oluştur
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists(DATA_PATH):
    df_init = pd.DataFrame(columns=["yemek", "etli", "aci", "sebzeli", "begeni"])
    df_init.to_csv(DATA_PATH, index=False)

# Başlık
st.title("🍽️ Lezzet Asistanı: Yapay Zeka ile Tat Profili")
st.write("Her gün yediğin yemekleri gir, sistem zevklerini öğrensin ve önerilerde bulunsun!")

# --- Günlük Yemek Kaydı ---
st.header("📋 Bugün Ne Yedin?")
yemek_adi = st.text_input("Yemek Adı")
col1, col2, col3 = st.columns(3)
etli = col1.checkbox("Etli")
aci = col2.checkbox("Acı")
sebzeli = col3.checkbox("Sebzeli")
begeni = st.radio("Bu yemeği beğendin mi?", ["Evet", "Hayır"])

if st.button("Kaydet"):
    yeni_kayit = pd.DataFrame.from_dict({
        "yemek": [yemek_adi],
        "etli": [int(etli)],
        "aci": [int(aci)],
        "sebzeli": [int(sebzeli)],
        "begeni": [1 if begeni == "Evet" else 0]
    })
    yeni_kayit.to_csv(DATA_PATH, mode='a', header=False, index=False)
    st.success("Kayıt başarıyla eklendi! Öğrenme verisi güncellendi.")

# --- Tat Tahmini ---
st.header("🔮 Tat Tahmini")
st.write("Yeni bir yemek hakkında tahmin al")

test_etli = st.checkbox("Etli mi?", key="test_etli")
test_aci = st.checkbox("Acı mı?", key="test_aci")
test_sebzeli = st.checkbox("Sebzeli mi?", key="test_sebzeli")

tahmin_btn = st.button("Beğenir miyim?")

if tahmin_btn:
    test_input = [[int(test_etli), int(test_aci), int(test_sebzeli)]]
    # Eğitim verisini oku
    data = pd.read_csv(DATA_PATH)
    X = data[["etli", "aci", "sebzeli"]].values
    y = data["begeni"].values.reshape(-1, 1)

    # Modeli eğit ve tahmin yap
    prob = predict(test_input, X, y)
    st.success(f"Bu yemeği beğenme ihtimalin: %{prob * 100:.2f}")

# --- Profil Özeti ---
st.header("📊 Zevk Profilin")
veri = pd.read_csv(DATA_PATH)
if len(veri) >= 5:
    ort = veri.groupby("begeni")[["etli", "aci", "sebzeli"]].mean()
    st.write("Beğendiğin yemeklerin ortalama özellikleri:")
    st.dataframe(ort.loc[1].T.rename(columns={1: "Ortalama"}))
else:
    st.info("Profilini görmek için en az 5 yemek girişi yapmalısın.")

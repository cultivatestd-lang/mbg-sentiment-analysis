"""Streamlit app: Analisis Sentimen Program Makan Bergizi Gratis (MBG)."""
import re
import pickle
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Analisis Sentimen MBG",
    page_icon="🍚",
    layout="centered",
)


@st.cache_resource
def load_model():
    with open("svm_tuned_mbg.pkl", "rb") as f:
        model = pickle.load(f)
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    return model


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


LABEL_COLOR = {
    "Positif": "#1B998B",
    "Negatif": "#E63946",
    "Netral":  "#6C757D",
}
LABEL_EMOJI = {"Positif": "😊", "Negatif": "😠", "Netral": "😐"}


st.markdown(
    "<h1 style='text-align:center;margin-bottom:0;'>🍚 Analisis Sentimen MBG</h1>"
    "<p style='text-align:center;color:#666;margin-top:4px;'>"
    "Program Makan Bergizi Gratis — SVM + TF-IDF"
    "</p><hr/>",
    unsafe_allow_html=True,
)

st.markdown(
    "Aplikasi ini mengklasifikasikan opini publik mengenai **Program Makan "
    "Bergizi Gratis (MBG)** ke dalam tiga kategori sentimen: **Positif**, "
    "**Negatif**, atau **Netral**. Model dilatih menggunakan algoritma "
    "*Support Vector Machine* dengan kernel linear pada representasi TF-IDF "
    "dari 5.698 tweet berbahasa Indonesia."
)

user_tweet = st.text_area(
    "Masukkan teks tweet / opini publik:",
    placeholder="Contoh: program makan bergizi gratis sangat membantu anak sekolah...",
    height=140,
)

col1, col2 = st.columns([1, 1])
predict_btn = col1.button("🔍 Analisis Sentimen", use_container_width=True, type="primary")
clear_btn = col2.button("🧹 Bersihkan", use_container_width=True)

if clear_btn:
    st.rerun()

if predict_btn:
    if not user_tweet.strip():
        st.warning("Mohon masukkan teks terlebih dahulu.")
    else:
        model = load_model()
        cleaned = clean_text(user_tweet)
        probs = model.predict_proba([cleaned])[0]
        idx = int(np.argmax(probs))
        label = model.classes_[idx]
        confidence = float(probs[idx])

        st.markdown("### Hasil Prediksi")
        st.markdown(
            f"<div style='padding:18px;border-radius:10px;"
            f"background:{LABEL_COLOR[label]}15;"
            f"border-left:6px solid {LABEL_COLOR[label]};'>"
            f"<h2 style='margin:0;color:{LABEL_COLOR[label]};'>"
            f"{LABEL_EMOJI[label]} {label}</h2>"
            f"<p style='margin:6px 0 0;color:#333;'>"
            f"Tingkat keyakinan model: <b>{confidence*100:.2f}%</b></p></div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Distribusi Probabilitas Kelas")
        for cls, p in zip(model.classes_, probs):
            st.write(f"**{cls}**")
            st.progress(float(p), text=f"{p*100:.2f}%")

        with st.expander("🔧 Detail Preprocessing"):
            st.code(f"Input asli   : {user_tweet}\nSetelah bersih: {cleaned}", language="text")

st.markdown(
    "<hr/><p style='text-align:center;color:#888;font-size:12px;'>"
    "Final Project Machine Learning · Universitas AMIKOM Yogyakarta · 2026"
    "</p>",
    unsafe_allow_html=True,
)

"""Quick training of TF-IDF + SVM on small labeled MBG sentiment data to produce svm_tuned_mbg.pkl."""
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

data = [
    ("program makan bergizi gratis sangat bagus membantu anak sekolah", "Positif"),
    ("mbg ini bagus sekali untuk gizi anak indonesia", "Positif"),
    ("saya mendukung penuh program mbg presiden prabowo", "Positif"),
    ("makanan bergizi ini bermanfaat untuk masa depan anak", "Positif"),
    ("program ini luar biasa membantu keluarga kurang mampu", "Positif"),
    ("mbg sukses menurunkan angka stunting di sekolah", "Positif"),
    ("anak saya senang mendapat makanan bergizi gratis", "Positif"),
    ("kualitas makanan mbg enak dan sehat", "Positif"),
    ("salut dengan program makan bergizi ini sangat berguna", "Positif"),
    ("program mbg memberi dampak positif untuk pendidikan", "Positif"),
    ("mbg adalah hadiah terbaik untuk anak indonesia", "Positif"),
    ("terima kasih pemerintah atas program mbg yang luar biasa", "Positif"),

    ("program mbg ini gagal anggaran membengkak tidak transparan", "Negatif"),
    ("makanan basi bikin keracunan anak sekolah", "Negatif"),
    ("mbg buang anggaran negara tidak tepat sasaran", "Negatif"),
    ("kasian anak keracunan gara gara makan bergizi gratis", "Negatif"),
    ("program ini cuma pencitraan saja tidak efektif", "Negatif"),
    ("makanan mbg tidak layak konsumsi banyak basi", "Negatif"),
    ("anggaran mbg dikorupsi oleh oknum tidak bertanggung jawab", "Negatif"),
    ("mbg gagal total banyak anak yang sakit", "Negatif"),
    ("program ini hanya menambah hutang negara percuma", "Negatif"),
    ("buruk sekali pelaksanaan mbg di daerah saya", "Negatif"),
    ("mbg menyebabkan banyak masalah kesehatan", "Negatif"),
    ("kacau sekali distribusi makanan bergizi gratis ini", "Negatif"),

    ("program mbg dimulai hari ini di sekolah dasar", "Netral"),
    ("pemerintah mengumumkan jadwal pelaksanaan mbg", "Netral"),
    ("berita mbg hari ini di televisi", "Netral"),
    ("rapat koordinasi mbg digelar di kantor walikota", "Netral"),
    ("jumlah penerima manfaat mbg mencapai ribuan siswa", "Netral"),
    ("mbg masuk dalam program kerja kementerian", "Netral"),
    ("evaluasi program mbg dilakukan setiap bulan", "Netral"),
    ("rincian anggaran mbg telah dipublikasikan", "Netral"),
    ("petugas distribusi mbg datang tepat waktu hari ini", "Netral"),
    ("data penerima mbg dapat diakses publik", "Netral"),
    ("mbg melibatkan banyak vendor catering lokal", "Netral"),
    ("informasi mbg tersedia di situs resmi", "Netral"),
]

X = [t for t, _ in data]
y = [l for _, l in data]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
    ("svm", SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced")),
])
pipe.fit(X, y)

with open("svm_tuned_mbg.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("Saved svm_tuned_mbg.pkl")
print("Train accuracy:", pipe.score(X, y))

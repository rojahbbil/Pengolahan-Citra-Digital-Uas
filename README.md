# Klasifikasi Citra Bunga Menggunakan Multi-Layer Perceptron (MLP)

**Mata Kuliah:** Pengolahan Citra Digital (SIF202) — Genap 2025/2026  
**Dosen:** Teuku Rizky Noviandy, S.Kom., M.Kom.

| | |
|---|---|
| **Nama** | Roja Hubbil Khairi |
| **NIM** | 24146036 |

## Deskripsi

Proyek ini membangun sistem klasifikasi 5 jenis bunga (Daisy, Dandelion, Rose, Sunflower, Tulip) menggunakan algoritma **Multi-Layer Perceptron (MLP)** dari scikit-learn. Dataset berasal dari [Kaggle Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).

## Struktur Proyek

```
├── dataset/
│   └── flowers/
│       ├── daisy/          (764 citra)
│       ├── dandelion/      (1052 citra)
│       ├── rose/           (784 citra)
│       ├── sunflower/      (733 citra)
│       └── tulip/          (984 citra)
├── notebook/
│   └── klasifikasi_bunga.ipynb
├── .gitignore
└── README.md
```

## Pipeline

1. **Preprocessing** — Resize 64×64, normalisasi [0,1], flatten → 12288 fitur
2. **Data Splitting** — 80% training, 20% testing (`random_state` = NIM)
3. **MLP Training** — 2 hidden layers (128, 64), ReLU, Adam, max_iter=500
4. **Evaluasi** — Classification Report (`digits=4`), Confusion Matrix, visualisasi 25 sampel

## Cara Menjalankan

```bash
pip install numpy matplotlib pillow scikit-learn notebook
jupyter notebook notebook/klasifikasi_bunga.ipynb
```

Lalu klik **Run All** dari menu Cell.

## Hasil

| Metrik | Nilai |
|---|---|
| Akurasi | 44.10% |
| Iterasi | 125 |
| Loss | 0.1139 |

## Referensi

- [Kaggle Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- Scikit-learn: `MLPClassifier`, `train_test_split`, `classification_report`

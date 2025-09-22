# llama3-rag
Proyek sederhana untuk mencari judul skripsi yang cocok menggunakan Llama 3 dan RAG (Retrieval Augmented Generation).

## 📋 Deskripsi
Project ini menggunakan model Llama 3 untuk membantu mahasiswa mencari dan merekomendasikan judul skripsi yang sesuai dengan minat dan bidang studi mereka. Dengan menggunakan teknik RAG, sistem dapat memberikan rekomendasi yang lebih akurat berdasarkan database topik yang tersedia.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Model Llama 3
Buka dan jalankan notebook `download_llama3_model.ipynb` untuk mendownload model Llama 3:

```bash
jupyter notebook download_llama3_model.ipynb
```

Notebook ini menyediakan beberapa metode download:
- **Hugging Face Transformers** (Recommended)
- **Hugging Face Hub**
- **Ollama** (Alternative)

### 3. Mulai Menggunakan
Setelah model berhasil didownload, Anda dapat mulai menggunakan sistem RAG untuk mencari rekomendasi judul skripsi.

## 📁 Struktur Project
```
llama3-rag/
├── download_llama3_model.ipynb    # Notebook untuk download model
├── requirements.txt               # Dependencies
├── models/                       # Directory untuk menyimpan model
└── README.md                     # Documentation
```

## 💡 Features
- Multiple metode download model Llama 3
- Automatic error handling dan progress tracking
- Optimized untuk berbagai hardware (CPU/GPU)
- Dokumentasi lengkap dalam Bahasa Indonesia
- Ready-to-use example untuk RAG implementation

## 🔧 System Requirements
- Python 3.8+
- Minimal 15GB free disk space
- Koneksi internet yang stabil
- (Optional) CUDA-compatible GPU untuk performa optimal

## 📖 Cara Penggunaan
1. Clone repository ini
2. Install dependencies dengan `pip install -r requirements.txt`
3. Buka dan jalankan `download_llama3_model.ipynb`
4. Pilih metode download yang sesuai
5. Tunggu proses download selesai
6. Model siap digunakan untuk RAG tasks

## ⚡ Tips
- Untuk penggunaan pertama, disarankan menggunakan model 8B-Instruct
- Pastikan memiliki space yang cukup sebelum download
- Gunakan Ollama jika ingin setup yang lebih mudah
- Request access ke Meta Llama models di Hugging Face jika diperlukan

## 🤝 Contributing
Kontribusi selalu welcome! Silakan buat issue atau pull request.

## 📄 License
MIT License

# 📡 Peramalan WiFi IndiBiz — Holt's DES (Gradio App)

Aplikasi web interaktif untuk peramalan jumlah pemasangan WiFi IndiBiz memakai
Holt's Double Exponential Smoothing, lengkap dengan data cleaning otomatis,
train-test split, optimasi parameter (α, γ), dan evaluasi model (MAE, RMSE, MAPE).

## Menjalankan di lokal

```bash
pip install -r requirements.txt
python app.py
```

Lalu buka `http://127.0.0.1:7860` di browser. Upload file `Data_Mentah.xlsx`
(harus punya kolom `TGL_PS` berformat `YYYYMMDD`), atur parameter di sidebar,
klik **Jalankan Analisis**.

## Deploy ke Hugging Face Spaces (gratis, paling gampang)

1. Buat akun di https://huggingface.co (kalau belum punya).
2. Buka https://huggingface.co/new-space
   - Pilih **SDK: Gradio**
   - Pilih nama Space, visibilitas public/private bebas
3. Setelah Space dibuat, upload 2 file ini ke Space tersebut (lewat tab "Files" → "Add file"):
   - `app.py`
   - `requirements.txt`
4. Space otomatis build dan jalan dalam 1-2 menit. Selesai — aplikasi bisa diakses
   lewat URL Space kamu (mis. `https://huggingface.co/spaces/username/nama-space`).

Alternatif: pakai Git —
```bash
git clone https://huggingface.co/spaces/<username>/<nama-space>
cp app.py requirements.txt <nama-space>/
cd <nama-space>
git add . && git commit -m "deploy" && git push
```

## Deploy dengan link publik sementara (tanpa hosting)

Untuk demo cepat tanpa deploy permanen, tambahkan `share=True`:

```python
demo.launch(share=True)
```

Ini menghasilkan link publik `*.gradio.live` yang aktif selama proses lokal
berjalan (biasanya 72 jam), cocok untuk demo ke dosen/atasan tanpa setup server.

## Struktur file

- `app.py` — seluruh logika (cleaning, split, optimasi, evaluasi, forecast) + UI Gradio
- `requirements.txt` — daftar dependency Python

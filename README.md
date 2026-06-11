# MAMG

Multi-Agent AI Microstock Generator for Google Colab.

## File Utama
- `mamg_colab.ipynb` untuk Colab
- `main.py` untuk run script lokal
- `requirements_colab.txt` untuk install di Colab

## Cara Pakai di Colab
1. Buka `mamg_colab.ipynb`
2. Run cell mount Google Drive
3. Run cell clone repo
4. Run cell install library
5. Edit cell `Setting Generate`
6. Run cell pipeline

## Setting Generate
Isi variabel ini di cell setting:
- `CATEGORY`
- `GEO`
- `MAX_KEYWORDS`
- `NUM_IMAGES`
- `MODEL_NAME`
- `HEIGHT`
- `WIDTH`
- `SEED`
- `USE_CPU_OFFLOAD`
- `ENABLE_LORA`
- `LORA_PATHS`
- `VISUAL_TYPE_LORA_MAP`
- `UPSCALE_TARGET`
- `NUM_INFERENCE_STEPS`
- `GUIDANCE_SCALE`
- `PROMPT_MIN_KEYWORDS`
- `PROMPT_MAX_KEYWORDS`

## Output
Hasil tersimpan di Google Drive:
- `MyDrive/AI_Microstock_Agent/outputs/<tanggal>_<trend>/images`
- `MyDrive/AI_Microstock_Agent/outputs/<tanggal>_<trend>/metadata.csv`

## Catatan
- Model default yang paling aman untuk Colab gratis adalah `stabilityai/sdxl-turbo`.
- `NUM_IMAGES` adalah jumlah gambar yang ingin di-generate.
- `SEED = None` berarti hasilnya bisa bervariasi.

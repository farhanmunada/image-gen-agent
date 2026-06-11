# Agent

Colab-focused image generation pipeline untuk aset microstock.

## File Utama

- `agent_colab.ipynb` untuk Colab
- `requirements_colab.txt` untuk install di Colab

## Cara Pakai di Colab

1. Buka `agent_colab.ipynb`
2. Jalankan cell mount Google Drive
3. Jalankan cell clone repo
4. Jalankan cell install library
5. Isi form `Setting Generate`
6. Jalankan pipeline

## Setting Generate

Isi form ini di cell setting:

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
- Ukuran gambar minimal adalah Full HD `1920x1080`.
- `NUM_IMAGES` adalah jumlah gambar yang ingin di-generate.
- `SEED = None` berarti hasilnya bisa bervariasi.

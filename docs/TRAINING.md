# Eğitim Kılavuzu - KuroNeko TR

## Genel Bakış

KuroNeko TR, Microsoft'un Phi-4-mini-instruct (3.8B) modelini Türkçe fine-tune eden bir projedir. Lightning AI LitGPT kütüphanesini kullanır.

## Donanım Gereksinimleri

| Yöntem | RAM | GPU VRAM | Süre (1 epoch) | Maliyet |
|--------|-----|----------|----------------|---------|
| QLoRA 4-bit (GPU) | ~16GB | ~8GB (T4) | 2-3 saat | ~$0.40/saat |
| LoRA 16-bit (GPU) | ~32GB | ~16GB (A10) | 1-2 saat | ~$1.00/saat |
| QLoRA (CPU) | ~16GB | Yok | 12-24 saat | Ücretsiz |
| Full fine-tune | ~64GB | ~80GB (A100) | 4-8 saat | ~$2.50/saat |

## Önerilen Yol

1. **Colab T4 GPU** (Ücretsiz): QLoRA 4-bit, 1-2 epoch
2. **Kaggle T4 GPU** (Ücretsiz): QLoRA 4-bit, internet gerekli
3. **CPU** (Yavaş): QLoRA 4-bit, çok uzun sürer

## Adım Adım Eğitim

### 1. Veri Hazırlığı

```bash
python data/prepare_dataset.py
```

Bu komut:
- HuggingFace'den Türkçe veri setlerini indirir
- Sentetik Türkçe veri oluşturur
- Akıl yürütme verileri ekler
- `data/processed/train.jsonl` ve `data/processed/val.jsonl` oluşturur

### 2. Eğitim (GPU - Önerilen)

```bash
# Colab veya GPU sunucuda
python scripts/train_lora.py \
  --config config/phi4_mini_lora.yaml \
  --hf_token hf_xxxxx \
  --push_to_hub
```

### 3. Eğitim (CPU - Yavaş)

```bash
python scripts/train_lora.py \
  --config config/phi4_mini_qlora.yaml \
  --precision 32-true \
  --devices 1
```

### 4. Eğitimi Devam Et (Resume)

```bash
python scripts/resume.py \
  --config config/phi4_mini_lora.yaml \
  --checkpoint output/last.ckpt \
  --additional_epochs 1
```

### 5. Değerlendirme

```bash
python scripts/evaluate.py \
  --model_path output/merged \
  --device auto
```

### 6. Export & Push

```bash
python scripts/export.py \
  --model_path output/merged \
  --format all \
  --push_to_hub \
  --repo_id KuroNeko1234t/phi4-mini-tr \
  --hf_token hf_xxxxx
```

## Konfigürasyon

### LoRA (config/phi4_mini_lora.yaml)
- `lora_r: 16` - LoRA rank
- `lora_alpha: 32` - LoRA alpha
- `learning_rate: 2e-4` - Öğrenme oranı
- `max_epochs: 3` - Epoch sayısı
- `batch_size: 2` - Batch size
- `precision: bf16-mixed` - Precision

### QLoRA (config/phi4_mini_qlora.yaml)
- LoRA + 4-bit quantization
- Daha az bellek kullanır
- CPU için uygun

## Checkpoint Yönetimi

Eğitim sırasında checkpoint'ler `output/` dizinine kaydedilir:
- `output/step-00100.ckpt` - Her 100 adımda
- `output/last.ckpt` - En son checkpoint

Yarım kalan eğitimi devam ettirmek için:
```bash
python scripts/resume.py --config config/phi4_mini_lora.yaml
```

## Sorun Giderme

### "CUDA out of memory"
- `micro_batch_size` değerini 1'e düşür
- `max_seq_length` değerini 1024'e düşür
- QLoRA kullan

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Eğitim çok yavaş
- GPU kullanıldığından emin ol
- `precision: bf16-mixed` kullan
- `gradient_accumulation_steps` artır

## Sonraki Adımlar

1. Eğitimi tamamla
2. Modeli değerlendir
3. HF Hub'a push et
4. HF Space oluştur
5. Gradio arayüzünü deploy et

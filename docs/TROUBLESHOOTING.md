# Sorun Giderme - KuroNeko TR

## Eğitim Sorunları

### "CUDA out of memory"
**Çözüm:**
```yaml
# config/phi4_mini_qlora.yaml
micro_batch_size: 1
max_seq_length: 1024
gradient_accumulation_steps: 32
quantize: bitsandbytes.nf4
```

### "ModuleNotFoundError: No module named 'litgpt'"
**Çözüm:**
```bash
pip install "litgpt[all]"
```

### Eğitim çok yavaş (CPU)
**Beklenen:** CPU'da 3.8B model için 1 epoch ~12-24 saat
**Çözüm:**
- QLoRA 4-bit kullan
- `max_seq_length` 1024'e düşür
- `max_epochs` 1 yap
- GPU bul (Colab/Kaggle)

### Loss düşmüyor
**Çözüm:**
- Learning rate artır: `3e-4` veya `5e-4`
- Daha fazla veri ekle
- `max_epochs` artır
- LoRA rank artır: `lora_r: 32`

### "Connection error" (HF Hub)
**Çözüm:**
```bash
export HF_TOKEN=hf_xxxxx
huggingface-cli login
```

## Inference Sorunları

### Model yüklenmiyor (Space)
**Çözüm:**
1. Model repo adını kontrol et
2. HF Token ekle (private model için)
3. Quantized model kullan (Q4_K_M)
4. Donanım yükselt (T4)

### Yanıt kalitesi düşük
**Çözüm:**
- Temperature ayarla: 0.7
- System prompt ekle
- Daha fazla epoch eğit
- Daha kaliteli veri kullan

### "Token indices sequence length is longer than..."
**Çözüm:**
```python
max_new_tokens = 256  # Daha kısa yanıt
max_seq_length = 1024  # Daha kısa prompt
```

## Kaggle Sorunları

### Internet yok
**Çözüm:**
- Settings → Internet → ON
- Unsloth wheel dataset ekle: `barnobarno/unsloth-wheels`
- Veya Colab kullan

### "Notebook not found" (push)
**Çözüm:**
```python
# SDK kullan
from kaggle_client import ApiSaveKernelRequest
request = ApiSaveKernelRequest.from_dict({
    "newTitle": "phi4-tr-train",
    "language": "python",
    "kernelType": "notebook",
    "text": notebook_json,
})
```

## Colab Sorunları

### GPU limiti dolmuş
**Çözüm:**
- Farklı bir Google hesap dene
- Colab Pro al
- Kaggle dene
- CPU ile çalıştır (yavaş)

### "Runtime disconnected"
**Çözüm:**
- Checkpoint kaydetmeyi unutma
- `resume.py` ile devam et
- Daha küçük batch size kullan

## HF Space Sorunları

### Space başlamıyor
**Çözüm:**
1. `requirements.txt` kontrol et
2. Import hatalarını log'dan kontrol et
3. Donanım yetersiz olabilir → CPU Basic'e düşür
4. Model boyutu çok büyük → Q4 quantized kullan

### "Error loading model"
**Çözüm:**
```python
# app.py'de model ID kontrol et
MODEL_ID = "KuroNeko1234t/phi4-mini-tr"  # Doğru repo adı olduğundan emin ol
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Token ekle
```

## Performans Optimizasyonu

### Daha hızlı eğim
- `precision: bf16-mixed` kullan
- `gradient_accumulation_steps` artır
- `micro_batch_size` artır (VRAM yeterliyse)
- Flash Attention ekle

### Daha az bellek
- QLoRA 4-bit kullan
- `lora_r` düşür (8 veya 16)
- `max_seq_length` düşür
- CPU offloading kullan

### Daha iyi kalite
- Daha fazla veri ekle
- Daha fazla epoch çalıştır
- `lora_r` artır (32 veya 64)
- Daha yüksek `learning_rate` dene

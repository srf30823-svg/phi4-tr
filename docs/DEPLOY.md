# Deployment Kılavuzu - KuroNeko TR

## HuggingFace Space Oluşturma

### 1. Space Oluştur
1. https://huggingface.co/new-space adresine git
2. **Space adı:** `phi4-mini-tr`
3. **Space SDK:** Gradio
4. **Hardware:** CPU Basic (ücretsiz) veya T4 Small (hızlı)
5. **Visibility:** Public

### 2. Dosyaları Yükle
```bash
git clone https://huggingface.co/spaces/KuroNeko1234t/phi4-mini-tr
cd phi4-mini-tr
cp space/app.py .
cp space/requirements.txt .
git add .
git commit -m "Initial Space setup"
git push
```

### 3. Environment Variables
Space Settings → Secrets:
- `HF_TOKEN`: HuggingFace token (private model için)
- `MODEL_ID`: Model repo adı (varsayılan: KuroNeko1234t/phi4-mini-tr)

## Donanım Seçenekleri

| Hardware | CPU | RAM | GPU | Fiyat | Uygunluğu |
|----------|-----|-----|-----|-------|-----------|
| CPU Basic | 2 vCPU | 16GB | - | Ücretsiz | Küçük model (Q4) |
| CPU Upgrade | 8 vCPU | 32GB | - | $0.03/saat | Orta model |
| T4 Small | 4 vCPU | 15GB | 16GB | $0.40/saat | Phi-4 Q4 |
| T4 Medium | 8 vCPU | 30GB | 16GB | $0.60/saat | Phi-4 Q4 |
| L4 | 8 vCPU | 30GB | 24GB | $0.80/saat | Phi-4 Q8 |

## Model Boyutları

| Format | Boyut | RAM Gereksinimi |
|--------|-------|-----------------|
| FP16 | ~8GB | ~16GB |
| Q8 | ~4.5GB | ~10GB |
| Q4_K_M | ~2.5GB | ~6GB |
| Q2_K | ~1.5GB | ~4GB |

## API Kullanımı

### Gradio Client
```python
from gradio_client import Client

client = Client("KuroNeko1234t/phi4-mini-tr")
result = client.predict(
    message="Merhaba!",
    api_name="/predict"
)
print(result)
```

### REST API
```python
import requests

API_URL = "https://kuroneko1234t-phi4-mini-tr.hf.space/api/predict"
response = requests.post(API_URL, json={
    "data": ["Merhaba!"]
})
print(response.json())
```

## ZeroGPU (Beta)

Pro hesabı için ZeroGPU kullanılabilir:
1. Space Settings → Hardware → ZeroGPU
2. `spaces` kütüphanesini ekle:
```python
import spaces

@spaces.GPU
def generate(message):
    return model.generate(message)
```

## Fallback Stratejisi

Space'de model yüklenemezse:
1. Q4_K_M quantized model dene
2. Q2_K dene
3. CPU inference dene
4. Hata mesajı göster

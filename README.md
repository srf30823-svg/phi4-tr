# KuroNeko Phi-4 Mini TR - Lightning AI LitGPT Eğitim Projesi

## 🎯 Hedef
Phi-4-mini-instruct (3.8B) modelini Türkçe fine-tune ederek:
- LLM Arena'da sohbette ilk 100, code'da 50'ye giren model
- 16RAM/2CPU HF Space'de çalışabilir model
- CPU ile eğitilebilir (yavaş ama mümkün), GPU ile hızlı eğitim

## 📁 Dizin Yapısı

```
kuroneko-tr/
├── README.md                    # Bu dosya
├── .gitignore                   # Git ignore
├── requirements.txt             # Python bağımlılıkları
│
├── config/                      # Konfigürasyon dosyariablesı
│   ├── phi4_mini_lora.yaml      # LoRA eğitim config
│   ├── phi4_mini_qlora.yaml     # QLoRA eğitim config
│   └── phi4_mini_full.yaml      # Full fine-tune config
│
├── data/                        # Veri setleri
│   ├── prepare_dataset.py       # Veri hazırlama scripti
│   ├── turkish_instructions.py  # Türkçe talimat verileri
│   ├── code_generation.py       # Kod üretim verileri
│   ├── reasoning.py             # Akıl yürütme verileri
│   └── synthetic.py             # Sentetik veri üretimi
│
├── scripts/                     # Eğitim scriptleri
│   ├── train_lora.py            # LoRA eğitim
│   ├── train_qlora.py           # QLoRA eğitim
│   ├── train_full.py            # Full fine-tune
│   ├── evaluate.py              # Değerlendirme
│   ├── export.py                # Model export (GGUF, etc)
│   └── resume.py                # Eğitimi devam ettir
│
├── notebooks/                   # Colab/Kaggle notebookları
│   ├── train_colab.ipynb        # Colab eğitim notebook
│   ├── train_kaggle.ipynb       # Kaggle eğitim notebook
│   └── evaluate.ipynb            # Değerlendirme notebook
│
├── space/                       # HuggingFace Space
│   ├── app.py                   # Gradio arayüzü
│   ├── requirements.txt         # Space bağımlılıkları
│   └── Dockerfile               # Docker config (opsiyonel)
│
├── tests/                       # Test scriptleri
│   ├── test_model.py            # Model test
│   ├── test_data.py             # Veri test
│   └── test_inference.py        # Inference test
│
└── docs/                        # Dokümantasyon
    ├── TRAINING.md              # Eğitim kılavuzu
    ├── DATASET.md               # Veri seti kılavuzu
    ├── DEPLOY.md                # Deployment kılavuzu
    └── TROUBLESHOOTING.md       # Sorun giderme
```

## 🚀 Hızlı Başlangıç

### Colab ile Eğitim (Önerilen)
1. `notebooks/train_colab.ipynb` açın
2. Runtime → Change runtime type → T4 GPU
3. HF Token girin
4. Run All

### CPU ile Eğitim (Yavaş ama mümkün)
```bash
pip install -r requirements.txt
python scripts/train_qlora.py --config config/phi4_mini_qlora.yaml
```

### GPU ile Eğitim (Hızlı)
```bash
python scripts/train_lora.py --config config/phi4_mini_lora.yaml
```

## 📊 Donanım Gereksinimleri

| Yöntem | RAM | GPU VRAM | Süre (1 epoch) |
|--------|-----|----------|-----------------|
| QLoRA 4-bit | ~16GB | ~8GB (T4) | 2-3 saat |
| LoRA 16-bit | ~32GB | ~16GB (A10) | 1-2 saat |
| Full fine-tune | ~64GB | ~80GB (A100) | 4-8 saat |
| CPU QLoRA | ~16GB | Yok | 12-24 saat |

## 🔧 Fallback Modelleri
- Primary: OWL (OpenRouter)
- Fallback 1: GLM-4.7 Flash (z.ai)
- Fallback 2: GLM-4.5 Flash (z.ai)

## 📝 Adım Listesi (600 adım)
- [ ] Adım 1-50: Repo temizliği & Planlama
- [ ] Adım 51-100: Araştırma & Bilgi Toplama
- [ ] Adım 101-200: Veri Hazırlığı
- [ ] Adım 201-300: LitGPT Konfigürasyon
- [ ] Adım 301-400: Eğitim Scriptleri
- [ ] Adım 401-500: Test & Değerlendirme
- [ ] Adım 501-550: HF Space Deploy
- [ ] Adım 551-600: Dokümantasyon & Tamamlama

## 📄 Lisans
MIT License

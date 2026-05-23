# Proje Planı - KuroNeko TR (600 Adım)

## Genel Bakış
Phi-4-mini-instruct (3.8B) modelini Türkçe fine-tune ederek LLM Arena'da rekabetçi bir model oluşturmak.

## Hedefler
- **Sohbet:** LLM Arena ilk 100
- **Code:** LLM Arena ilk 50
- **Donanım:** HF Space 16RAM/2CPU'da çalışır
- **Dil:** Türkçe akıcılık, akıl yürütme, kod üretimi

## 600 Adım Planı

### Faz 1: Temel Altyapı (Adım 1-100) ✅
- [x] GitHub repo oluşturma ve temizlik
- [x] Dizin yapısı
- [x] README.md
- [x] .gitignore
- [x] Config dosyaları (LoRA, QLoRA)
- [x] Veri hazırlama scripti
- [x] Eğitim scriptleri
- [x] Dokümantasyon

### Faz 2: Veri & Konfigürasyon (Adım 101-200) ✅
- [x] Türkçe veri setleri entegrasyonu
- [x] Sentetik veri üretimi
- [x] Akıl yürütme verileri
- [x] Kod üretim verileri
- [x] Benchmark scripti
- [x] Checkpoint yönetimi

### Faz 3: Eğitim & Optimizasyon (Adım 201-300) ✅
- [x] LoRA eğitim scripti
- [x] QLoRA eğitim scripti
- [x] DPO eğitim scripti
- [x] Resume (devam ettirme) desteği
- [x] Quantization scripti
- [x] Hızlı başlangıç scripti

### Faz 4: Test & Değerlendirme (Adım 301-400) 🔄
- [x] Test scriptleri
- [x] Benchmark scripti
- [x] Değerlendirme scripti
- [ ] Otomatik test pipeline
- [ ] CI/CD entegrasyonu

### Faz 5: Deployment (Adım 401-500) 🔄
- [x] HF Space app.py
- [x] Gradio arayüzü
- [x] API entegrasyonu
- [ ] ZeroGPU desteği
- [ ] Multi-Space API birleştirme

### Faz 6: Dokümantasyon & Tamamlama (Adım 501-600) 🔄
- [x] Eğitim kılavuzu
- [x] Veri seti kılavuzu
- [x] Deployment kılavuzu
- [x] Sorun giderme
- [ ] Son kontroller
- [ ] Performans optimizasyonu

## Donanım Stratejisi

### Eğitim
1. **Colab T4 GPU** (Ücretsiz) - QLoRA 4-bit, 1-2 epoch
2. **Kaggle T4 GPU** (Ücretsiz) - Internet gerekli
3. **CPU** (Yavaş) - QLoRA 4-bit, çok uzun sürer

### Inference (HF Space)
1. **CPU Basic** (16GB RAM) - Q4_K_M quantized model
2. **T4 Small** (15GB RAM, 16GB VRAM) - Q4 veya Q8
3. **ZeroGPU** (Pro) - Dinamik GPU

### Fallback Modelleri
- **Primary:** OWL (OpenRouter)
- **Fallback 1:** GLM-4.7 Flash (z.ai)
- **Fallback 2:** GLM-4.5 Flash (z.ai)

## Önemli Kararlar

### CPU ile Eğitim
- **Mümkün mü?** Evet, ama yavaş (1 epoch ~12-24 saat)
- **Hedefi karşılar mı?** Evet, QLoRA 4-bit ile 3.8B model eğitilebilir
- **GPU gerekir mi?** Hayır, ama çok daha hızlı olur

### Phi-4 Mini Mimarisi
- **Parametre:** 3.8B
- **Context:** 128K token
- **Mimari:** Transformer (Phi-3 ile benzer)
- **Quantize:** 4-bit NF4 ile ~2.5GB

### HF Space Limitleri
- **CPU Basic:** 2 vCPU, 16GB RAM, ücretsiz
- **Model boyutu:** Q4_K_M ile ~2.5GB
- **RAM kullanımı:** ~6GB (model) + ~2GB (sistem) = ~8GB
- **Kalan:** 8GB (yeterli)

## Sonraki Adımlar
1. Eğitimi başlat (Colab T4 GPU)
2. Modeli değerlendir
3. HF Hub'a push et
4. HF Space oluştur
5. Gradio arayüzünü deploy et
6. Benchmark yap
7. İterasyon

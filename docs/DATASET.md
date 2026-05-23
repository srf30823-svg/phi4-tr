# Veri Seti Kılavuzu - KuroNeko TR

## Veri Formatı

LitGPT, Alpaca formatında JSONL veri bekler:

```json
{"instruction": "Soru veya talimat", "input": "Ek bağlam (opsiyonel)", "output": "Cevap"}
```

## Veri Kaynakları

### HuggingFace Veri Setleri
- `merve/turkish_instructions` - Türkçe talimat veri seti
- `ucekmez/OpenOrca-tr` - Türkçe OpenOrca
- `atasoglu/databricks-dolly-15k-tr` - Türkçe Dolly
- `ytu-ce-cosmos/gsm8k_tr` - Türkçe matematik
- `beratcmn/no_robots_turkish` - Türkçe No Robots
- `beratcmn/lima-tr` - Türkçe LIMA

### Sentetik Veri
- Matematik / Akıl yürütme (50+ örnek)
- Kod üretim (30+ örnek)
- Türkçe kültür (20+ örnek)
- Genel kültür (15+ örnek)

## Veri Dağılımı

| Kategori | Örnek | Yüzde |
|----------|-------|-------|
| Türkçe sohbet | ~5000 | %25 |
| Kod üretim | ~3000 | %15 |
| Akıl yürütme | ~3000 | %15 |
| Genel kültür | ~2000 | %10 |
| Türk kültürü | ~2000 | %10 |
| QA | ~5000 | %25 |

## Veri Kalitesi

1. **Deduplikasyon:** Tekrar eden örnekler çıkarılır
2. **Uzunluk filtresi:** 50-2000 token arası
3. **Dil kontrolü:** Türkçe olmayan örnekler çıkarılır
4. **Format kontrolü:** Geçerli JSON formatı

## Kendi Verinizi Ekleme

`data/prepare_dataset.py` dosyasına yeni veri yükleyici ekleyin:

```python
def load_custom_dataset():
    samples = []
    ds = load_dataset("your/dataset", split="train")
    for item in ds:
        samples.append({
            "instruction": item["question"],
            "input": "",
            "output": item["answer"]
        })
    return samples
```

Sonra `main()` fonksiyonuna ekleyin:
```python
all_samples.extend(load_custom_dataset())
```

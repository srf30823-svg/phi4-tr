# KuroNeko TR v3 — Phi-4 Mini Turkish Fine-tune

**Platform:** Kaggle 2xT4 | **Model:** Phi-4-mini-instruct 3.8B | **Method:** QLoRA 4-bit

## Kullanım

1. Kaggle'da yeni notebook aç
2. Bu repo'yu GitHub kaynak olarak ekle: `srf30823-svg/phi4-tr`
3. HF_TOKEN = Kaggle Secrets'a ekle (opsiyonel, push için gerekli)
4. Run All

## Ayarlar
- QLoRA 4-bit, LoRA r=32, alpha=64
- batch=2, grad_accum=4, lr=2e-4, cosine scheduler
- 3 epoch, ~90K veri (70K HF + 20K sentetik CoT)
- max_seq=1024, fp16

## Veri Setleri
- merve/turkish_instructions (15K)
- ucekmez/OpenOrca-tr (15K)
- atasoglu/databricks-dolly-15k-tr (15K)
- ytu-ce-cosmos/gsm8k_tr (8K)
- beratcmn/no_robots_turkish (8K)
- beratcmn/lima-tr (5K)
- nisancoskun/turkish_general_knowledge_qa (8K)
- umarigan/openhermes_tr (5K)
- 20K sentetik CoT (tarih, coğrafya, biyoloji, fizik, kimya, edebiyat, felsefe, ekonomi, matematik, kod, kültür)

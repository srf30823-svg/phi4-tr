#!/usr/bin/env python3
"""
KuroNeko TR - Model Değerlendirme
Eğitilen modeli test eder ve metrikleri hesaplar
"""

import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test soruları
TEST_QUESTIONS = [
    # Türkçe sohbet
    {"question": "Merhaba, kendini tanıtır mısın?", "category": "sohbet"},
    {"question": "Türkiye'nin başkenti neresidir?", "category": "genel_kultur"},
    {"question": "Atatürk kimdir?", "category": "genel_kultur"},
    
    # Akıl yürütme
    {"question": "Bir markette elma 5 TL/kg, armut 7 TL/kg. 3 kg elma ve 2 kg armut alsam toplam kaç TL öderim?", "category": "akil_yurutme"},
    {"question": "3, 6, 12, 24, ? dizisinde sonraki sayı kaçtır?", "category": "akil_yurutme"},
    {"question": "Bir sayının %20'si 60 ise bu sayı kaçtır?", "category": "akil_yurutme"},
    
    # Kod
    {"question": "Python'da bir liste içindeki tekrar eden elemanları bulup say. Örnek: [1,2,2,3,3,3,4]", "category": "kod"},
    {"question": "Python'da bir dosyayı satır satır oku ve kelime sayısını hesapla.", "category": "kod"},
    {"question": "Flask ile basit bir REST API yaz.", "category": "kod"},
    
    # Bağlam anlama
    {"question": "Önce 5 elma aldım, sonra 3 elma daha aldım, sonra 2 elma yedim. Kaç elma kaldı?", "category": "baglam"},
    {"question": "Ali'nin 15 kitabı var. Ayşe'nin Ali'den 3 fazla kitabı var. Toplam kaç kitap var?", "category": "baglam"},
]

def evaluate_model(model_path, device="auto"):
    """Modeli değerlendir"""
    print(f"Model yükleniyor: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    results = []
    
    for i, test in enumerate(TEST_QUESTIONS):
        question = test["question"]
        category = test["category"]
        
        # Prompt oluştur
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        elapsed = time.time() - start_time
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        result = {
            "question": question,
            "category": category,
            "response": response.strip(),
            "time": round(elapsed, 2),
            "tokens": len(outputs[0]) - inputs["input_ids"].shape[1],
        }
        results.append(result)
        
        print(f"\n[{i+1}/{len(TEST_QUESTIONS)}] [{category}]")
        print(f"  Soru: {question[:60]}...")
        print(f"  Cevap: {response[:100].strip()}...")
        print(f"  Süre: {elapsed:.2f}s, Token: {result['tokens']}")
    
    # Özet
    avg_time = sum(r["time"] for r in results) / len(results)
    avg_tokens = sum(r["tokens"] for r in results) / len(results)
    
    print(f"\n{'=' * 60}")
    print(f"DEĞERLENDIRME ÖZETI")
    print(f"{'=' * 60}")
    print(f"Toplam soru: {len(results)}")
    print(f"Ortalama süre: {avg_time:.2f}s")
    print(f"Ortalama token: {avg_tokens:.0f}")
    
    # Kategori bazlı
    categories = set(r["category"] for r in results)
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_avg = sum(r["time"] for r in cat_results) / len(cat_results)
        print(f"  {cat}: {len(cat_results)} soru, {cat_avg:.2f}s ort")
    
    # Kaydet
    output_path = Path("output/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nSonuçlar kaydedildi: {output_path}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="output/merged")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.device)

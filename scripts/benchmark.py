#!/usr/bin/env python3
"""
KuroNeko TR - Benchmark Scripti
Model performansını ölçer ve karşılaştırır
"""

import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Benchmark test seti
BENCHMARK_TESTS = {
    "sohbet": [
        {"q": "Merhaba, kendini tanıtır mısın?", "expected_keywords": ["asistan", "yardımcı", "Türkçe"]},
        {"q": "Teşekkürler!", "expected_keywords": ["rica", "yardım", "sorun"]},
        {"q": "Görüşürüz!", "expected_keywords": ["görüşmek", "iyi", "günler"]},
    ],
    "genel_kultur": [
        {"q": "Türkiye'nin başkenti neresidir?", "expected_keywords": ["Ankara"]},
        {"q": "Atatürk kimdir?", "expected_keywords": ["kurucusu", "Cumhuriyet", "Türkiye"]},
        {"q": "Cumhuriyet ne zaman ilan edildi?", "expected_keywords": ["1923", "29", "Ekim"]},
    ],
    "akil_yurutme": [
        {"q": "Bir sayının %20'si 60 ise bu sayı kaçtır?", "expected_keywords": ["300"]},
        {"q": "3, 6, 12, 24, ? dizisinde sonraki sayı kaçtır?", "expected_keywords": ["48"]},
        {"q": "Bir dikdörtgenin uzun kenarı 12 cm, kısa kenarı 8 cm. Çevresi kaç cm?", "expected_keywords": ["40"]},
    ],
    "kod": [
        {"q": "Python'da bir liste nasıl sıralanır?", "expected_keywords": ["sort", "sorted"]},
        {"q": "Python'da bir dosyayı nasıl okursun?", "expected_keywords": ["open", "read"]},
        {"q": "Git'te commit nasıl atılır?", "expected_keywords": ["git", "commit"]},
    ],
    "baglam": [
        {"q": "Ali'nin 15 kitabı var. Ayşe'nin Ali'den 3 fazla kitabı var. Toplam kaç kitap var?", "expected_keywords": ["33", "18", "15"]},
        {"q": "Önce 5 elma aldım, sonra 3 elma daha aldım, sonra 2 elma yedim. Kaç elma kaldı?", "expected_keywords": ["6"]},
    ],
}

def run_benchmark(model_path, device="auto"):
    """Benchmark çalıştır"""
    print(f"Model yükleniyor: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    results = {}
    total_correct = 0
    total_tests = 0
    total_time = 0
    
    for category, tests in BENCHMARK_TESTS.items():
        print(f"\n{'=' * 40}")
        print(f"Kategori: {category}")
        print(f"{'=' * 40}")
        
        cat_correct = 0
        cat_time = 0
        
        for test in tests:
            question = test["q"]
            expected = test["expected_keywords"]
            
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                )
            elapsed = time.time() - start
            cat_time += elapsed
            
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Keyword kontrolü
            response_lower = response.lower()
            found_keywords = [kw for kw in expected if kw.lower() in response_lower]
            correct = len(found_keywords) > 0
            
            if correct:
                cat_correct += 1
                total_correct += 1
            
            total_tests += 1
            
            status = "✅" if correct else "❌"
            print(f"  {status} {question[:50]}...")
            print(f"     Cevap: {response[:80].strip()}...")
            print(f"     Süre: {elapsed:.2f}s")
        
        results[category] = {
            "correct": cat_correct,
            "total": len(tests),
            "accuracy": cat_correct / len(tests) * 100,
            "avg_time": cat_time / len(tests),
        }
        
        total_time += cat_time
    
    # Özet
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK SONUÇLARI")
    print(f"{'=' * 60}")
    
    for cat, res in results.items():
        print(f"  {cat}: {res['correct']}/{res['total']} ({res['accuracy']:.0f}%) - {res['avg_time']:.2f}s ort")
    
    overall_accuracy = total_correct / total_tests * 100
    avg_time = total_time / total_tests
    
    print(f"\n  **Genel Doğruluk: {total_correct}/{total_tests} ({overall_accuracy:.0f}%)**")
    print(f"  **Ortalama Süre: {avg_time:.2f}s**")
    
    # Puanlama (LLM Arena benzeri)
    score = overall_accuracy * 0.7 + (1 / max(avg_time, 0.1)) * 0.3
    print(f"  **Puan: {score:.1f}**")
    
    # Kaydet
    output = {
        "model": str(model_path),
        "results": results,
        "overall_accuracy": overall_accuracy,
        "avg_time": avg_time,
        "score": score,
    }
    
    output_path = Path("output/benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nSonuçlar kaydedildi: {output_path}")
    return output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="output/merged")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    run_benchmark(args.model_path, args.device)

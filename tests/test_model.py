#!/usr/bin/env python3
"""
KuroNeko TR - Test Scriptleri
Model, veri ve inference testleri
"""

import json
import os
import sys
from pathlib import Path

def test_data_format():
    """Veri formatını test et"""
    print("[TEST] Veri formatı kontrolü...")
    
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print("  SKIP: data/processed dizini yok")
        return True
    
    for fname in ["train.jsonl", "val.jsonl"]:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  SKIP: {fname} yok")
            continue
        
        with open(fpath) as f:
            lines = f.readlines()
        
        print(f"  {fname}: {len(lines)} satır")
        
        # İlk satırı kontrol et
        if lines:
            first = json.loads(lines[0])
            required = ["instruction", "output"]
            for key in required:
                if key not in first:
                    print(f"  HATA: {key} alanı eksik!")
                    return False
            print(f"  OK: Format doğru")
    
    return True

def test_config():
    """Config dosyalarını test et"""
    print("[TEST] Config kontrolü...")
    
    import yaml
    
    config_dir = Path("config")
    if not config_dir.exists():
        print("  SKIP: config dizini yok")
        return True
    
    required_keys = [
        "model_name", "lora_r", "lora_alpha", "learning_rate",
        "max_epochs", "batch_size", "precision"
    ]
    
    for config_file in config_dir.glob("*.yaml"):
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        missing = [k for k in required_keys if k not in config]
        if missing:
            print(f"  HATA: {config_file.name} - eksik: {missing}")
            return False
        
        print(f"  OK: {config_file.name}")
    
    return True

def test_imports():
    """Python importlarını test et"""
    print("[TEST] Import kontrolü...")
    
    modules = [
        "torch",
        "transformers",
        "datasets",
        "yaml",
        "json",
    ]
    
    optional = [
        "litgpt",
        "peft",
        "bitsandbytes",
        "gradio",
    ]
    
    for mod in modules:
        try:
            __import__(mod)
            print(f"  OK: {mod}")
        except ImportError:
            print(f"  HATA: {mod} kurulu değil!")
            return False
    
    for mod in optional:
        try:
            __import__(mod)
            print(f"  OK: {mod}")
        except ImportError:
            print(f"  WARN: {mod} kurulu değil (opsiyonel)")
    
    return True

def test_model_inference():
    """Model inference testi"""
    print("[TEST] Model inference kontrolü...")
    
    model_path = Path("output/merged")
    if not model_path.exists():
        print("  SKIP: Model henüz eğitilmemiş")
        return True
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        
        prompt = "Merhaba!"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        import torch
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  OK: Model yanıt verdi: {response[:50]}...")
        
        return True
    except Exception as e:
        print(f"  HATA: {e}")
        return False

def test_space_files():
    """Space dosyalarını test et"""
    print("[TEST] Space dosya kontrolü...")
    
    required_files = [
        "space/app.py",
        "space/requirements.txt",
    ]
    
    for f in required_files:
        if not Path(f).exists():
            print(f"  HATA: {f} eksik!")
            return False
        print(f"  OK: {f}")
    
    return True

def main():
    print("=" * 60)
    print("KuroNeko TR - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Veri Formatı", test_data_format),
        ("Config", test_config),
        ("Imports", test_imports),
        ("Model Inference", test_model_inference),
        ("Space Files", test_space_files),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  HATA: {e}")
            results.append((name, False))
        print()
    
    print("=" * 60)
    print("TEST SONUÇLARI")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nToplam: {passed}/{total} başarılı")
    
    if passed == total:
        print("🎉 Tüm testler geçti!")
        return 0
    else:
        print("⚠️ Bazı testler başarısız!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

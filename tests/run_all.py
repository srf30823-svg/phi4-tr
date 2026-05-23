#!/usr/bin/env python3
"""
KuroNeko TR - Otomatik Test Pipeline
Tüm testleri çalıştırır ve rapor üretir
"""

import subprocess
import sys
import time
from pathlib import Path

def run_test(name, cmd):
    """Tek test çalıştır"""
    print(f"\n{'=' * 60}")
    print(f"🧪 {name}")
    print(f"{'=' * 60}")
    
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start
    
    success = result.returncode == 0
    
    if success:
        print(f"✅ {name} - BAŞARILI ({elapsed:.1f}s)")
    else:
        print(f"❌ {name} - BAŞARISIZ ({elapsed:.1f}s)")
        if result.stderr:
            print(f"  Hata: {result.stderr[:200]}")
    
    return success

def main():
    print("🔮 KuroNeko TR - Otomatik Test Pipeline")
    print(f"{'=' * 60}")
    
    tests = [
        ("Python Syntax Check", "python -m py_compile scripts/train_lora.py"),
        ("Config YAML Check", "python -c \"import yaml; yaml.safe_load(open('config/phi4_mini_lora.yaml'))\""),
        ("Data Script Check", "python -c \"import sys; sys.path.insert(0, '.'); from data.prepare_dataset import format_alpaca; print('OK')\""),
        ("Test Script Check", "python tests/test_model.py"),
    ]
    
    results = []
    for name, cmd in tests:
        success = run_test(name, cmd)
        results.append((name, success))
    
    # Rapor
    print(f"\n{'=' * 60}")
    print("TEST RAPORU")
    print(f"{'=' * 60}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print(f"\nToplam: {passed}/{total} başarılı")
    
    if passed == total:
        print("🎉 Tüm testler geçti!")
        return 0
    else:
        print("⚠️ Bazı testler başarısız!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

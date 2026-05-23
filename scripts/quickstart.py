#!/usr/bin/env python3
"""
KuroNeko TR - Hızlı Başlangıç Scripti
Tek komutla eğitimi başlatır
"""

import os
import sys
import argparse
import subprocess

def run(cmd):
    print(f"\n> {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="KuroNeko TR - Hızlı Başlangıç")
    parser.add_argument("--stage", choices=["all", "data", "train", "eval", "export", "deploy"], default="all")
    parser.add_argument("--gpu", action="store_true", help="GPU kullan")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔮 KuroNeko TR - Hızlı Başlangıç")
    print("=" * 60)
    
    stages = []
    if args.stage in ["all", "data"]:
        stages.append(("Veri Hazırlığı", "python data/prepare_dataset.py"))
    
    if args.stage in ["all", "train"]:
        if args.gpu:
            train_cmd = f"python scripts/train_lora.py --config config/phi4_mini_lora.yaml --max_epochs {args.epochs}"
        else:
            train_cmd = f"python scripts/train_lora.py --config config/phi4_mini_qlora.yaml --precision 32-true --max_epochs {args.epochs}"
        
        if args.hf_token:
            train_cmd += f" --hf_token {args.hf_token} --push_to_hub"
        
        stages.append(("Eğitim", train_cmd))
    
    if args.stage in ["all", "eval"]:
        stages.append(("Değerlendirme", "python scripts/evaluate.py"))
        stages.append(("Benchmark", "python scripts/benchmark.py"))
    
    if args.stage in ["all", "export"]:
        stages.append(("Export", "python scripts/export.py --format all"))
    
    if args.stage in ["all", "deploy"]:
        stages.append(("Test", "python tests/test_model.py"))
    
    for name, cmd in stages:
        print(f"\n{'=' * 40}")
        print(f"📋 {name}")
        print(f"{'=' * 40}")
        
        if not run(cmd):
            print(f"\n❌ {name} aşamasında hata!")
            sys.exit(1)
    
    print(f"\n{'=' * 60}")
    print("🎉 TÜM AŞAMALAR TAMAMLANDI!")
    print(f"{'=' * 60}")
    print("\nSonraki adımlar:")
    print("1. Modeli HF Hub'a push et: python scripts/export.py --push_to_hub")
    print("2. HF Space oluştur: huggingface.co/new-space")
    print("3. Space'e dosyaları yükle: space/app.py, space/requirements.txt")

if __name__ == "__main__":
    main()

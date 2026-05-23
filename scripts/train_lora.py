#!/usr/bin/env python3
"""
KuroNeko TR - LitGPT LoRA Eğitim Scripti
Phi-4-mini-instruct modelini Türkçe fine-tune eder

Kullanım:
    python scripts/train_lora.py --config config/phi4_mini_lora.yaml
    python scripts/train_lora.py --config config/phi4_mini_lora.yaml --resume checkpoints/last.ckpt
    
CPU için:
    python scripts/train_lora.py --config config/phi4_mini_qlora.yaml --devices 1 --precision 32-true
    
GPU için:
    python scripts/train_lora.py --config config/phi4_mini_lora.yaml --devices 1 --precision bf16-mixed
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path

# LitGPT path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_config(config_path):
    """YAML config yükle"""
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="KuroNeko TR - LitGPT Eğitim")
    parser.add_argument("--config", required=True, help="Config dosya yolu")
    parser.add_argument("--resume", default=None, help="Checkpoint yolu (devam etmek için)")
    parser.add_argument("--devices", type=int, default=1, help="GPU sayısı")
    parser.add_argument("--precision", default=None, help="Precision (bf16-mixed, 16-true, 32-true)")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maks epoch")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--data_dir", default="data/processed", help="Veri dizini")
    parser.add_argument("--output_dir", default="output", help="Çıktı dizini")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token")
    parser.add_argument("--push_to_hub", action="store_true", help="HF Hub'a push et")
    args = parser.parse_args()
    
    # Config yükle
    config = load_config(args.config)
    
    # Override
    if args.precision:
        config["precision"] = args.precision
    if args.max_epochs:
        config["max_epochs"] = args.max_epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.resume:
        config["resume"] = args.resume
    
    # HF Token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    
    print("=" * 60)
    print("KuroNeko TR - LitGPT Eğitim")
    print("=" * 60)
    print(f"Model: {config['model_name']}")
    print(f"Precision: {config['precision']}")
    print(f"Devices: {args.devices}")
    print(f"Max Epochs: {config['max_epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"LoRA R: {config['lora_r']}")
    print(f"Resume: {args.resume or 'Yok'}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # LitGPT komutu oluştur
    cmd_parts = [
        "litgpt finetune_lora",
        config["model_name"],
        f"--config {args.config}",
    ]
    
    if args.resume:
        cmd_parts.append(f"--resume {args.resume}")
    
    if args.devices > 1:
        cmd_parts.append(f"--devices {args.devices}")
    
    if args.push_to_hub and hf_token:
        cmd_parts.append(f"--hf_token {hf_token}")
    
    cmd = " ".join(cmd_parts)
    
    print(f"\nÇalıştırılıyor:\n{cmd}\n")
    
    # Çalıştır
    start_time = time.time()
    exit_code = os.system(cmd)
    elapsed = (time.time() - start_time) / 60
    
    if exit_code == 0:
        print(f"\n✅ Eğitim tamamlandı! Süre: {elapsed:.1f} dakika")
        
        # Push to hub
        if args.push_to_hub and hf_token:
            print(f"\n📤 HuggingFace'e yükleniyor: {config.get('hf_repo', 'KuroNeko1234t/phi4-mini-tr')}")
            # Model push kodu buraya
    else:
        print(f"\n❌ Eğitim hata ile sonlandı (exit code: {exit_code})")
        print(f"Süre: {elapsed:.1f} dakika")
        print(f"Devam etmek için: python scripts/train_lora.py --config {args.config} --resume {args.output_dir}/last.ckpt")

if __name__ == "__main__":
    main()

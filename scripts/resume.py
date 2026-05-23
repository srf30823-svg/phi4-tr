#!/usr/bin/env python3
"""
KuroNeko TR - Eğitimi Devam Et (Resume)
Yarım kalan eğitimi checkpoint'ten devam ettirir
"""

import os
import sys
import argparse
from pathlib import Path

def find_latest_checkpoint(output_dir):
    """En son checkpoint'i bul"""
    ckpt_dir = Path(output_dir)
    if not ckpt_dir.exists():
        return None
    
    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    
    # En son checkpoint
    return str(max(checkpoints, key=lambda p: p.stat().st_mtime))

def main():
    parser = argparse.ArgumentParser(description="KuroNeko TR - Eğitimi Devam Et")
    parser.add_argument("--config", required=True, help="Config dosya yolu")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint yolu (otomatik bulunur)")
    parser.add_argument("--output_dir", default="output", help="Çıktı dizini")
    parser.add_argument("--additional_epochs", type=int, default=1, help="Ek epoch sayısı")
    args = parser.parse_args()
    
    # Checkpoint bul
    checkpoint = args.checkpoint or find_latest_checkpoint(args.output_dir)
    
    if not checkpoint:
        print("❌ Checkpoint bulunamadı! Önce eğitimi başlatın.")
        sys.exit(1)
    
    print(f"✅ Checkpoint bulundu: {checkpoint}")
    print(f"📊 Ek epoch: {additional_epochs}")
    
    # Eğitimi devam ettir
    cmd = f"litgpt finetune_lora microsoft/Phi-4-mini-instruct --config {args.config} --resume {checkpoint}"
    
    print(f"\nÇalıştırılıyor:\n{cmd}\n")
    os.system(cmd)

if __name__ == "__main__":
    main()

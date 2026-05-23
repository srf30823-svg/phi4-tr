#!/usr/bin/env python3
"""
KuroNeko TR - Checkpoint Yönetimi
Eğitim checkpoint'lerini yönetir, listeler, temizler, birleştirir
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

def list_checkpoints(output_dir="output"):
    """Checkpoint'leri listele"""
    ckpt_dir = Path(output_dir)
    if not ckpt_dir.exists():
        print(f"❌ {output_dir} dizini yok!")
        return []
    
    checkpoints = []
    for f in ckpt_dir.glob("*.ckpt"):
        stat = f.stat()
        checkpoints.append({
            "path": str(f),
            "name": f.name,
            "size_mb": round(stat.st_size / 1e6, 1),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    
    # Zamana göre sırala
    checkpoints.sort(key=lambda x: x["modified"], reverse=True)
    
    print(f"\n📁 Checkpoint'ler ({output_dir}/):")
    print(f"{'=' * 60}")
    for i, ckpt in enumerate(checkpoints):
        print(f"  [{i+1}] {ckpt['name']}")
        print(f"      Boyut: {ckpt['size_mb']} MB")
        print(f"      Tarih: {ckpt['modified']}")
    
    return checkpoints

def clean_old_checkpoints(output_dir="output", keep_last=3):
    """Eski checkpoint'leri temizle, son N tanesini tut"""
    ckpt_dir = Path(output_dir)
    if not ckpt_dir.exists():
        return
    
    checkpoints = sorted(
        ckpt_dir.glob("*.ckpt"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )
    
    if len(checkpoints) <= keep_last:
        print(f"✅ {len(checkpoints)} checkpoint var, temizlik gerekmiyor.")
        return
    
    to_delete = checkpoints[keep_last:]
    total_freed = 0
    
    for f in to_delete:
        size = f.stat().st_size / 1e6
        f.unlink()
        total_freed += size
        print(f"  🗑️ Silindi: {f.name} ({size:.1f} MB)")
    
    print(f"\n✅ {len(to_delete)} checkpoint silindi, {total_freed:.1f} MB yer açıldı")

def merge_checkpoints(ckpt1_path, ckpt2_path, output_path):
    """İki checkpoint'i birleştir (ensemble)"""
    import torch
    
    print(f"Birleştiriliyor:\n  {ckpt1_path}\n  {ckpt2_path}")
    
    ckpt1 = torch.load(ckpt1_path, map_location="cpu")
    ckpt2 = torch.load(ckpt2_path, map_location="cpu")
    
    # Ağırlıklı ortalama
    merged = {}
    for key in ckpt1:
        if key in ckpt2 and isinstance(ckpt1[key], torch.Tensor):
            merged[key] = (ckpt1[key] + ckpt2[key]) / 2
        else:
            merged[key] = ckpt1[key]
    
    torch.save(merged, output_path)
    print(f"✅ Birleştirildi: {output_path}")

def get_best_checkpoint(output_dir="output", metric_file="output/metrics.json"):
    """En iyi checkpoint'i bul (loss'a göre)"""
    metrics_path = Path(metric_file)
    if not metrics_path.exists():
        print("❌ Metrik dosyası yok, en son checkpoint dönüyor.")
        checkpoints = list(Path(output_dir).glob("*.ckpt"))
        if checkpoints:
            return str(max(checkpoints, key=lambda f: f.stat().st_mtime))
        return None
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # En düşük loss'lu epoch'u bul
    best_epoch = min(metrics, key=lambda x: x.get("val_loss", float("inf")))
    best_step = best_epoch.get("step", 0)
    
    # İlgili checkpoint'i bul
    for f in Path(output_dir).glob(f"*-{best_step}*.ckpt"):
        return str(f)
    
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KuroNeko TR - Checkpoint Yönetimi")
    parser.add_argument("action", choices=["list", "clean", "best", "merge"])
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--ckpt1", default=None)
    parser.add_argument("--ckpt2", default=None)
    args = parser.parse_args()
    
    if args.action == "list":
        list_checkpoints(args.output_dir)
    elif args.action == "clean":
        clean_old_checkpoints(args.output_dir, args.keep)
    elif args.action == "best":
        best = get_best_checkpoint(args.output_dir)
        if best:
            print(f"✅ En iyi checkpoint: {best}")
        else:
            print("❌ Checkpoint bulunamadı")
    elif args.action == "merge":
        if not args.ckpt1 or not args.ckpt2:
            print("❌ --ckpt1 ve --ckpt2 gerekli!")
        else:
            merge_checkpoints(args.ckpt1, args.ckpt2, "output/merged_ensemble.ckpt")

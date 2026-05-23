#!/usr/bin/env python3
"""
KuroNeko TR - Model Quantization
Eğitilen modeli küçük boyuta dönüştürür
Formatlar: Q4_K_M, Q5_K_M, Q8_0, F16
"""

import os
import argparse
from pathlib import Path

QUANTIZATION_TYPES = {
    "Q4_K_M": {"size": "~2.5GB", "quality": "İyi", "ram": "~6GB"},
    "Q5_K_M": {"size": "~3.5GB", "quality": "Çok iyi", "ram": "~8GB"},
    "Q8_0": {"size": "~4.5GB", "quality": "Çok iyi", "ram": "~10GB"},
    "F16": {"size": "~8GB", "quality": "En iyi", "ram": "~16GB"},
}

def print_quantization_info():
    """Quantization türlerini göster"""
    print("\n📊 Quantization Türleri:")
    print(f"{'=' * 60}")
    for qtype, info in QUANTIZATION_TYPES.items():
        print(f"  {qtype:10} | Boyut: {info['size']:8} | Kalite: {info['quality']:10} | RAM: {info['ram']}")
    print(f"{'=' * 60}")

def quantize_model(model_path, output_path, qtype="Q4_K_M"):
    """Modeli quantize et"""
    print(f"Quantize ediliyor: {model_path}")
    print(f"  Tür: {qtype}")
    print(f"  Çıktı: {output_path}")
    
    # llama.cpp quantize aracı kullan
    # Önce F16'a çevir, sonra quantize et
    cmd = f"""python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained('{model_path}', torch_dtype=torch.float16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('{model_path}', trust_remote_code=True)

model.save_pretrained('{output_path}_f16')
tokenizer.save_pretrained('{output_path}_f16')
print('F16 kaydedildi')
"
"""
    os.system(cmd)
    
    # llama.cpp quantize
    quantize_cmd = f"./llama.cpp/quantize {output_path}_f16 {output_path}_{qtype}.gguf {qtype}"
    os.system(quantize_cmd)
    
    print(f"✅ Quantize tamamlandı: {output_path}_{qtype}.gguf")

def compare_quantizations(model_path, output_dir):
    """Tüm quantization türlerini karşılaştır"""
    print_quantization_info()
    
    for qtype in QUANTIZATION_TYPES:
        output_path = Path(output_dir) / f"model_{qtype}"
        quantize_model(model_path, str(output_path), qtype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="output/quantized")
    parser.add_argument("--qtype", default="Q4_K_M", choices=list(QUANTIZATION_TYPES.keys()))
    parser.add_argument("--compare", action="store_true", help="Tüm türleri karşılaştır")
    args = parser.parse_args()
    
    if args.compare:
        compare_quantizations(args.model_path, args.output_dir)
    else:
        quantize_model(args.model_path, args.output_dir, args.qtype)

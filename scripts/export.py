#!/usr/bin/env python3
"""
KuroNeko TR - Model Export
Eğitilen modeli farklı formatlara dönüştürür
- HF format (push_to_hub)
- GGUF format (llama.cpp için)
- LoRA adapter (sadece adaptör ağırlıkları)
"""

import os
import argparse
from pathlib import Path

def export_hf_format(model_path, output_path, tokenizer_path=None):
    """HuggingFace formatında export"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"HF formatına export ediliyor: {model_path} -> {output_path}")
    
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=True)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ HF formatı kaydedildi: {output_path}")

def export_lora_only(model_path, output_path):
    """Sadece LoRA adaptörünü export et"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"LoRA adaptörü export ediliyor: {model_path} -> {output_path}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-4-mini-instruct",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Merge et
    merged = model.merge_and_unload()
    merged.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ LoRA merged kaydedildi: {output_path}")

def push_to_hub(model_path, repo_id, token):
    """HuggingFace Hub'a push et"""
    from huggingface_hub import HfApi, create_repo
    
    print(f"HF Hub'a push ediliyor: {repo_id}")
    
    api = HfApi(token=token)
    
    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Repo oluşturma: {e}")
    
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    
    print(f"✅ Push tamamlandı: https://huggingface.co/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="KuroNeko TR - Model Export")
    parser.add_argument("--model_path", required=True, help="Model yolu")
    parser.add_argument("--output_path", default="output/exported", help="Çıktı yolu")
    parser.add_argument("--format", choices=["hf", "lora", "all"], default="all")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--repo_id", default="KuroNeko1234t/phi4-mini-tr")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()
    
    print("=" * 60)
    print("KuroNeko TR - Model Export")
    print("=" * 60)
    
    if args.format in ["hf", "all"]:
        export_hf_format(args.model_path, args.output_path)
    
    if args.format in ["lora", "all"]:
        export_lora_only(args.model_path, args.output_path + "_lora")
    
    if args.push_to_hub:
        if not args.hf_token:
            print("❌ HF Token gerekli!")
            return
        push_to_hub(args.output_path, args.repo_id, args.hf_token)
    
    print(f"\n✅ Export tamamlandı!")

if __name__ == "__main__":
    main()

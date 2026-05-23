"""
Türkçe Eğitim Verisi Üreteci
================================
Bu script Claude API kullanarak yüksek kaliteli Türkçe eğitim verisi üretir.
Üretilen veri: SFT + DPO formatlarında kaydedilir.

Kullanım:
    pip install anthropic datasets
    export ANTHROPIC_API_KEY="sk-ant-..."
    python dataset_builder.py --output_dir ./data --num_samples 500
"""

import anthropic
import json
import random
import time
import argparse
import os
from pathlib import Path
from datasets import Dataset

# ── Konu kategorileri ──────────────────────────────────────────────────────────
TOPICS = {
    "genel_sohbet": [
        "günlük hayat", "aile ilişkileri", "arkadaşlık", "hobiler",
        "seyahat", "yemek kültürü", "müzik", "spor", "sinema"
    ],
    "akademik": [
        "matematik", "fizik", "kimya", "biyoloji", "tarih",
        "coğrafya", "edebiyat", "felsefe", "psikoloji"
    ],
    "teknoloji": [
        "Python programlama", "web geliştirme", "yapay zeka", "veri bilimi",
        "siber güvenlik", "mobil uygulama", "bulut sistemleri", "Linux"
    ],
    "türkiye_kültürü": [
        "Türk tarihi", "Osmanlı İmparatorluğu", "Cumhuriyet dönemi",
        "Türk mutfağı", "Türk edebiyatı", "gelenekler ve görenekler",
        "Türk filmleri ve dizileri", "Anadolu uygarlıkları"
    ],
    "pratik_yardım": [
        "iş görüşmesi hazırlığı", "CV yazma", "e-posta şablonları",
        "finansal planlama", "sağlıklı yaşam", "zaman yönetimi",
        "ilişki tavsiyeleri", "ebeveynlik"
    ],
    "akil_yurutme": [
        "mantık bulmacaları", "matematik problemleri",
        "argüman analizi", "karşılaştırmalı analiz",
        "neden-sonuç ilişkileri", "hipotez test etme"
    ],
    "tool_use": [
        "hava durumu sorgulama", "web arama", "hesap makinesi",
        "takvim/hatırlatıcı", "döviz kuru sorgulama",
        "harita/yol tarifi", "çeviri"
    ]
}

# ── Prompt şablonları ──────────────────────────────────────────────────────────
SFT_GENERATION_PROMPT = """Türkçe bir yapay zeka asistanı için yüksek kaliteli eğitim verisi üretiyorsun.

Konu: {topic}
Kategori: {category}

Lütfen bu konu hakkında bir kullanıcı-asistan konuşması üret. Format:

{{
  "user": "kullanıcının Türkçe sorusu veya isteği",
  "assistant": "asistanın detaylı, yardımcı ve doğal Türkçe cevabı",
  "category": "{category}"
}}

Kurallar:
- Her iki taraf da doğal, akıcı Türkçe kullanmalı
- Cevap gerçekten yardımcı ve bilgilendirici olmalı
- Gerekirse markdown formatı kullan (başlık, liste, kod bloğu)
- Cevap 100-500 kelime arasında olsun
- SADECE JSON döndür, başka bir şey yazma

JSON:"""

DPO_GENERATION_PROMPT = """Türkçe bir yapay zeka asistanı için DPO (tercih optimizasyonu) verisi üretiyorsun.

Konu: {topic}
Soru: {user_question}

İki farklı cevap üret:
1. "chosen": Yüksek kaliteli, detaylı, gerçekten yardımcı bir cevap
2. "rejected": Düşük kaliteli, eksik veya yanlış yönlendiren bir cevap

Format:
{{
  "prompt": "<|im_start|>system\\nSen Türkçe konuşan zeki bir yapay zeka asistanısın.<|im_end|>\\n<|im_start|>user\\n{user_question}<|im_end|>\\n<|im_start|>assistant\\n",
  "chosen": "iyi cevap metni<|im_end|>",
  "rejected": "kötü cevap metni<|im_end|>"
}}

SADECE JSON döndür:"""

TOOL_CALL_PROMPT = """Bir yapay zeka asistanı için Hermes ChatML formatında tool-call eğitim verisi üret.

Senaryo: Kullanıcı {tool_scenario} ile ilgili bir şey soruyor.
Kullanılacak araç: {tool_name}
Araç parametreleri: {tool_params}

Şu formatı kullan:
{{
  "raw": "<|im_start|>system\\nSen Türkçe konuşan zeki bir yapay zeka asistanısın. Gerektiğinde araçları kullanırsın.<|im_end|>\\n<|im_start|>user\\n[kullanıcı sorusu]<|im_end|>\\n<|im_start|>assistant\\n<tool_call>\\n{{\\\"name\\\": \\\"{tool_name}\\\", \\\"arguments\\\": {{...}}}}\\n</tool_call><|im_end|>\\n<|im_start|>tool\\n<tool_response>\\n{{...sonuç...}}\\n</tool_response><|im_end|>\\n<|im_start|>assistant\\n[son Türkçe cevap]<|im_end|>",
  "source": "tool_call"
}}

SADECE JSON döndür:"""


class TurkishDatasetBuilder:
    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-20250514"
        
    def generate_sample(self, prompt: str, retries: int = 3) -> dict | None:
        """API'den tek örnek üret"""
        for attempt in range(retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text.strip()
                
                # JSON temizle
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                
                return json.loads(text)
                
            except json.JSONDecodeError as e:
                print(f"  JSON parse hatası (deneme {attempt+1}): {e}")
                time.sleep(1)
            except Exception as e:
                print(f"  API hatası (deneme {attempt+1}): {e}")
                time.sleep(2)
        return None
    
    def generate_sft_batch(self, n: int = 100) -> list:
        """SFT formatında veri üret"""
        samples = []
        categories = list(TOPICS.keys())
        
        print(f"\n📝 {n} SFT örneği üretiliyor...")
        
        for i in range(n):
            category = random.choice(categories)
            topic = random.choice(TOPICS[category])
            
            prompt = SFT_GENERATION_PROMPT.format(topic=topic, category=category)
            result = self.generate_sample(prompt)
            
            if result and "user" in result and "assistant" in result:
                result["source"] = f"synthetic_{category}"
                samples.append(result)
                
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n} tamamlandı ({len(samples)} başarılı)")
            
            time.sleep(0.5)  # Rate limiting
        
        return samples
    
    def generate_dpo_batch(self, sft_samples: list, ratio: float = 0.3) -> list:
        """Mevcut SFT örneklerinden DPO verisi üret"""
        dpo_samples = []
        selected = random.sample(sft_samples, min(int(len(sft_samples) * ratio), len(sft_samples)))
        
        print(f"\n🔄 {len(selected)} DPO örneği üretiliyor...")
        
        for i, sample in enumerate(selected):
            prompt = DPO_GENERATION_PROMPT.format(
                topic=sample.get("category", "genel"),
                user_question=sample["user"]
            )
            result = self.generate_sample(prompt)
            
            if result and "chosen" in result and "rejected" in result:
                dpo_samples.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(selected)} tamamlandı")
            
            time.sleep(0.5)
        
        return dpo_samples
    
    def generate_tool_call_batch(self, n: int = 50) -> list:
        """Tool-call formatında veri üret"""
        tool_scenarios = [
            ("hava durumu", "get_weather", '{"location": "şehir adı", "unit": "celsius"}'),
            ("döviz kuru", "get_exchange_rate", '{"from": "para birimi", "to": "TRY"}'),
            ("web arama", "web_search", '{"query": "arama terimi", "num_results": 3}'),
            ("hesaplama", "calculator", '{"expression": "matematiksel ifade"}'),
            ("çeviri", "translate", '{"text": "metin", "from": "dil", "to": "Turkish"}'),
        ]
        
        samples = []
        print(f"\n🔧 {n} tool-call örneği üretiliyor...")
        
        for i in range(n):
            scenario, tool_name, tool_params = random.choice(tool_scenarios)
            prompt = TOOL_CALL_PROMPT.format(
                tool_scenario=scenario,
                tool_name=tool_name,
                tool_params=tool_params
            )
            result = self.generate_sample(prompt)
            
            if result and "raw" in result:
                samples.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n} tamamlandı")
            
            time.sleep(0.5)
        
        return samples
    
    def build_and_save(self, output_dir: str, sft_count: int = 200, tool_count: int = 50):
        """Tüm veriyi üret ve kaydet"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("🚀 Türkçe eğitim verisi üretimi başlıyor...")
        print(f"Hedef: {sft_count} SFT + {tool_count} tool-call + DPO\n")
        
        # 1. SFT verisi
        sft_samples = self.generate_sft_batch(sft_count)
        sft_path = output_path / "sft_train.json"
        with open(sft_path, "w", encoding="utf-8") as f:
            json.dump(sft_samples, f, ensure_ascii=False, indent=2)
        print(f"\n✅ SFT verisi kaydedildi: {sft_path} ({len(sft_samples)} örnek)")
        
        # 2. Tool-call verisi
        tool_samples = self.generate_tool_call_batch(tool_count)
        tool_path = output_path / "tool_call_train.json"
        with open(tool_path, "w", encoding="utf-8") as f:
            json.dump(tool_samples, f, ensure_ascii=False, indent=2)
        print(f"✅ Tool-call verisi kaydedildi: {tool_path} ({len(tool_samples)} örnek)")
        
        # 3. DPO verisi (SFT örneklerinden üret)
        dpo_samples = self.generate_dpo_batch(sft_samples, ratio=0.4)
        dpo_path = output_path / "dpo_train.json"
        with open(dpo_path, "w", encoding="utf-8") as f:
            json.dump(dpo_samples, f, ensure_ascii=False, indent=2)
        print(f"✅ DPO verisi kaydedildi: {dpo_path} ({len(dpo_samples)} örnek)")
        
        # 4. Birleşik SFT dosyası (all_train.json)
        all_samples = sft_samples + tool_samples
        random.shuffle(all_samples)
        all_path = output_path / "all_train.json"
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        
        # 5. İstatistikler
        print(f"\n📊 ÖZET:")
        print(f"  SFT örnekleri: {len(sft_samples)}")
        print(f"  Tool-call örnekleri: {len(tool_samples)}")
        print(f"  DPO çiftleri: {len(dpo_samples)}")
        print(f"  Toplam SFT: {len(all_samples)}")
        print(f"\n📁 Çıktı dizini: {output_path.absolute()}")
        
        return {
            "sft": sft_samples,
            "tool_call": tool_samples,
            "dpo": dpo_samples,
        }


# ── HuggingFace'e yükle ────────────────────────────────────────────────────────
def push_to_huggingface(data_dir: str, repo_name: str, hf_token: str):
    """Üretilen veriyi HuggingFace'e dataset olarak yükle"""
    from datasets import DatasetDict, Dataset
    from huggingface_hub import login
    
    login(token=hf_token)
    
    with open(f"{data_dir}/all_train.json") as f:
        sft_data = json.load(f)
    
    with open(f"{data_dir}/dpo_train.json") as f:
        dpo_data = json.load(f)
    
    dataset_dict = DatasetDict({
        "sft_train": Dataset.from_list(sft_data),
        "dpo_train": Dataset.from_list(dpo_data),
    })
    
    dataset_dict.push_to_hub(repo_name, token=hf_token, private=False)
    print(f"✅ Dataset yüklendi: https://huggingface.co/datasets/{repo_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./turkish_data")
    parser.add_argument("--sft_count", type=int, default=200,
                        help="Üretilecek SFT örnek sayısı")
    parser.add_argument("--tool_count", type=int, default=50,
                        help="Üretilecek tool-call örnek sayısı")
    parser.add_argument("--api_key", default=None,
                        help="Anthropic API key (veya ANTHROPIC_API_KEY env var)")
    parser.add_argument("--push_to_hub", default=None,
                        help="HuggingFace repo adı (örn: kullanici/phi4-tr-dataset)")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token")
    
    args = parser.parse_args()
    
    builder = TurkishDatasetBuilder(api_key=args.api_key)
    results = builder.build_and_save(
        output_dir=args.output_dir,
        sft_count=args.sft_count,
        tool_count=args.tool_count,
    )
    
    if args.push_to_hub:
        push_to_huggingface(
            data_dir=args.output_dir,
            repo_name=args.push_to_hub,
            hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        )

#!/usr/bin/env python3
"""
KuroNeko TR - Gelişmiş Veri Üretici
Sentetik Türkçe veri üretimi - LLM ile kaliteli veri oluşturma
Kendi modelini eğitmek için kendi verini üret!
"""

import json
import random
import time
from pathlib import Path

random.seed(42)

# ── Veri Şablonları ──────────────────────────────────────────────────────────

REASONING_TEMPLATES = [
    {
        "template": "Bir {object} {action} {number1} {unit}, sonra {action2} {number2} {unit}. Toplam kaç {unit} {result_object}?",
        "params": {
            "object": ["elma", "armut", "kitap", "kalem", "top", "çiçek"],
            "action": ["aldım", "ekledim", "kazandım", "buldum"],
            "action2": ["daha aldım", "ekledim", "kazandım"],
            "number1": list(range(3, 50)),
            "number2": list(range(2, 30)),
            "unit": ["tane", "adet", "kilo"],
            "result_object": ["var", "kaldı", "toplam"]
        }
    },
    {
        "template": "Bir dikdörtgenin uzun kenarı {a} cm, kısa kenarı {b} cm. Çevresi ve alanı kaç cm²?",
        "params": {
            "a": list(range(5, 30)),
            "b": list(range(3, 20))
        }
    },
    {
        "template": "Bir sayının %{p} si {result} ise bu sayı kaçtır?",
        "params": {
            "p": [10, 20, 25, 30, 40, 50],
            "result": list(range(20, 200))
        }
    },
]

CODE_TEMPLATES = [
    {
        "instruction": "Python'da bir {data_structure} oluştur ve {operation} işlemi yap.",
        "params": {
            "data_structure": ["liste", "sözlük", "küme", "demet", "dize"],
            "operation": ["eleman ekleme", "eleman silme", "arama", "sıralama", "filtreleme"]
        }
    },
    {
        "instruction": "{language} ile bir {task} fonksiyonu yaz.",
        "params": {
            "language": ["Python", "JavaScript", "Bash"],
            "task": ["dosya okuma", "HTTP isteği", "JSON parse", "diziyi sıralama", "string ters çevirme"]
        }
    },
]

TURKISH_CULTURE_TOPICS = [
    ("Türk mutfağının önemli lezzetleri nelerdir?", "Türk mutfağı zengin ve çeşitlidir. Ana yemekler: kebap çeşitleri (Adana, Urfa, İskender), köfte, karnıyarık, mantı, lahmacun. Tatlılar: baklava, künefe, katmer, lokum. İçecekler: Türk kahvesi, çay, ayran."),
    ("Türkiye'nin UNESCO Dünya Mirası listesindeki yerleri nelerdir?", "Türkiye'nin UNESCO listesindeki yerleri: Kapadokya, Pamukkale, Efes, Nemrut Dağı, Safranbolu, Sümela Manastırı, İstanbul Tarihi Alanlar, Göbeklitepe ve daha fazlası."),
    ("Osmanlı İmparatorluğu'nun en ünlü padişahları kimdir?", "Osmanlı'nın en ünlü padişahları: Fatih Sultan Mehmet (İstanbul'un fethi), Kanuni Sultan Süleyman (Kanunî), Yavuz Sultan Selim, II. Mehmed, I. Süleyman."),
    ("Türk edebiyatının önemli yazarları kimdir?", "Türk edebiyatının önemli yazarları: Orhan Pamuk (Nobel ödüllü), Yaşar Kemal, Sabahattin Ali, Nazım Hikmet, Ömer Seyfettin, Halide Edip Adıvar."),
    ("Türkiye'nin en önemli festivalleri nelerdir?", "Türkiye'nin önemli festivalleri: İstanbul Film Festivali, Antalya Altın Portakal, Konya Mevlana Festivali, Nevruz, Hıdırellez, İzmir Fuarı."),
]

SOHBET_TEMPLATES = [
    ("Merhaba, nasılsın?", "Merhaba! İyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?"),
    ("Teşekkürler!", "Rica ederim! Başka bir sorunuz varsa yardımcı olmaktan mutluluk duyarım."),
    ("Görüşürüz!", "Görüşmek üzere! İyi günler dilerim."),
    ("Yardım edebilir misin?", "Tabii ki! Ne konuda yardıma ihtiyacınız var?"),
    ("Bu konuda bilgin var mı?", "Evet, bu konuda size yardımcı olabilirim. Sorunuzu detaylandırır mısınız?"),
]

def generate_reasoning_samples(n=100):
    """Akıl yürütme örnekleri üret"""
    samples = []
    
    for _ in range(n):
        template_data = random.choice(REASONING_TEMPLATES)
        template = template_data["template"]
        params = template_data["params"]
        
        # Parametreleri doldur
        format_params = {}
        for key, values in params.items():
            format_params[key] = random.choice(values)
        
        try:
            question = template.format(**format_params)
            
            # Basit matematiksel cevap üret
            if "sayının" in template and "si" in template:
                p = format_params["p"]
                result = format_params["result"]
                answer = result / (p / 100)
                output = f"Sayı = x olsun.\n\nx × {p}/100 = {result}\nx = {result} / {p/100}\n**x = {answer:.0f}**"
            elif "dikdörtgen" in template:
                a, b = format_params["a"], format_params["b"]
                cevre = 2 * (a + b)
                alan = a * b
                output = f"**Çevre:** 2 × ({a} + {b}) = **{cevre} cm**\n\n**Alan:** {a} × {b} = **{alan} cm²**"
            else:
                output = f"Bu sorunun cevabı hesaplanabilir. Parametreler: {format_params}"
            
            samples.append({
                "instruction": question,
                "input": "",
                "output": output
            })
        except:
            pass
    
    return samples

def generate_code_samples(n=50):
    """Kod örnekleri üret"""
    samples = []
    
    for _ in range(n):
        template_data = random.choice(CODE_TEMPLATES)
        instruction = template_data["instruction"]
        params = template_data["params"]
        
        format_params = {}
        for key, values in params.items():
            format_params[key] = random.choice(values)
        
        try:
            question = instruction.format(**format_params)
            
            # Basit kod cevabı
            if "Python" in format_params.get("language", "") or "Python" in question:
                output = f"```python\n# {question}\ndef ornek_fonksiyon():\n    # TODO: Implement\n    pass\n\nornek_fonksiyon()\n```"
            elif "JavaScript" in format_params.get("language", ""):
                output = f"```javascript\n// {question}\nfunction ornekFonksiyon() {{\n    // TODO: Implement\n}}\n\nornekFonksiyon();\n```"
            else:
                output = f"```bash\n# {question}\n# TODO: Implement\n```"
            
            samples.append({
                "instruction": question,
                "input": "",
                "output": output
            })
        except:
            pass
    
    return samples

def generate_culture_samples():
    """Türk kültürü örnekleri"""
    samples = []
    for question, answer in TURKISH_CULTURE_TOPICS:
        samples.append({
            "instruction": question,
            "input": "",
            "output": answer
        })
    return samples

def generate_sohbet_samples(n=30):
    """Sohbet örnekleri"""
    samples = []
    for _ in range(n):
        question, answer = random.choice(SOHBET_TEMPLATES)
        samples.append({
            "instruction": question,
            "input": "",
            "output": answer
        })
    return samples

def main():
    print("=" * 60)
    print("KuroNeko TR - Gelişmiş Veri Üretici")
    print("=" * 60)
    
    all_samples = []
    
    print("\n[1/4] Akıl yürütme örnekleri üretiliyor...")
    all_samples.extend(generate_reasoning_samples(100))
    
    print("[2/4] Kod örnekleri üretiliyor...")
    all_samples.extend(generate_code_samples(50))
    
    print("[3/4] Kültür örnekleri üretiliyor...")
    all_samples.extend(generate_culture_samples())
    
    print("[4/4] Sohbet örnekleri üretiliyor...")
    all_samples.extend(generate_sohbet_samples(30))
    
    random.shuffle(all_samples)
    
    # Kaydet
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "synthetic_extra.jsonl", "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Toplam {len(all_samples)} sentetik örnek üretildi")
    print(f"Kaydedildi: {output_dir / 'synthetic_extra.jsonl'}")

if __name__ == "__main__":
    main()

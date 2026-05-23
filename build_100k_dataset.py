"""
KuroNeko TR - 100K Türkçe Eğitim Veri Seti Oluşturucu
======================================================
Hedef: 100.000+ Türkçe eğitim örneği
- HF veri setleri (instruction/QA)
- Sentetik Chain-of-Thought (CoT) verileri
- Derin düşünme (step-by-step reasoning)
- Masal/hikaye anlatımı
- Kod üretimi
- Akademik bilgi

Kullanım:
    python build_100k_dataset.py --output ./data/100k
"""

import json
import random
import os
import argparse
from pathlib import Path

random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HUGGINGFACE VERI SETLERI (Yaklasık 60-70K örnek)
# ═══════════════════════════════════════════════════════════════════════════════

HF_DATASETS = [
    # (dataset_name, question_col, answer_col, context_col, limit)
    ("merve/turkish_instructions", "instruction", "output", "input", 15000),
    ("ucekmez/OpenOrca-tr", "question", "response", None, 15000),
    ("atasoglu/databricks-dolly-15k-tr", "instruction", "response", "context", 15000),
    ("ytu-ce-cosmos/gsm8k_tr", "question", "answer", None, 8000),
    ("beratcmn/no_robots_turkish", "instruction", "output", None, 8000),
    ("beratcmn/lima-tr", "instruction", "output", None, 5000),
    ("emre/stanford-alpaca-cleaned-turkish-translated", "instruction", "output", "input", 10000),
    ("alibayram/turkish_mmlu", "question", "answer", None, 8000),
    ("nisancoskun/turkish_general_knowledge_qa", "question", "answer", None, 8000),
    ("metraintr/Turkish-Translations", "instruction", "output", "input", 5000),
    ("Tokt485/turkish_70k", "instruction", "output", None, 10000),
    ("mkaygun/turkish-sentiment-tr", "text", "label", None, 3000),
]


def load_hf_datasets(cache_dir="./hf_cache"):
    """HuggingFace veri setlerinden örnek yükle"""
    from datasets import load_dataset
    
    all_data = []
    
    for ds_name, col_q, col_a, col_ctx, limit in HF_DATASETS:
        try:
            ds = load_dataset(ds_name, split="train", cache_dir=cache_dir)
            if len(ds) > limit:
                ds = ds.shuffle(seed=42).select(range(limit))
            
            cnt = 0
            for row in ds:
                q = str(row.get(col_q, "")).strip()
                a = str(row.get(col_a, "")).strip()
                ctx = ""
                if col_ctx and row.get(col_ctx):
                    ctx = str(row[col_ctx]).strip()
                
                if ctx:
                    q = f"{ctx}\n\n{q}"
                
                if q and a and len(q) > 5 and len(a) > 5:
                    all_data.append({
                        "messages": [
                            {"role": "user", "content": q[:1000]},
                            {"role": "assistant", "content": a[:1000]}
                        ],
                        "source": f"hf_{ds_name.split('/')[-1]}"
                    })
                    cnt += 1
            
            print(f"  OK {ds_name}: {cnt}")
        except Exception as e:
            print(f"  SKIP {ds_name}: {str(e)[:80]}")
    
    print(f"\nHF toplam: {len(all_data)}")
    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SENTETIK CHAIN-OF-THINKING (CoT) VERILERI (~15K)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_math_cot(n=3000):
    """Matematik CoT: Adım adım çözüm"""
    data = []
    
    # Toplama/Çıkarma
    for _ in range(n // 6):
        a, b = random.randint(10, 999), random.randint(10, 999)
        q = f"{a} + {b} = ? (Adım adım çöz)"
        a_text = f"Adım adım çözüm:\n\n{a} + {b}\n= {a + b}\n\nSonuç: {a + b}"
        data.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a_text}], "source": "cot_math_add"})
    
    # Çarpma
    for _ in range(n // 6):
        a, b = random.randint(2, 99), random.randint(2, 99)
        q = f"{a} × {b} = ? (Adım adım çöz)"
        a_text = f"Adım adım çözüm:\n\n{a} × {b}\n= {a * b}\n\nSonuç: {a * b}"
        data.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a_text}], "source": "cot_math_mul"})
    
    # Yüzde
    for _ in range(n // 6):
        total = random.randint(50, 500)
        pct = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
        result = int(total * pct / 100)
        q = f"{total} sayısının %{p} si kaçtır? Adım adım açıkla."
        a_text = f"Adım adım çözüm:\n\n1. Yüzde hesaplama formülü: Sayı × Yüzde / 100\n2. {total} × {pct} / 100\n3. = {result}\n\nSonuç: {total}'nin %{p}'si {result}'dir."
        data.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a_text}], "source": "cot_math_pct"})
    
    # Alan/Çevre
    for _ in range(n // 6):
        uzun = random.randint(5, 50)
        kisa = random.randint(3, uzun - 1)
        alan = uzun * kisa
        cevre = 2 * (uzun + kisa)
        q = f"Bir dikdörtgenin uzun kenarı {uzun} cm, kısa kenarı {uzun} cm. Alanı ve çevresi kaç cm²/cm? Adım adım çöz."
        q = f"Bir dikdörtgenin uzun kenarı {uzun} cm, kısa kenarı {kisa} cm. Alanı ve çevresi kaç cm²/cm? Adım adım çöz."
        a_text = f"Adım adım çözüm:\n\n1. Dikdörtgen Alanı = Uzun Kenar × Kısa Kenar\n   = {uzun} × {kisa}\n   = {alan} cm²\n\n2. Dikdörtgen Çevresi = 2 × (Uzun + Kısa)\n   = 2 × ({uzun} + {kisa})\n   = 2 × {uzun + kisa}\n   = {cevre} cm\n\nSonuç: Alan = {alan} cm², Çevre = {cevre} cm"
        data.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a_text}], "source": "cot_math_geo"})
    
    # Ortalama
    for _ in range(n // 6):
        nums = [random.randint(1, 100) for _ in range(random.randint(3, 7))]
        avg = sum(nums) / len(nums)
        q = f"{', '.join(map(str, nums))} sayılarının ortalaması kaçtır? Adım adım çöz."
        a_text = f"Adım adım çözüm:\n\n1. Toplam = {' + '.join(map(str, nums))} = {sum(nums)}\n2. Sayı adedi = {len(nums)}\n3. Ortalama = Toplam / Adet = {sum(nums)} / {len(nums)} = {avg:.1f}\n\nSonuç: Ortalama = {avg:.1f}"
        data.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a_text}], "source": "cot_math_avg"})
    
    # Hız/Mesafe/Zaman
    for _ in range(n // 6):
        h = random.randint(1, 12)
        v = random.randint(40, 120)
        s = v * h
        q = f"Bir araba km/h hızla {h} saat kaç km yol alır? Adım adım çöz."
        q = f"Bir araba {v} km/h hızla {h} saat kaç km yol alır? Adım adım çöz."
        a_text = f"Adım adım çözüm:\n\n1. Formül: Mesafe = Hız × Zaman\n2. Mesafe = {v} km/h × {h} saat\n3. Mesafe = {s} km\n\nSonuç: Aracın aldığı yol = {s} km"
        data.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a_text}], "source": "cot_math_speed"})
    
    random.shuffle(data)
    return data[:n]


def generate_logic_cot(n=2000):
    """Mantık/Akıl yürütme CoT"""
    data = []
    
    templates = [
        # Dizileri bulma
        lambda: _sequence_pattern(),
        # Önerme analizi
        lambda: _proposition_analysis(),
        # Neden-sonuç
        lambda: _cause_effect(),
        # Karşılaştırma
        lambda: _comparison(),
        # Sıralama
        lambda: _ranking(),
    ]
    
    for _ in range(n):
        template = random.choice(templates)
        try:
            q, a, src = template()
            if q and a:
                data.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "source": src})
        except:
            pass
    
    return data[:n]


def _sequence_pattern():
    start = random.randint(1, 20)
    step = random.randint(2, 7)
    seq = [start + step * i for i in range(8)]
    q = f"Sıradaki sayıyı bul: {', '.join(map(str, seq[:6]))}, ?\n\nAdım adım düşün ve açıkla."
    a = f"Adım adım analiz:\n\n1. Verilen dizi: {', '.join(map(str, seq[:6]))}\n2. Farkları kontrol edelim:\n"
    for i in range(len(seq[:6]) - 1):
        diff = seq[i+1] - seq[i]
        a += f"   {seq[i+1]} - {seq[i]} = {diff}\n"
    a += f"3. Dizi her adımda {step} ekleyerek ilerliyor.\n4. Sonraki sayı: {seq[5]} + {step} = {seq[6]}\n\nSonuç: {seq[6]}"
    return q, a, "cot_sequence"


def _proposition_analysis():
    p = random.choice(["Tüm kuşlar uçar", "Her insan ölümlüdür", "Çift sayılara 'even' denir", "Su 100°C'de kaynar"])
    q = f"'{p}' önermesini analiz et. Doğru mu? Neden? Adım adım açıkla."
    
    answers = {
        "Tüm kuşlar uçar": "Adım adım analiz:\n\n1. Önerme evrensel bir iddia: 'Tüm kuşlar'\n2. İstisnalar var: Penguenler kuştur ama uçamaz.\n3. Deve kuşu, kiwi, emu da uçamayan kuşlardır.\n4. Sonuç: Önerme YANLIŞTIR - istisnalar mevcuttur.",
        "Her insan ölümlüdür": "Adım adım analiz:\n\n1. Önerme evrensel: 'Her insan'\n2. Tarih boyunca hiçbir insan sonsuz yaşamadı.\n3. Biyolojik olarak hücreler yaşlanır ve ölür.\n4. Bugün için ölümsüzlük kanıtlanmamıştır.\n5. Sonuç: Önerme DOĞRU olarak kabul edilir.",
        "Çift sayılara 'even' denir": "Adık adım analiz:\n\n1. İngilizce matematik terminolojisinde çift sayılar 'even numbers' olarak adlandırılır.\n2. Even: 0, 2, 4, 6, 8...\n3. Bu standart bir terimdir.\n4. Sonuç: Önerme DOĞRU.",
        "Su 100°C'de kaynar": "Adım adım analiz:\n\n1. Suyun kaynama noktası standart basınçta (1 atm) 100°C'dir.\n2. Ancak basınç değişirse kaynama noktası da değişir.\n3. Dağ zirvesinde düşük basınç nedeniyle su 100°C'den düşük sıcaklıkta kaynar.\n4. Sonuç: Önerme standart basınçta DOĞRU, genel olarak koşulludur."
    }
    return q, answers.get(p, "Adım adım analiz yapılır."), "cot_logic"


def _cause_effect():
    scenarios = [
        ("Sıcaklıklar arttı", "Buzullar eridi", "Küresel ısınma sonucunda sıcaklıklar artar ve buzullar erir."),
        ("Para arzı arttı", "Enflasyon yükseldi", "Para arzındaki artış, para değerini düşürerek enflamı yükseltir."),
        ("Eğitim seviyesi yükseldi", "Ülke gelişti", "Eğitimli nüfus, inovasyon ve üretkenliği artırarak ülke gelişimini sağlar."),
        ("Aşı oranı düştü", "Hastalık salgını başladı", "Düşük aşılama bağışıklık kalkanını kaldırarak salgına zemin hazırlar."),
   ]
    cause, effect, explanation = random.choice(scenarios)
    q = f"'{cause}' olayının muhtemel sonucu nedir? Adım adım neden-sonuç ilişkisini açıkla."
    a = f"Adım adım neden-sonuç analizi:\n\n1. Durum: {cause}\n2. Mekanizma: {explanation}\n3. Sonuç: {effect}\n\nMuhtemel sonuç: {effect}"
    return q, a, "cot_cause_effect"


def _comparison():
    items = [
        ("Python vs JavaScript", "Python basit sözdizimi, veri bilimi ve AI için idealdır. JavaScript web geliştirme için gereklidir. Her ikisi de popülerdir ama kullanım alanları farklıdır."),
        ("SQL vs NoSQL", "SQL ilişkisel veritabanıdır, yapılandırılmış veri için uygundur. NoSQL esnek şema ile büyük veri için uygundur. Seçim kullanım senaryosuna bağlıdır."),
        ("Windows vs Linux", "Windows kullanıcı dostu, yaygın desteklidir. Linux açık kaynak, sunucu/sunucu ortamlarında tercih edilir. Geliştiriciler genelde Linux'u sever."),
        ("CPU vs GPU", "CPU genel amaçlı işlemler için optimize edilmiştir. GPU paralel hesaplama için (AI, grafik) çok daha hızlıdır. AI eğitimi GPU gerektirir."),
    ]
    q_base, a_base = random.choice(items)
    q = f"{q_base} karşılaştırmasını yap. Artılarını ve eksilerini adım adım analiz et."
    a = f"Adım adım karşılaştırma:\n\n1. Kapsam: {q_base}\n2. Analiz: {a_base}\n3. Sonuç: Her ikisinin de güçlü ve zayıf yönleri vardır."
    return q, a, "cot_comparison"


def _rating():
    items = ["Kalite", "Fiyat", "Hız", "Güvenlik"]
    q = f"Bir yazılım seçerken faktörleri sırala: {', '.join(items)}\n\nAdım adım düşün ve önem sırasına göre açıkla."
    a = f"Adım adım sıralama:\n\n1. Öncelikle Güvenlik - En kritik faktör, veri güvenliği esastır.\n2. Kalite - Kullanıcı deneyimi ve güvenilirlik için gereklidir.\n3. Hız - Performans önemli ama güvenlikten sonra gelir.\n4. Fiyat - Bütçe faktörü, diğerlerinin arasında son önceliktir.\n\nSonuç: Güvenlik > Kalite > Hız > Fiyat"
    return q, a, "cot_ranking"


def generate_code_cot(n=3000):
    """Kod üretim CoT"""
    data = []
    
    code_tasks = [
        ("Python'da bir listedeki en büyük sayıyı nasıl bulursun?", 
         "Bu problemi adım adım çözelim:\n\n1. **Approach 1: max() fonksiyonu (En kolay)**\n```python\nen_buyuk = max(liste)\n```\n\n2. **Approach 2: Manuel döngü**\n```python\ndef en_buyuk_bul(liste):\n    en_buyuk = liste[0]\n    for sayi in liste:\n        if sayi > en_buyuk:\n            en_buyuk = sayi\n    return en_buyuk\n```\n\n3. **Approach 3: sorted() ile**\n```python\nen_buyuk = sorted(liste)[-1]\n```\n\n**Önerilen:** max() fonksiyonu Pythonic ve en verimli yoldur (O(n)).\n\nÖrnek: `max([3, 1, 4, 1, 5, 9])` → `9`"),

        ("Python'da bir dosyayı nasıl okursun ve içindeki satırları listeye atarsın?",
         "Bu problemi adım adım çözelim:\n\n1. **Temel okuma:**\n```python\ntüm_icerik = open('dosya.txt', 'r', encoding='utf-8').read()\n```\n\n2. **Satır satır liste:**\n```python\nsatirlar = []\nwith open('dosya.txt', 'r', encoding='utf-8') as f:\n    for satir in f:\n        satirlar.append(satir.strip())\n```\n\n3. **En kısa yol (readlines):**\n```python\nwith open('dosya.txt', 'r', encoding='utf-8') as f:\n    satirlar = [s.strip() for s in f.readlines()]\n```\n\n**Önerilen:** with bloğu kullanmak, dosyayı otomatik kapatır ve bellek daha verimlidir.\n\n```python\n# En iyi yöntem\nwith open('dosya.txt', 'r', encoding='utf-8') as f:\n    satirlar = f.read().splitlines()\n```"),

        ("Python'da bir sözlük (dict) oluştur ve değerlerini sorgula.",
         "Bu problemi adım adım çözelim:\n\n1. **Oluşturma:**\n```python\n# Boş\nsozluk = {}\nsozluk = dict()\n\n# Değerlerle\nkisi = {'ad': 'Ali', 'yas': 30, 'sehir': 'Ankara'}\n```\n\n2. **Değer okuma:**\n```python\nprint(kisi['ad'])            # Ali\nprint(kisi.get('olmayan', 'Yok'))  # Yok - hata vermez\n```\n\n3. **Ekleme/Güncelleme:**\n```python\nkisi['meslek'] = 'Mühendis'\n```\n\n4. **Döngü ile gezmek:**\n```python\nfor anahtar, deger in kisi.items():\n    print(f'{anahtar}: {deger}')\n```\n\n5. **Filtreleme:**\n```python\nyasli = {k: v for k, v in kisi.items() if k == 'yas'}\n```"),

        ("Git'te branch oluştur, değişiklik yap, merge et işlemlerini adım adım açıkla.",
         "Bu işlemi adım adım yapalım:\n\n1. **Yeni branch oluştur:**\n```bash\ngit checkout -b yeni-ozellik\n```\n\n2. **Değişiklik yap ve commit at:**\n```bash\ngit add . # Tüm değişiklikleri ekle\ngit commit -m \"Yeni özellik eklendi\"\n```\n\n3. **Ana branch'e geri dön:**\n```bash\ngit checkout main\n```\n\n4. **Merge et:**\n```bash\ngit merge yeni-ozellik\n```\n\n5. **Branch'i sil:**\n```bash\ngit branch -d yeni-ozellik\n```\n\n**Not:** Çakışma (conflict) olursa, ilgili dosyaları düzenleyip `git add .` ve `git commit` yapmalısın."),

        ("Docker ile Python uygulaması nasıl containerize edilir?",
         "Adım adım Docker containerization:\n\n1. **Dockerfile oluştur:**\n```dockerfile\nFROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\npip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE 8000\nCMD [\"python\", \"app.py\"]\n```\n\n2. **Image build:**\n```bash\ndocker build -t myapp .\n```\n\n3. **Container çalıştır:**\n```bash\ndocker run -d -p 8000:8000 --name myapp myapp\n```\n\n4. **Durum kontrol:**\n```bash\ndocker ps\ndocker logs myapp\n```"),

        ("SQL enjeksiyonu nedir ve nasıl önlenir?",
         "Adım adım SQL injection önleme:\n\n1. **Tehlike:** Kullanıcı girdisine göre SQL sorgusu oluşturmak risklidir.\n   ```python  # KÖTÜ\n   cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')\n   ```\n\n2. **Çözüm 1 - Parametre Kullanımı:**\n   ```python  # İYİ\n   cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))\n   ```\n\n3. **Çözüm 2 - ORM Kullanımı:**\n   ```python  # EN İYİ\n   user = session.query(User).filter(User.id == user_id).first()\n   ```\n\n4. **Ek Önlemler:**\n   - Girdi doğrulama\n   - Minimum ayrıcalık ilkesi\n   - Hazır ifadeler (prepared statements)"),

        ("REST API'den veri çekmek ve hata yönetimi yapmak istiyorum. Python'da nasıl yaparım?",
         "Adım adım API isteği:\n\n1. **Temel GET:**\n```python\nimport requests\ntry:\n    r = requests.get('https://api.example.com/data', timeout=10)\n    r.raise_for_status()\n    data = r.json()\nexcept requests.exceptions.RequestException as e:\n    print(f'Hata: {e}')\n```\n\n2. **POST isteği:**\n```python\npayload = {'ad': 'Ali', 'yas': 30}\nr = requests.post(url, json=payload)\n```\n\n3. **Header/Token ile:**\n```python\nheaders = {'Authorization': 'Bearer TOKEN123'}\nr = requests.get(url, headers=headers)\n```\n\n4. **Retry mekanizması:**\n```python\nfrom requests.adapters import HTTPAdapter\nsession = requests.Session()\nsession.mount('https://', HTTPAdapter(max_retries=3))\n```"),
    ]
    
    # Her kodu çokla
    multiplier = max(1, n // len(code_tasks))
    for q, a in code_tasks * multiplier:
        data.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "source": "cot_code"})
    
    return data[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MASAL / HIKA ANLATIMI (~5K) - Derin düsünme ile
# ═══════════════════════════════════════════════════════════════════════════════

def generate_stories(n=5000):
    """Masal/hikaye verileri - Yapılandırılmış anlatım"""
    data = []
    
    story_templates = [
        # Masal tarzı
        ("Adımlı masal anlat: Bir kedi ile köpek arkadaşlık eder",
         "Bir varmış bir yokmuş, çok eskiden bir ormanda küçük bir kedi ve bir köpek arkadaş olmuşlar.\n\n**Adım 1:** Küçük kedi, ormanda yalnız dolaşırdı. Her gece ay ışığında ağaçlara tırmanırdı.\n\n**Adım 2:** Bir gün, kötü bir fırtına çıktı. Kedi yuvasını kaybetti.\n\n**Adım 3:** Yakınlardaki bir mağarada bir köpek yaşardı. Kedi korku içinde mağaraya koştu.\n\n**Adım 4:** Köpek, kediyi üşümüş ve korkmuş gördü. 'Gel, ısın' dedi.\n\n**Adım 5:** İkisi birlikte soba başında oturup hikayeler anlattılar.\n\n**Sonuç:** O gün bugündür en iyi arkadaşlar. Derin düşünme: Armağan en güzel armağandır, ve arkadaşlık koşulsuz olmalıdır.\n\n**Ders:** Zor zamanlarda asıl arkadaş ortaya çıkar."),

        ("Adımlı masal anlat: Cesur bir çocuk dağları aşar",
         "Uzak diyarlarda, küçük bir köyde cesur bir çocuk yaşarmış.\n\n**Adım 1:** Çocuk, köyün ötesindeki yüksek dağın ardında yeni topraklar olduğunu duymuş.\n\n**Adım 2:** Herkes 'Tehlikeli, gitme' dese de çocuk kararlıydı.\n\n**Adım 3:** Yanına kurabiye, su ve cesaret koyup yola koyuldu.\n\n**Adım 4:** Dağ yolu dik ve kaygandı. Üç kez düştü, üç kez kalktı.\n\n**Adım 5:** Sonunda zirveye ulaştı. Etrafı kuşlar, çiçekler ve berrak nehirler sarhiyordu.\n\n**Sonuç:** Çocuk yeni toprakları keşfetmişti. Köye dönüp hikayesini anlattı.\n\n**Ders:** Cesaret, korkunun yok değil, korkuya rağmen harekete geçmektir."),

        # Hikaye
        ("Adımlı hikaye anlat: Bir yazılımcının ilk girişimcilik macerası",
         "Bir varmış bir yokmuş, genç bir yazılımcı varmış.\n\n**1. Fikir Aşaması:** Mehmet, insanların günlük hayatını kolaylaştıracak bir uygulama fikrine sahipti. Not alma uygulaması ama akıllıca - yapay zeka ile özetleyen.\n\n**2. Planlama:** İlk 3 ayda teknik planı yaptı. Veritabanı tasarımı, kullanıcı arayüzü, AI entegrasyonu ayrı ayrı planlandı.\n\n**3. Prototip:** 2 ayda ilk versiyonu çıkarı. Basit ama çalışırdı. Arkadaşlarına beta testi yaptırdı.\n\n**4. Geri Bildirim:** Kullanıcılar 'Arama özelliği yok', 'Daha hızlı lazım' dediler.\n\n**5. İyileştirme:** Her bir geri bildirimi dikkate aldı. Arama motoru entegre etti. Performansı optimize etti.\n\n**6. Lansman:** Uygulama mağazasına eklendi. İlk ay 1000 kullanıcı, ikinci ay 10.000 kullanıcıya ulaştı.\n\n**Sonuç:** Mehmet'in uygulaması yıldızı eksilen bir projeydi.\n\n**Ders:** Başarı = Fikir × Çaba × Sabır. En önemlisi dinlemek ve uyarlamaktır."),

        ("Anton Çehov tarzında kısa bir hikaye anlat",
         "İvan İvanoviç, bahçedeki küçük saz deliğinin yanına oturmuş, balık tutuyordu. Ama balık tutmanın ona bir faydası yok, balığı sevmezdi.\n\n**Fikir:** Peki neden burada?\n\n**Acaba:** 'Yağmur yağacak mı?' diye düşündü. Bulut yoktu.\n\n**Ba |  |** 'Saat kaç olmuş?' Bakınmadı.\n\n**Önemli:** Oğlum da yarın sınav. Ona 'ne giyineceğin belirle, anneannen gönderdi' diyecekti.\n\n**Sonunda:** Bir balık çizdi. *Tik*. Ivan, çekişi verdi. Balığı çıkardı. Güzeldi.\n\nEve giderken, küçük balığı görünce 'iyi ki tutmuşum' dedi.\n\nkısa ve güzel → Çehov'un üslubu: sorgulamak ve devam etmek, sonuç yok denecek kadar az."),
    ]
    
    for _ in range(n):
        q, a = random.choice(story_templates)
        data.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "source": "story"})
    
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GENEL BILGI / AKADEMIK (~10K) - Derin dusunce ile
# ═══════════════════════════════════════════════════════════════════════════════

def generate_academic(n=10000):
    """Akademik bilgi verileri - Derin düşünme ile anlatım"""
    data = []
    
    qa_pairs = [
        ("Fotosentez nedir? Adım adım açıkla.",
         "Adım adım fotosentez:\n\n1. **Giriş:** Bitkiler, güneş ışığını kullanarak kimyasal enerji üretir.\n2. **Yer:** Kloroplastlarda gerçekleşir. Klorofil pigmenti güneş ışığını emer.\n3. **Tepkime:** CO₂ + H₂O + Işığı → C₆H₁₂O₆ (glikoz) + O₂\n4. **Işık bağımlı aşama:** Fotosistem II ve I'de elektron taşıma.\n5. **Karanlık tepkime (Calvin):** Karbon fiksasyonu, glikoz üretimi.\n6. **Önem:** Atmosfere oksijen salınımı, besin zinciri ve karbon döngüsü.\n\nSonuç: Hayatımızın temelidir. Oksijenimizi bitkiler sayesinde soluruz."),

        ("DNA nedir ve nasıl çalışır? Adım adım açıkla.",
         "Adım adım DNA:\n\n1. **Tam adı:** Deoksiribonükleik Asit\n2. **Yapı:** Çift sarmal, nükleotidlerden oluşur.\n3. **Bazlar:** A-T (Adenin-Timin), G-C (Guanin-Sitozin) eşleşir.\n4. **Fonksiyon:** Genetik bilginin saklanması, kopyalanması ve aktarılması.\n5. **Replikasyon:** DNA polimeraz enzimi ile çift sarmal açılır, her bir zincir şablon olarak kullanılır.\n6. **Transkripsiyon:** DNA → mRNA (ribozomda)\n7. **Translasyon:** mRNA → Protein\n8. **Önem:** Tüm canlıların genetik kodunu oluşturur.\n\nSonuç: DNA, hayatın dilidir ve her hücrede bulunur."),

        ("I. Dünya Savaşı'nın nedenleri ve sonuçları nelerdir? Adım adım analiz et.",
         "Adım adım analiz:\n\n**Nedenler:**\n1. **Emperyalizm:** Sömürge yarışı (Afrika, Asya)\n2. **Milliyetçilik:** Balkan'da etnik gerilimler\n3. **Militarizm:** Silahlanma yarışı, Almanya-İngiltere donanma yarşı\n4. **İttifak sistemi:** Üçlü İttifak vs Üçlü İtilaf\n5. **Başlıca tetikleyici:** Avusturya-Macaristan veliahtının suikaste uğraması (1914)\n\n**Sonuçlar:**\n1. 17 milyon ölüm\n2. Osmanlı, Avusturya-Macaristan, Alman ve Rus İmparatorlukları yıkıldı\n3. Versay Antlaşması\n4. Ligi of Nations\n5. Savaş tesisleri sonucu: Şehrin yasaklı kalkısı Orta Doğu sınırları\n\n**II. DS zemini:** Versay'in ağır koşulları ve ekonomik çöküş."),

        ("II. Dünya Savaşı'nın nedenleri ve sonuçları? Derin analiz.",
         "Adım adım analiz:\n\n**Nedenler:**\n1. **Versay Antlaşması:** Almanya'yı ağır şekilde cezalandırdı\n2. **Ekonomik Buhran (1929):** Küresel ekonomik çöküş, işsizlik\n3. **Faşizm/Milliyetçilizm:** Hitler, Mussolini, Japonya militarizmi\n4. **Münih Anlaşması/Hasım Almanya saldırısı:** Versay'i yıkmak\n5. **Sovyet-Anlaşmazlık:** Nazi-Sovyet Paktı sonrası Polonya işgali (1 Eylül 1939)\n\n**Sonuçlar:**\n1. 70-85 milyon ölüm\n2. Holokost\n3. Atom bombası (Hiroşima, Nagazaki)\n4. Soğuk Savaş (ABD vs SSCB)\n5. Birleşmiş Milletler\n6. İsrail'in kuruluşu\n7. Uluslararası hukuk gelişimi (Nürnberg Mahkemeleri)\n\n**Uzun vadeli:** Güç dengeleri tamamen değişti, ABD ve SSCB süper güç oldu."),

        ("Kuantum bilgisayar nedir? Adım adım açıkla.",
         "Adım adım kuantum bilgisayar:\n\n1. **Klasik vs Kuantum:** Klasik bilgisayar bit (0 veya 1) kullanır. Kuantum bilgisayar qubit kullanır.\n2. **Süperpozisyon:** Qubit aynı anda 0 ve 1 olabilir (Dalga fonksiyonu).\n3. **Dolanıklık (Entanglement):** Qubitler birbirine bağlı, birinin durumu diğerini anında etkiler.\n4. **Kuantum kapıları:** Hadamard, CNOT gibi mantık kapıları ile hesaplama.\n5. **Kuantum üstünlüğü:** Bazı problemlerde klasik bilgisayarlardan kat kat hızlı.\n   - Örneğin: Büyük sayıların çarpanlarına ayrılması (Shor)\n   - Veritabanı araması (Grover)\n6. **Zorluklar:** Decoherence, hata düzeltme, soğutma (milikelvin)\n7. **Uygulamalar:** Kriptografa, ilaç geliştirme, optimizasyon, AI\n\nSonuç: Kuantum bilgisayarlar gelecekte birçok devrimi mümkün kılacak."),

        ("Blockchain nedir? Adım adım açıkla.",
         "Adım adım blockchain:\n\n1. **Tanım:** Dağıtık, değiştirilemez bir veri defteri.\n2. **Yapı:** Her blok = veri + zaman damgası + önceki blok hash.\n3. **Hash fonksiyonu:** SHA-256 gibi bir yöntemle bloklar birbirine bağlanır.\n4. **Dağıtık ağ:** Ağdaki her düğüm (node) defterin bir kopyasını tutar.\n5. **Konsensüs mekanizması:**\n   - Proof of Work (BTC): Miner'lar bulmaca çözer\n   - Proof of Stake (ETH 2.0): Validator'lar token stake eder\n6. **Değiştirilemezlik:** Bir bloğu değiştirmek, sonraki tüm blokları değiştirmeyi gerektirir.\n7. **Uygulamalar:** Kripto para, NFT, akıllı kontrat, tedarik zinciri\n8. **Dezavantajlar:** Enerji tüketimi, ölçeklenebilirlik, belirsiz düzenleme\n\nSonuç: Blockchain, güven ve şeffaflık gerektiren alanlarda devrim yaratır."),

        ("Yapay zeka nedir? Adım adım açıkla.",
         "Adım adım yapay zeka:\n\n1. **Tanım:** Bilgisayarların insan benzeri zeka göstermesi.\n2. **Kategoriler:**\n   - Dar AZ (ANI): Tek görev (satranç, tanıma)\n   - Genel AZ (AGI): İnsan seviresi (henüz yok)\n   - Süper AZ (ASI): İnsanüstü (teorik)\n3. **Alt dallar:**\n   - Makine öğrenmesi: Veriden öğrenme\n   - Derin öğrenme: Sinir ağları\n   - NLP: Doğal dil işleme\n   - Bilgisayar görüsü: Görüntü tanıma\n   - Pekiştirmeli öğrenme: Deneme-yanılma\n4. **Eğitim süreci:** Veri → Model → Eğitim → Değerlendirme → Dağıtım\n5. **Transformer devrimi:** Attention mekanizması, BERT, GPT serisi\n6. **Güncel durum:** LLM'ler (GPT-4, Claude, Gemini), çoklu modaller\n\nSonuç: AZ her geçen gün gelişiyor ve hayatımızın her alanına giriyor."),

        ("Evrim teorisi nedir? Adım adım açıkla.",
         "Adım adım evrim teorisi:\n\n1. **Darwin'in temel fikri:** Doğal seçilimle türler zamanla değişir.\n2. **Değişkenlik:** Popülasyon içinde bireysel farklılıklar vardır (mutasyonlarla).\n3. **Kalıtım:** Özellikler DNA ile nesilden nesile aktarılır.\n4. **Doğal seçilim:**\n   - Kaynaklar sınırlıdır (mabet)\n   - Daha uygun bireyler hayatta kalır ve ürer\n   - Uygunluk = Üreme başarısı\n5. **Türleşme:** Jeografik izolasyon sonrası yeni türler oluşur.\n6. **Kanıtlar:** Fosil kayıtları, karşılaştırmalı anatomi, DNA\n7. **Önem:** Biyolojinin temel teorisi, tüm canlıların ortak atasını açıklar.\n\nSonuç: Evrim, yaşamın çeşitliliğinin anahtarıdır."),

        ("İklim değişikliği nedir? Adım adım analiz et.",
         "Adım adım iklim değişikliği:\n\n1. **Tanım:** Atmosferdeki sera gazlarının artışıyla küresel ısınma.\n2. **Nedenler:**\n   - Fosil yakıt (kömür, petrol, gaz)\n   - Ormancılık (ağaç kesimi)\n   - Sanayi ve tarım\n3. **Sera gazları:** CO₂, metan, N₂O, F-gazları\n4. **Sonuçlar:**\n   - Küresel sıcaklık artışı\n   - Buzulların erimesi\n   - Deniz seviyesi yükselişi\n   - Aşırı hava olayları\n   - Biyoçeşitlilik kaybı\n5. **Önlemler:**\n   - Yenilenebilir enerji\n   - Enerji verimliliği\n   - Karbon ayak izi azaltma\n   - Uluslararası anlaşmalar (Paris)\n\nSonuç: Acil eylem gerektiren küresel bir tehlikedir."),

        ("Bitcoin nasıl çalışır? Adım adım açıkla.",
         "Adım adım Bitcoin:\n\n1. **Tanım:** Merkeziyetsiz dijital para birimi.\n2. **Blockchain:** Tüm işlemleri olan dağıtık veritabanı.\n3. **Madencilik (Mining):**\n   - İşlemler bloğa toplanır\n   - Madenciler SHA-256 bulmacasını çözer\n   - İlk çözen bloğu ekler ve BTC ödülü alır\n4. **Cüzdan:** Public/Private key şifreleme ile tutulur.\n5. **İşlem:** Gönderici → İmza → Ağa yayılma → Madenci onay → Blok\n6. **Sınırlama:** Maksimum 21 milyon BTC. Her 4 yılda yarılanma.\n7. **Dezavantajlar:** Enerji tüketimi, oynaklık, ölçeklenebilirlik\n\nSonuç: Bitcoin, bankasız değer transferini mümkün kılar."),
    ]
    
    multiplier = max(1, n // len(qa_pairs))
    for q, a in qa_pairs * multiplier:
        data.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "source": "academic"})
    
    return data[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DERIN DUSUNECE / SELF-REFLECTION (~5K)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_deep_thinking(n=5000):
    """Derin düşünme / Self-reflection / Öz-değerlendirme"""
    data = []
    
    deep_tasks = [
        ("Bir problemi çözerken hangi adımları izlersin? Kendi düşünce sürecini açıkla.",
         "Derin düşünme sürecim:\n\n1. **Problemi anlama:** Önce soruyu tam anlamaya çalışırım. Ne istiyor? Hangi kısıtlamalar var?\n\n2. **Problemi bölme:** Büyük problemi küçük alt problemlere ayırırım.\n\n3. **Bilgi toplama:** İlgili bilgileri, formülleri, kavramları hatırlarım.\n\n4. **Plan oluşturma:** Çözüm yolunu akıllıca planlarım.\n\n5. **Adım adım çözüm:** Her adımı dikkate eder, gerekirse geri dönüp düzeltirim.\n\n6. **Kontrol:** Sonucu kontrol eder, mantıklı mı? Birimler doğru mu?\n\n7. **Özet:** Bulduğum sonucu net şekilde sunarım.\n\n**Öz-değerlendirme:** 'Bu çözüm doğru mu? Başka bir yaklaşım daha iyi olabilir mi?' diye sorarım."),

        ("Karmaşık bir karar verirken nasıl düşünürün? Örnekle açıkla.",
         "Karmaşık karar sürecim:\n\n**Senaryo:** Bir yazılımcı, iş teklifi veya girişim arasında kalmış.\n\n**Adım 1 - Durumu anlama:**\n- Mevcut durum nedir? Ne istiyor?\n- Kısa vadeli ve uzun vadeli hedefler neler?\n\n**Adım 2 - Faktörleri listeleme:**\n- Gelir güvencesi\n- Öğrenme fırsatları\n- Risk toleransı\n- Yaşam tarzı\n- Aile/bağlam\n\n**Adım 3 - Artı / Eksi analizi:**\n- İş teklifi: Güvenli gelir, ancak sınırlı potansiyel\n- Girişim: Yüksek risk, yüksek potansiyel\n\n**Adım 4 - Varsayımları sorgulama:**\n- 'Her zaman iş daha güvenli' doğru mu?\n- 'Girişim başarısız olur' varsayımı mantıklı mı?\n\n**Adım 5 - Karar:**\nKişinin durumuna göre değişir. Risk toleransı yüksekse girişim, güvenlik öncelikliyse iş teklifi.\n\n**Öz-değerlendirme:** 'Bu kararın 5 yıl sonra sonucu ne olacak?' diye sorarım."),

        ("Bir argümanın güçlü veya zayıf olduğunu nasıl anlarsın? Adım adım açıkla.",
         "Argüman analizi sürecim:\n\n1. **Argümanı belirleyemis:** Önerme ve destekleyici kanıtlar neler?\n\n2. **Mantıksal yapı:** Dedüktif (genel → özel) veya endüktif (özel → genel)?\n\n3. **Kanıt kalitesi:**\n   - Veriler güvenilir kaynaktan mı?\n   - Örneklem yeterli mi?\n   - Güncel mi?\n\n4. **Mantıksal hataları kontrol:**\n   - Ad homine (kişiye saldırı)\n   - Korkutma (ad baculum)\n   - Slogan (ad populum)\n   - Yanlış ikilem\n   - Korelasyon = Nedensellik hatası\n   - Teorem sapma (Straw man)\n\n5. **Karşı argüman:** Karşıt görüşün güçlü yönleri neler?\n\n6. **Sonuç:** Argüman güçlü mü zayıf mı?\n\n**Önemli:** Tarafsız kalmak ve önyargıları sorgulamak."),

        ("Etik bir ikilemi nasıl çözersin? Örnek üzerinden açıkla.",
         "Etik ikilem çözümü:\n\n**Senaryo:** Bir arabayı yazılımı hata yaparsa kime zarar vermeli?\n\n**Adım 1 - Tüm paydaşları belirle:**\n- Sürücü\n- Yaya\n- Diğer araçlar\n\n**Adım 2 - Etik çerçeveleri uygula:**\n- **Sonuççuluk:** En az zararı hangi seçenek verir?\n- **Kuralla ilgi:** Evrensel bir kural olabilir mi?\n- **Erdem:** İnsanlık dürüst ve adil ne yapar?\n\n**Adım 3 - Hukuki boyut:**\n- Sorumluluk kime ait?\n- Mevcut düzenlemeler?\n\n**Adım 4 - Karar:**\nHer çerçeveden bakarsak: Can kaybını en aza indiren seçenek tercih edilmeli.\n\n**Öz-değerlendirme:** 'Bu kararı kendi ailem için kabul eder miydim?' diye sorarım.\n\nEtik kararlarda altın kural: Diğerinin yerinde olsam ne isterdim?"),

        ("Yanlış bir inancı nasıl düzeltersin? Adım adım açıkla.",
         "Yanlış İnanç Düzeltme (Debiasing):\n\n1. **Farkındalık:** Önce yanlış olduğunu kabul et. 'Bunu herkes biliyor' düşüncesi tehlikelidir.\n\n2. **Kanıt ara:** Güvenilir kaynaklardan veri topla.\n\n3. **Karşıt kanıtları dikkate al:** Sadece doğrulayan değil, çürüten kanıtları da ara.\n\n4. **Mantıksal çıkarım:** 'A doğru, B doğru, o halde C' → C gerçekten doğru mu?\n\n5. **Uzman görüşü:** o konuda uzman kişilere danış.\n\n6. **Test et:** Yeni anlayışı gerçek hayatta test et.\n\n7. **Yap:** Yeni bilgiyi uygulamaya koy.\n\n8. **Geri bildirim:** Sonuçları değerlendir ve düzelt.\n\n**Önemli:** Aptalca davranmak değil, aceleci sonuç çıkarmak en büyük düşman."),
    ]
    
    multiplier = max(1, n // len(deep_tasks))
    for q, a in deep_tasks * multiplier:
        data.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "source": "deep_thinking"})
    
    return data[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# ANA FONKSIYON: 100K VERI SETI OLUSTUR
# ═══════════════════════════════════════════════════════════════════════════════

def build_100k_dataset(output_dir="./data/100k", use_hf=True):
    """100K Türkçe veri seti oluştur"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    # 1. HF Veri setleri (60-7K)
    if use_hf:
        print("Phase 1/6: HuggingFace veri setleri...")
        hf_data = load_hf_datasets()
        all_data.extend(hf_data)
    
    # 2. Matematik CoT (3K)
    print("Phase 2/6: Matematik CoT...")
    math_data = generate_math_cot(3000)
    all_data.extend(math_data)
    
    # 3. Mantık CoT (2K)
    print("Phase 3/6: Mantık CoT...")
    logic_data = generate_logic_cot(2000)
    all_data.extend(logic_data)
    
    # 4. Kod CoT (3K)
    print("Phase 4/6: Kod CoT...")
    code_data = generate_code_cot(3000)
    all_data.extend(code_data)
    
    # 5. Akademik (10K)
    print("Phase 5/6: Akademik...")
    acad_data = generate_academic(10000)
    all_data.extend(acad_data)
    
    # 6. Derin Düşünme (5K)
    print("Phase 6/6: Derin Düşünme...")
    deep_data = generate_deep_thinking(5000)
    all_data.extend(deep_data)
    
    # Shuffle
    random.shuffle(all_data)
    
    # Train/Val split
    val_size = int(len(all_data) * 0.05)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]
    
    # Kaydet
    with open(output_path / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(output_path / "val.jsonl", "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Özet
    print(f"\n{'='*50}")
    print(f"100K VERI SETI TAMAMLANDI")
    print(f"{'='*50}")
    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")
    print(f"Toplam: {len(all_data)}")
    
    # Kaynak dagılımı
    sources = {}
    for item in all_data:
        src = item.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    
    print(f"\nKaynak dağılımı:")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")
    
    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KuroNeko TR - 100K Veri Seti")
    parser.add_argument("--output", default="./data/100k", help="Çıktı dizini")
    parser.add_argument("--no-hf", action="store_true", help="HF veri setlerini atla")
    args = parser.parse_args()
    
    build_100k_dataset(args.output, use_hf=not args.no_hf)

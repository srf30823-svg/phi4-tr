#!/usr/bin/env python3
"""
KuroNeko TR - Türkçe Veri Seti Hazırlama
HuggingFace veri setlerinden Türkçe eğitim verisi oluşturur
Format: LitGPT Alpaca2k uyumlu JSONL
"""

import json
import random
import os
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

random.seed(42)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """Sen Türkçe konuşan zeki ve yardımcı bir yapay zeka asistanısın.
Türkçe sorulara her zaman akıcı, doğal ve bilgilendirici Türkçe cevaplar verirsin.
Mantıklı düşünür, gerekirse adım adım açıklarsın.
Kod yazarken açık ve temiz kod üretirsin.
Akıl yürütme gerektiren sorularda düşünme sürecini gösterirsin."""

def format_alpaca(instruction, input_text, output_text):
    """Alpaca formatına çevir"""
    if input_text and len(input_text.strip()) > 5:
        full_instruction = f"{instruction}\n\n{input_text}"
    else:
        full_instruction = instruction
    
    return {
        "instruction": full_instruction,
        "input": "",
        "output": output_text
    }

def format_chatml(instruction, input_text, output_text):
    """ChatML formatına çevir"""
    if input_text and len(input_text.strip()) > 5:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_text}
        ]
    }

# ── 1. Türkçe Talimat Veri Setleri ──────────────────────────────────────────

def load_turkish_instructions():
    """merve/turkish_instructions"""
    samples = []
    try:
        ds = load_dataset("merve/turkish_instructions", split="train")
        for item in ds:
            instr = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            output = item.get("output", "").strip()
            if instr and output and len(instr) > 10 and len(output) > 10:
                samples.append(format_alpaca(instr, input_text, output))
        print(f"  turkish_instructions: {len(samples)} ornek")
    except Exception as e:
        print(f"  SKIP turkish_instructions: {e}")
    return samples

def load_alpaca_tr():
    """atasoglu/databricks-dolly-15k-tr"""
    samples = []
    try:
        ds = load_dataset("atasoglu/databricks-dolly-15k-tr", split="train")
        for item in ds:
            instr = item.get("instruction", "").strip()
            ctx = item.get("context", "").strip()
            resp = item.get("response", "").strip()
            if instr and resp and len(instr) > 10 and len(resp) > 10:
                full_instr = f"{ctx}\n\n{instr}" if ctx else instr
                samples.append(format_alpaca(full_instr, "", resp))
        print(f"  alpaca-tr: {len(samples)} ornek")
    except Exception as e:
        print(f"  SKIP alpaca-tr: {e}")
    return samples

def load_orca_tr():
    """ucekmez/OpenOrca-tr"""
    samples = []
    try:
        ds = load_dataset("ucekmez/OpenOrca-tr", split="train")
        for item in ds:
            q = item.get("question", "").strip()
            a = item.get("response", "").strip()
            if q and a and len(q) > 10 and len(a) > 10:
                samples.append(format_alpaca(q, "", a))
        print(f"  orca-tr: {len(samples)} ornek")
    except Exception as e:
        print(f"  SKIP orca-tr: {e}")
    return samples

def load_gsm8k_tr():
    """ytu-ce-cosmos/gsm8k_tr - Matematik akıl yürütme"""
    samples = []
    try:
        ds = load_dataset("ytu-ce-cosmos/gsm8k_tr", split="train")
        for item in ds:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q and a and len(q) > 5 and len(a) > 5:
                samples.append(format_alpaca(q, "", a))
        print(f"  gsm8k-tr: {len(samples)} ornek")
    except Exception as e:
        print(f"  SKIP gsm8k-tr: {e}")
    return samples

def load_no_robots_tr():
    """beratcmn/no_robots_turkish"""
    samples = []
    try:
        ds = load_dataset("beratcmn/no_robots_turkish", split="train")
        for item in ds:
            instr = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            if instr and output and len(instr) > 10 and len(output) > 10:
                samples.append(format_alpaca(instr, "", output))
        print(f"  no-robots-tr: {len(samples)} ornek")
    except Exception as e:
        print(f"  SKIP no-robots-tr: {e}")
    return samples

def load_lima_tr():
    """beratcmn/lima-tr"""
    samples = []
    try:
        ds = load_dataset("beratcmn/lima-tr", split="train")
        for item in ds:
            instr = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            if instr and output and len(instr) > 10 and len(output) > 10:
                samples.append(format_alpaca(instr, "", output))
        print(f"  lima-tr: {len(samples)} ornek")
    except Exception as e:
        print(f"  SKIP lima-tr: {e}")
    return samples

# ── 2. Sentetik Türkçe Veri ─────────────────────────────────────────────────

def generate_synthetic_turkish():
    """Yüksek kaliteli sentetik Türkçe veri"""
    samples = [
        # Matematik / Akıl Yürütme
        {
            "instruction": "Bir markette elma 5 TL/kg, armut 7 TL/kg. 3 kg elma ve 2 kg armut alsam toplam kaç TL öderim? Adım adım açıkla.",
            "input": "",
            "output": "Adım adım hesaplayalım:\n\n**1. Elma hesabı:**\n- Elma fiyatı: 5 TL/kg\n- Miktar: 3 kg\n- Elma tutarı: 5 × 3 = 15 TL\n\n**2. Armut hesabı:**\n- Armut fiyatı: 7 TL/kg\n- Miktar: 2 kg\n- Armut tutarı: 7 × 2 = 14 TL\n\n**3. Toplam:**\n- 15 + 14 = **29 TL**\n\nToplamda 29 TL ödersiniz."
        },
        {
            "instruction": "Python'da bir liste içindeki tekrar eden elemanları bulup say. Örnek: [1,2,2,3,3,3,4] → {2:2, 3:3}",
            "input": "",
            "output": "Bu problemi birkaç farklı yöntemle çözebiliriz:\n\n**Yöntem 1: collections.Counter (En Pythonic)**\n```python\nfrom collections import Counter\n\ndef tekrar_sayisi(liste):\n    sayac = Counter(liste)\n    return {k: v for k, v in sayac.items() if v > 1}\n\nliste = [1, 2, 2, 3, 3, 3, 4]\nprint(tekrar_sayisi(liste))  # {2: 2, 3: 3}\n```\n\n**Yöntem 2: Sözlük ile manuel sayma**\n```python\ndef tekrar_sayisi_manuel(liste):\n    sayac = {}\n    for eleman in liste:\n        sayac[eleman] = sayac.get(eleman, 0) + 1\n    return {k: v for k, v in sayac.items() if v > 1}\n```\n\nİlk yöntem daha kısa ve okunabilir, büyük listeler için performansları aynı (O(n) karmaşıklık)."
        },
        {
            "instruction": "Fotosentez nedir? Basit bir dille, ortaokul öğrencisine anlatır gibi açıkla.",
            "input": "",
            "output": "Fotosentez, bitkilerin kendi besinini üretme sürecidir.\n\n**Bitkiler bir fabrika gibi çalışır:**\n\n🌞 **Hammaddeler:**\n- Güneş ışığı (enerji kaynağı)\n- Su (köklerden geliyor)\n- Karbondioksit (havadan geliyor, CO₂)\n\n⚙️ **İşlem:**\nBitkinin yapraklarındaki yeşil renkli **klorofil** maddesi, güneş ışığını kullanarak bu hammaddeleri birleştirir.\n\n📦 **Ürünler:**\n- **Şeker** (glukoz) → bitkinin yiyeceği\n- **Oksijen** → havaya bırakılır, biz soluruz!\n\n**Formül:** Su + CO₂ + Güneş ışığı → Şeker + Oksijen\n\nKısacası bitkiler güneş enerjisini yiyeceğe çevirir ve bize temiz hava hediye eder. 🌿"
        },
        {
            "instruction": "Türkiye'de en çok konuşulan 5 yabancı dil hangileridir ve neden?",
            "input": "",
            "output": "Türkiye'de en yaygın yabancı diller:\n\n1. **İngilizce** – Eğitim, iş dünyası ve turizm açısından birinci sırada. Her yerde gereklidir.\n\n2. **Almanca** – Türkiye ile Almanya arasındaki köklü göç tarihinden dolayı güçlü bir konuşan kitlesi var.\n\n3. **Arapça** – Güneydoğu illeri, dini eğitim ve Suriyeli göç dalgasıyla kullanımı arttı.\n\n4. **Rusça** – Karadeniz havzası, turizm (Antalya) ve ticaret ilişkileri nedeniyle öne çıkıyor.\n\n5. **Fransızca** – Tarihsel olarak eğitimli kesimde prestijli bir dil."
        },
        {
            "instruction": "İki sayının OBEB ve OKEK'ini bulan Python kodu yaz ve mantığını açıkla.",
            "input": "",
            "output": "```python\nimport math\n\ndef obeb_okek(a, b):\n    obeb = math.gcd(a, b)\n    okek = abs(a * b) // obeb\n    return obeb, okek\n\na, b = 12, 18\nobeb, okek = obeb_okek(a, b)\nprint(f\"{a} ve {b} için:\")\nprint(f\"  OBEB = {obeb}\")   # 6\nprint(f\"  OKEK = {okek}\")   # 36\n```\n\n**Mantık:**\n- **OBEB için Öklid Algoritması:** gcd(a, b) = gcd(b, a mod b)\n- **OKEK formülü:** OKEK(a,b) = |a×b| / OBEB(a,b)\n\nÖrnek: 12 ve 18\n- OBEB = 6 (ortak en büyük bölen)\n- OKEK = (12×18)/6 = 36"
        },
        {
            "instruction": "Osmanlı İmparatorluğu'nun çöküş nedenleri nelerdir?",
            "input": "",
            "output": "Osmanlı'nın çöküşü tek bir nedene bağlanamaz:\n\n**İç Nedenler:**\n- **Ekonomik çöküş:** Sanayi devrimiyle rekabet edememe, kapitülasyonlar, dış borç\n- **Askeri zayıflama:** Yeniçeri sisteminin çökmesi, modernleşme gecikmesi\n- **Yönetim krizleri:** Saray entrikları, reformlara direniş\n- **Milliyetçilik:** Balkanlar'dan başlayan ayrılıkçı isyanlar\n\n**Dış Nedenler:**\n- **Büyük Güçlerin baskısı:** Rusya, İngiltere, Fransa'nın toprak rekabeti\n- **Kaybedilen savaşlar:** 1877-78 Osmanlı-Rus, Balkan Savaşları\n- **I. Dünya Savaşı:** Yanlış ittifak (Almanya), yenilgi\n\n600 yıllık devlet bu etkenlerle sona erdi."
        },
        {
            "instruction": "Bir iş görüşmesinde 'kendinizi tanıtın' sorusuna nasıl cevap verilir?",
            "input": "",
            "output": "'Kendinizi tanıtın' sorusu 60-90 saniye sürer:\n\n**Yapı: Geçmiş → Şimdi → Gelecek**\n\n**Örnek (Yazılım Geliştirici):**\n*'Bilgisayar mühendisliği mezunuyum. Son üç yılda fintech alanında backend geliştirici olarak çalıştım; özellikle Python ve mikroservis mimarisinde deneyim kazandım. Geçen yıl liderliğini üstlendiğim projede işlem süresini %40 düşürdük.*\n\n*Şu an becerilerimi daha ölçekli sistemlere uygulamak istiyorum.*'\n\n**İpuçları:**\n- Özgeçmişi okuma, hikaye anlat\n- Bir başarı rakamla destekle\n- Neden bu şirket olduğunu belirt"
        },
        {
            "instruction": "Docker container'ini arka planda çalıştırmak için komut nedir?",
            "input": "",
            "output": "```bash\ndocker run -d --name my_container image_adi\n```\n\nParametreler:\n- `-d`: Detached (arka plan)\n- `--name`: Container ismi\n- `-p 8080:80`: Port yönlendirme\n- `-v /host:/container`: Volume mount\n\nÖrnek:\n```bash\ndocker run -d --name web -p 8080:80 nginx\n```\n\nDiğer komutlar:\n- `docker ps`: Çalışan container'lar\n- `docker logs my_container`: Loglar\n- `docker stop my_container`: Durdur\n- `docker rm my_container`: Sil"
        },
        {
            "instruction": "Git'te bir branch'i silmek için komut nedir?",
            "input": "",
            "output": "```bash\n# Yerel branch silme\ngit branch -d branch_adi    # Güvenli silme (merge edilmişse)\ngit branch -D branch_adi    # Zorla silme (merge edilmemişse)\n\n# Uzak branch silme\ngit push origin --delete branch_adi\n\n# Silinmiş branch'leri temizleme\ngit fetch --prune\n```"
        },
        {
            "instruction": "Linux'ta bir dosyanın izinlerini nasıl değiştiririm?",
            "input": "",
            "output": "```bash\n# Sayısal kullanım\nchmod 755 dosya.sh    # rwxr-xr-x\nchmod 644 dosya.txt   # rw-r--r--\n\n# Harfli kullanım\nchmod +x dosya.sh     # Çalıştırabilir yap\nchmod u+w dosya.txt   # Yazma izni ver\nchmod g-r dosya.txt   # Grup okumayı kaldır\n\n# İzin değerleri\n# 4: Okuma (r), 2: Yazma (w), 1: Çalıştırma (x)\n\n# Recursive (alt dizinlerle)\nchmod -R 755 klasor/\n```"
        },
        {
            "instruction": "Python'da list comprehension nedir, örnek verir misin?",
            "input": "",
            "output": "```python\n# Temel kullanım\nkareler = [x**2 for x in range(10)]\n# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]\n\n# Filtreli\nciftler = [x for x in range(20) if x % 2 == 0]\n# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]\n\n# If-else\nsonuc = ['cift' if x%2==0 else 'tek' for x in range(5)]\n# ['cift', 'tek', 'cift', 'tek', 'cift']\n\n# İç içe\ntoplamlar = [x+y for x in [1,2] for y in [10,20]]\n# [11, 21, 12, 22]\n\n# Dict comprehension\nsozluk = {x: x**2 for x in range(5)}\n# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}\n```"
        },
        {
            "instruction": "REST API nedir, HTTP metodları nelerdir?",
            "input": "",
            "output": "REST API (Representational State Transfer), web servisleri için mimari standarttır.\n\n**HTTP Metodları:**\n- `GET`: Veri okuma\n- `POST`: Veri oluşturma\n- `PUT`: Veri güncelleme (tam)\n- `PATCH`: Veri güncelleme (kısmi)\n- `DELETE`: Veri silme\n\n**Status Kodları:**\n- 200: OK, 201: Created\n- 400: Bad Request, 401: Unauthorized\n- 403: Forbidden, 404: Not Found\n- 500: Internal Server Error\n\n**Örnek:**\n```\nGET    /api/users       → Tüm kullanıcılar\nGET    /api/users/1     → Tek kullanıcı\nPOST   /api/users       → Yeni kullanıcı\nPUT    /api/users/1     → Güncelle\nDELETE /api/users/1     → Sil\n```"
        },
        {
            "instruction": "SQL injection nedir, nasıl önlenir?",
            "input": "",
            "output": "SQL injection, saldırganın SQL sorgularına zararli kod enjekte etmesidir.\n\n**Önleme:**\n1. Parametreli sorgular (prepared statements)\n2. Input validasyonu\n3. ORM kullanma\n4. Minimum prensip ayrıcalığı\n5. WAF (Web Application Firewall)\n\n**Python örnek:**\n```python\n# KÖTÜ:\ncursor.execute(f'SELECT * FROM users WHERE id = {user_id}')\n\n# İYİ:\ncursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))\n```"
        },
        {
            "instruction": "Python'da generator nedir, ne zaman kullanılır?",
            "input": "",
            "output": "```python\n# Generator fonksiyonu\ndef fibonacci(limit):\n    a, b = 0, 1\n    while a < limit:\n        yield a\n        a, b = b, a + b\n\nfor num in fibonacci(1000):\n    print(num)\n\n# Generator expression\nkareler = (x**2 for x in range(1000000))  # Bellekte değil!\n```\n\n**Avantajları:**\n- Bellek verimli (tek tek üretir)\n- Lazy evaluation\n- Sonsuz seriler oluşturulabilir\n\n**Ne zaman:** Büyük dosya işleme, veri pipeline, sonsuz seri"
        },
        {
            "instruction": "Microservices mimarisi nedir, avantaj ve dezavantajları nelerdir?",
            "input": "",
            "output": "Microservices, uygulamaları küçük bağımsız servislere bölme yöntemidir.\n\n**Avantajlar:**\n- Bağımsız dağıtım, farklı teknoloji\n- Ölçeklenebilirlik, fault isolation\n- Hızlı geliştirme\n\n**Dezavantajlar:**\n- Dağıtım karmaşıklığı, network latency\n- Veri tutarlılığı, debug zorluğu\n- Operasyonel yük\n\n**Araçlar:** Docker, Kubernetes, API Gateway, Message Queue"
        },
        {
            "instruction": "Python'da context manager nedir, nasıl yazılır?",
            "input": "",
            "output": "```python\n# Hazır kullanım\nwith open('dosya.txt') as f:\n    data = f.read()\n\n# Sınıf olarak yazma\nclass Database:\n    def __init__(self, url):\n        self.url = url\n    def __enter__(self):\n        self.conn = connect(self.url)\n        return self.conn\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        self.conn.close()\n\n# Decorator ile\nfrom contextlib import contextmanager\n\n@contextmanager\ndef temp_file():\n    f = open('temp.txt', 'w')\n    try:\n        yield f\n    finally:\n        f.close()\n        os.remove('temp.txt')\n```"
        },
        {
            "instruction": "CI/CD nedir, neden önemlidir?",
            "input": "",
            "output": "CI/CD (Continuous Integration / Continuous Delivery), yazılım sürecini otomatikleştirme yöntemidir.\n\n**CI:** Kod değişikliği → otomatik test → merge öncesi kontrol\n**CD:** Otomatik dağıtım → ortam yönetimi → geri dönüşüm\n\n**Araçlar:** GitHub Actions, GitLab CI, Jenkins, CircleCI\n\n**Pipeline:**\n1. Code push → 2. Lint + Test → 3. Build → 4. Staging → 5. Production"
        },
        {
            "instruction": "Python'da decorator nedir, örnek verir misin?",
            "input": "",
            "output": "```python\n# Temel decorator\ndef log(func):\n    def wrapper(*args, **kwargs):\n        print(f'{func.__name__} çağrıldı')\n        result = func(*args, **kwargs)\n        print(f'{func.__name__} tamamlandı')\n        return result\n    return wrapper\n\n@log\ndef topla(a, b):\n    return a + b\n\n# Parametreli decorator\ndef repeat(n):\n    def decorator(func):\n        def wrapper(*args, **kwargs):\n            for _ in range(n):\n                func(*args, **kwargs)\n        return wrapper\n    return decorator\n\n@repeat(3)\ndef selamla():\n    print('Merhaba!')\n```"
        },
        {
            "instruction": "NoSQL ve SQL arasındaki farklar nelerdir?",
            "input": "",
            "output": "**SQL (İlişkeli):** Tablo yapısı, schema zorunlu, ACID uyumlu. Örn: PostgreSQL, MySQL\n\n**NoSQL (İlişkisel olmayan):** Esnek schema, CAP teoremi, yatay ölçeklenebilir.\n- Document: MongoDB\n- Key-Value: Redis\n- Column: Cassandra\n- Graph: Neo4j\n\n**Kullanım:** SQL → finans, e-ticaret. NoSQL → real-time, IoT, social media."
        },
        {
            "instruction": "Python'da async/await nedir, nasıl kullanılır?",
            "input": "",
            "output": "```python\nimport asyncio\nimport aiohttp\n\nasync def fetch(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.text()\n\nasync def main():\n    # Paralel\n    results = await asyncio.gather(\n        fetch('https://api.example.com/1'),\n        fetch('https://api.example.com/2'),\n        fetch('https://api.example.com/3')\n    )\n\nasyncio.run(main())\n```\n\n**Ne zaman:** HTTP istekleri, dosya işlemleri, veritabanı sorguları, I/O-bound işlemler"
        },
        {
            "instruction": "Docker ve Kubernetes arasındaki fark nedir?",
            "input": "",
            "output": "**Docker:** Container teknolojisi. Tek container oluşturma ve çalıştırma.\n\n**Kubernetes:** Container orchestration. Birden fazla container yönetimi.\n\n**İlişki:** Docker container'larını oluşturur, Kubernetes onları yönetir.\n\n**Kubernetes kavramları:**\n- Pod: En küçük birim (1+ container)\n- Service: Network erişimi\n- Deployment: Yönetim stratejisi\n- Namespace: İzolasyon"
        },
        {
            "instruction": "Python'da type hinting nedir, neden önemlidir?",
            "input": "",
            "output": "```python\n# Temel kullanım\ndef topla(a: int, b: int) -> int:\n    return a + b\n\n# Koleksiyonlar\nfrom typing import List, Dict, Optional\n\ndef isle(kullanicilar: List[Dict[str, str]]) -> Optional[str]:\n    if kullanicilar:\n        return kullanicilar[0]['name']\n    return None\n\n# Modern Python (3.10+)\ndef process(data: int | str) -> list[int]:\n    ...\n```\n\n**Avantajlar:** IDE desteği, hata tespiti (mypy), dokümantasyon, okunabilirlik"
        },
        {
            "instruction": "OAuth 2.0 nedir, nasıl çalışır?",
            "input": "",
            "output": "OAuth 2.0, yetkilendirme protokolüdür.\n\n**Akış:**\n1. Kullanıcı uygulamaya girer\n2. Uygulama Identity Provider'a yönlendirir\n3. Kullanıcı oturum açar ve izin verir\n4. Provider authorization code döner\n5. Uygulama code'u access token'a çevirir\n6. Token ile API çağrılır\n\n**Grant Types:** Authorization Code, PKCE, Client Credentials\n\n**Token Türleri:** Access Token (kısa ömürlü), Refresh Token (uzun ömürlü), ID Token (JWT)"
        },
        {
            "instruction": "Python'da GIL nedir, multi-threading'i nasıl etkiler?",
            "input": "",
            "output": "GIL (Global Interpreter Lock), aynı anda sadece bir thread'in bytecode çalıştırmasına izin verir.\n\n**Etkileri:**\n- CPU-bound işlemlerde paralellik sağlamaz\n- I/O-bound işlemlerde sorun yok\n- Multi-processing ile aşılabilir\n\n```python\n# CPU-bound: Thread işe yaramaz\n# I/O-bound: Thread işe yarar\n# Çözüm: multiprocessing\nfrom multiprocessing import Pool\n```"
        },
        {
            "instruction": "WebSocket nedir, HTTP'den farkı nedir?",
            "input": "",
            "output": "WebSocket, çift yönlü gerçek zamanlı iletişim sağlar.\n\n**HTTP vs WebSocket:**\n- HTTP: Request/Response, tek yönlü\n- WebSocket: Çift yönlü, sürekli bağlantı\n\n**Kullanım:** Chat, canlı bildirimler, oyun, finans verileri\n\n```python\nimport websockets, asyncio\n\nasync def handler(websocket):\n    async for message in websocket:\n        await websocket.send(f'Alındı: {message}')\n```"
        },
        {
            "instruction": "Python'da metaclass nedir, ne zaman kullanılır?",
            "input": "",
            "output": "```python\nclass SingletonMeta(type):\n    _instances = {}\n    def __call__(cls, *args, **kwargs):\n        if cls not in cls._instances:\n            cls._instances[cls] = super().__call__(*args, **kwargs)\n        return cls._instances[cls]\n\nclass Database(metaclass=SingletonMeta):\n    pass\n```\n\n**Ne zaman:** Singleton, ORM, API validation, Plugin sistemi"
        },
        {
            "instruction": "RabbitMQ nedir, message queue neden kullanılır?",
            "input": "",
            "output": "Message Queue, sistemler arası asynchronous iletişim sağlar.\n\n**RabbitMQ:** AMQP protokolü, Exchange/Queue/Binding, persistence\n\n**Kullanım:** Mikroservis iletişimi, task queue, event-driven, load leveling\n\n```python\nimport pika\nconnection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))\nchannel = connection.channel()\nchannel.queue_declare(queue='tasks')\nchannel.basic_publish(exchange='', routing_key='tasks', body='Hello')\n```"
        },
        {
            "instruction": "Python'da descriptor protocol nedir?",
            "input": "",
            "output": "```python\nclass Validator:\n    def __init__(self, min_value, max_value):\n        self.min_value = min_value\n        self.max_value = max_value\n    def __set_name__(self, owner, name):\n        self.name = name\n        self.private_name = f'_{name}'\n    def __get__(self, obj, objtype=None):\n        if obj is None: return self\n        return getattr(obj, self.private_name, None)\n    def __set__(self, obj, value):\n        if not self.min_value <= value <= self.max_value:\n            raise ValueError(f'{self.name} {self.min_value}-{self.max_value} arasında olmalı')\n        setattr(obj, self.private_name, value)\n\nclass Person:\n    age = Validator(0, 150)\n    salary = Validator(0, 1000000)\n```"
        },
        {
            "instruction": "CAP teoremi nedir, dağıtık sistemlerde önemi nedir?",
            "input": "",
            "output": "CAP Teoremi: Dağıtık sistemde üç özellikten sadece ikisi sağlanabilir.\n\n- **Consistency:** Tüm düğümler aynı veriyi görür\n- **Availability:** Her istek yanıt alır\n- **Partition Tolerance:** Network kesintisi dayanıklılığı\n\n**Seçenekler:**\n- CP: MongoDB, HBase\n- AP: Cassandra, DynamoDB\n- CA: PostgreSQL (tek sunucu)"
        },
        {
            "instruction": "Python'da dataclass nedir, normal class'tan farkı nedir?",
            "input": "",
            "output": "```python\nfrom dataclasses import dataclass, field\nfrom typing import List\n\n@dataclass\nclass User:\n    name: str\n    age: int\n    email: str = ''\n    tags: List[str] = field(default_factory=list)\n\nuser = User('Ali', 30, 'ali@example.com', ['python', 'ai'])\nprint(user)  # User(name='Ali', age=30, ...)\n\n# Frozen (immutable)\n@dataclass(frozen=True)\nclass Point:\n    x: float\n    y: float\n```"
        },
        {
            "instruction": "gRPC nedir, REST'ten farkı nedir?",
            "input": "",
            "output": "gRPC, Google'ın RPC framework'ü. HTTP/2 + Protocol Buffers kullanır.\n\n**REST vs gRPC:**\n- REST: HTTP/1.1, JSON, text tabanlı\n- gRPC: HTTP/2, binary, daha hızlı\n\n**Avantajları:** Hızlı serialization, streaming, kod oluşturma, multi-dil\n\n```protobuf\nsyntax = \"proto3\";\nservice UserService {\n  rpc GetUser (UserRequest) returns (UserResponse);\n}\n```"
        },
        {
            "instruction": "Python'da memory management nasıl çalışır?",
            "input": "",
            "output": "**Python bellek yönetimi:**\n1. **Reference Counting:** Her nesnenin referansı sayılır, 0 olunca silinir\n2. **Garbage Collection:** Circular reference'leri çözer\n3. **Memory Pool:** Küçük nesneler için Python kendi pool'unu kullanır\n\n**Optimizasyon:**\n- `__slots__`: Attribute dict kazancı\n- Generator: Lazy evaluation\n- WeakRef: Circular reference önleme\n- `del + gc.collect()`: Manuel temizleme"
        },
        {
            "instruction": "Türkiye'nin en güzel şehirleri hangileridir?",
            "input": "",
            "output": "Türkiye'nin öne çıkan şehirleri:\n\n1. **İstanbul** – Tarihi, kültürel ve doğal zenginlikleriyle eşsiz. Boğaz, camiler, saraylar.\n2. **Kapadokya (Nevşehir)** – Peri bacaları, yeraltı şehirleri, balon turları.\n3. **Antalya** – Akdeniz'in incisi, plajlar, antik kentler.\n4. **Bodrum** – Ege'nin maviliği, gece hayatı, tarihi dokusu.\n5. **Trabzon** – Yeşil doğası, Sümela Manastırı, Karadeniz lezzetleri.\n6. **Şanlıurfa** – Tarihi mekanlar, Balıklıgöl, gastronomi.\n7. **Mardin** – Taş mimari, kültürel mozaik.\n8. **Konya** – Mevlana, Selçuklu mirası.\n9. **Safranbolu** – UNESCO mirası Osmanlı evleri.\n10. **Pamukkale** – Doğal travertenler, termal sular."
        },
        {
            "instruction": "Atatürk kimdir ve neden önemlidir?",
            "input": "",
            "output": "Mustafa Kemal Atatürk (1881-1938), Türkiye Cumhuriyeti'nin kurucusu ve ilk Cumhurbaşkanıdır.\n\n**Neden önemlidir:**\n- Kurtuluş Savaşı'nı önderlik ederek bağımsız Türkiye'yi kurdu\n- Çağdaşlaşma ve modernleşme reformları yaptı\n- Kadınlara seçme ve seçme hakkı verdi\n- Laiklik ilkesini getirdi\n- Latin alfabesine geçişi sağladı\n- Sanayi ve eğitim alanında devrimler yaptı\n\nAtatürk, sadece Türkiye değil, dünyada birçok ülkeye ilham veren bir liderdir."
        },
        {
            "instruction": "Derin öğrenme nedir, makine öğrenmesinden farkı nedir?",
            "input": "",
            "output": "Derin öğrenme, makine öğrenmesinin çok katmanlı sinir ağları kullanan alt dalıdır.\n\n**Makine Öğrenmesi vs Derin Öğrenme:**\n- **ML:** Özellik mühendisliği gerekir, daha az veri yeterli\n- **DL:** Özellikleri kendisi öğrenir, büyük veri gerektirir\n\n**Derin Öğrenme Türleri:**\n- CNN: Görüntü işleme\n- RNN/LSTM: Zaman serisi, dil\n- Transformer: NLP, görüntü, çoklu modalite\n- GAN: Veri üretimi\n\n**Kullanım:** GPT, BERT, Stable Diffusion, AlphaGo"
        },
        {
            "instruction": "Atom nedir, yapısını açıkla.",
            "input": "",
            "output": "Atom, bir elementin kimyasal öelliklerini taşıyan en küçük birimidir.\n\n**Yapı:**\n- **Çekirdek:** Proton (+) ve nötron (nötr) içerir\n- **Elektron bulutu:** Elektron (-) çekirdeğin etrafında yörüngelerde döner\n\n**Temel Kavramlar:**\n- Atom numarası (Z) = proton sayısı\n- Kütle numarası (A) = proton + nötron\n- İzotop: Aynı element, farklı nötron sayısı\n- İyon: Elektron kazanan/yapan atom\n\n**Model Gelişimi:** Dalton → Thomson → Rutherford → Bohr → Kuantum"
        },
        {
            "instruction": "DNA nedir, ne işe yarar?",
            "input": "",
            "output": "DNA (Deoksiribonükleik Asit), canlıların genetik bilgisini taşıyan moleküldür.\n\n**Yapı:**\n- Çift sarmal yapı\n- 4 baz: Adenin (A), Timin (T), Guanin (G), Sitozin (C)\n- A-T, G-C eşleşir\n\n**İşlevleri:**\n- Genetik bilgiyi depolar\n- Protein sentezini yönetir\n- Hücre bölünmesinde kopyalanır\n- Nesilden nesile bilgi aktarır\n\n**Teknoloji:** DNA dizileme, CRISPR gen düzenlemesi, DNA bilgisayarlar"
        },
        {
            "instruction": "Güneş sistemi nedir, gezegenleri listele.",
            "input": "",
            "output": "Güneş sistemi, Güneş ve etrafında dönen gök cisimlerinden oluşur.\n\n**Gezegenler (Güneş'e yakınlığa göre):**\n1. **Merkür** – En küçük, en yakın\n2. **Venüs** – En sıcak, ters dönen\n3. **Dünya** – Yaşam var\n4. **Mars** – Kızıl gezegen\n5. **Jüpiter** – En büyük, dev gaz küresi\n6. **Satürn** – Halkalarıyla ünlü\n7. **Uranüs** – Yandan dönen\n8. **Neptün** – En uzak, en soğuk\n\n**Diğer:** Cüce gezegenler (Plüton), asteroit kuşağı, kuyruklu yıldızlar"
        },
        {
            "instruction": "I. Dünya Savaşı ne zaman başladı, nedenleri nelerdir?",
            "input": "",
            "output": "I. Dünya Savaşı 28 Temmuz 1914'te başladı, 11 Kasım 1918'de bitti.\n\n**Nedenleri:**\n- **Milliyetçilik:** Balkanlar'da etnik gerilimler\n- **Emperyalizm:** Sömürge rekabeti\n- **Militarizm:** Silahlanma yarışı\n- **İttifak sistemi:** İki blok oluştu\n- **Barışsızlık:** Avrupa'da gerilim arttı\n\n**Tetikleyici:** Avusturya-Macaristan veliahtı Franz Ferdinand'ın suikaste uğraması (28 Haziran 1914)\n\n**Sonuç:** 17 milyon ölüm, imparatorlukların çöküşü, Versay Antlaşması"
        },
        {
            "instruction": "II. Dünya Savaşı ne zaman başladı, nedenleri nelerdir?",
            "input": "",
            "output": "II. Dünya Savaşı 1 Eylül 1939'da başladı, 2 Eylül 1945'te bitti.\n\n**Nedenleri:**\n- **Versay Antlaşması:** Almanya'yı zor şartlara bağladı\n- **Faşizm:** Hitler, Mussolini, Hirohito\n- **Ekonomik kriz:** 1929 Büyük Buhran\n- **Appeasement:** Batılı ülkelerin yumuşak tutumu\n- **Lebensraum:** Hitler'in yayılma politikası\n\n**Tetikleyici:** Almanya'nın Polonya'yı işgali (1 Eylül 1939)\n\n**Sonuç:** 70-85 milyon ölüm, atom bombası, BM'nin kuruluşu"
        },
        {
            "instruction": "Kuantum bilgisayar nedir, klasik bilgisayardan farkı nedir?",
            "input": "",
            "output": "Kuantum bilgisayar, kuantum mekaniği ilkelerini kullanarak hesaplama yapar.\n\n**Klasik vs Kuantum:**\n- **Klasik:** Bit (0 veya 1)\n- **Kuantum:** Qubit (0 ve 1 aynı anda - süperpozisyon)\n\n**Temel Kavramlar:**\n- **Süperpozisyon:** Qubit aynı anda birden fazla durumda\n- **Dolanıklık:** Qubit'ler birbirine bağlı\n- **Kuantum parallelizm:** Aynı anda birçok hesaplama\n\n**Kullanım:** Kriptografi, ilaç keşfi, optimizasyon, yapay zeka\n\n**Şirketler:** IBM, Google, Microsoft, IonQ"
        },
        {
            "instruction": "Blockchain nedir, nasıl çalışır?",
            "input": "",
            "output": "Blockchain, dağıtık ve değiştirilemez bir veritabanıdır.\n\n**Nasıl çalışır:**\n1. İşlem oluşur\n2. İşlem blokta toplanır\n3. Blok hash ile önceki bloğa bağlanır\n4. Ağdaki düğümler doğrular\n5. Blok zincire eklenir\n\n**Özellikleri:**\n- **Dağıtık:** Tek bir sunucu yok\n- **Değiştirilemez:** Silinemez, değiştirilemez\n- **Şeffaf:** Herkes görebilir\n- **Güvenli:** Kriptografik hash\n\n**Kullanım:** Bitcoin, Ethereum, NFT, akıllı kontratlar"
        },
        {
            "instruction": "Yapay zeka nedir, türleri nelerdir?",
            "input": "",
            "output": "Yapay zeka, makinelerin insan benzeri zeka göstermesini sağlayan teknolojidir.\n\n**Türleri:**\n- **Dar AI (ANI):** Tek görevde uzman (GPT, AlphaGo)\n- **Genel AI (AGI):** İnsan düzeyi (henüz yok)\n- **Süper AI (ASI):** İnsanüstü (teorik)\n\n**Alt Dalları:**\n- **ML:** Veriden öğrenme\n- **DL:** Çok katmanlı sinir ağları\n- **NLP:** Dil işleme\n- **CV:** Görüntü işleme\n- **RL:** Pekiştirmeli öğrenme\n\n**Kullanım:** Chatbot, otonom araç, tıp, finans, eğitim"
        },
        {
            "instruction": "Python'da bir HTTP istemcisi yaz, requests kütüphanesi ile.",
            "input": "",
            "output": "```python\nimport requests\n\n# GET isteği\nresponse = requests.get('https://api.example.com/users')\nprint(response.status_code)\nprint(response.json())\n\n# POST isteği\ndata = {'name': 'Ali', 'age': 30}\nresponse = requests.post('https://api.example.com/users', json=data)\nprint(response.status_code)\n\n# Headers ile\nheaders = {'Authorization': 'Bearer token123'}\nresponse = requests.get('https://api.example.com/protected', headers=headers)\n\n# Hata yönetimi\ntry:\n    response = requests.get('https://api.example.com/data', timeout=10)\n    response.raise_for_status()\nexcept requests.exceptions.RequestException as e:\n    print(f'Hata: {e}')\n```"
        },
        {
            "instruction": "Flask ile basit bir REST API yaz.",
            "input": "",
            "output": "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\nitems = {}\n\n@app.route('/items', methods=['GET'])\ndef get_items():\n    return jsonify(items)\n\n@app.route('/items/<id>', methods=['GET'])\ndef get_item(id):\n    return jsonify(items.get(id, {}))\n\n@app.route('/items', methods=['POST'])\ndef create_item():\n    data = request.json\n    id = str(len(items) + 1)\n    items[id] = data\n    return jsonify({'id': id}), 201\n\n@app.route('/items/<id>', methods=['PUT'])\ndef update_item(id):\n    if id in items:\n        items[id] = request.json\n        return jsonify(items[id])\n    return jsonify({'error': 'Not found'}), 404\n\n@app.route('/items/<id>', methods=['DELETE'])\ndef delete_item(id):\n    items.pop(id, None)\n    return '', 204\n\nif __name__ == '__main__':\n    app.run(debug=True)\n```"
        },
        {
            "instruction": "Python'da bir dosyayı satır satır oku ve kelime sayısını hesapla.",
            "input": "",
            "output": "```python\ndef count_words(filename):\n    with open(filename, 'r', encoding='utf-8') as f:\n        text = f.read()\n    return len(text.split())\n\ndef count_words_line_by_line(filename):\n    total = 0\n    with open(filename, 'r', encoding='utf-8') as f:\n        for line in f:\n            total += len(line.split())\n    return total\n\n# Kullanım\nprint(count_words('test.txt'))\nprint(count_words_line_by_line('test.txt'))\n```"
        },
        {
            "instruction": "JavaScript'te bir Promise.all implementasyonu yaz.",
            "input": "",
            "output": "```javascript\nasync function fetchAll(urls) {\n    const promises = urls.map(url => fetch(url).then(r => r.json()));\n    const results = await Promise.all(promises);\n    return results;\n}\n\n// Kullanım\nfetchAll(['https://api.example.com/1', 'https://api.example.com/2'])\n    .then(data => console.log(data))\n    .catch(err => console.error(err));\n\n// Promise.allSettled (hepsi tamamlanır, hata olsa bile)\nasync function fetchAllSettled(urls) {\n    const promises = urls.map(url =>\n        fetch(url).then(r => r.json())\n    );\n    return await Promise.allSettled(promises);\n}\n```"
        },
        {
            "instruction": "Bash'te bir dosya yedekleme scripti yaz.",
            "input": "",
            "output": "```bash\n#!/bin/bash\n# Dosya yedekleme scripti\n\nSOURCE=\"$1\"\nBACKUP_DIR=\"backups\"\nTIMESTAMP=$(date +%Y%m%d_%H%M%S)\n\nif [ -z \"$SOURCE\" ]; then\n    echo \"Kullanım: $0 <dosya_veya_klasor>\"\n    exit 1\nfi\n\nmkdir -p \"$BACKUP_DIR\"\ncp -r \"$SOURCE\" \"${BACKUP_DIR}/${TIMESTAMP}_$(basename $SOURCE)\"\n\necho \"Yedek oluşturuldu: ${BACKUP_DIR}/${TIMESTAMP}_$(basename $SOURCE)\"\n```"
        },
        {
            "instruction": "SQL injection nedir, nasıl önlerin? Python örneği ver.",
            "input": "",
            "output": "SQL injection, saldırganın SQL sorgularına zararli kod enjekte etmesidir.\n\n**Önleme:**\n1. Parametreli sorgular (prepared statements)\n2. Input validasyonu\n3. ORM kullanma\n4. Minimum prensip ayrıcalığı\n\n**Python örnek:**\n```python\n# KÖTÜ:\ncursor.execute(f'SELECT * FROM users WHERE id = {user_id}')\n\n# İYİ:\ncursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))\n\n# ORM (SQLAlchemy)\nuser = session.query(User).filter(User.id == user_id).first()\n```"
        },
        {
            "instruction": "Docker Compose ile bir web uygulaması nasıl ayağa kaldırılır?",
            "input": "",
            "output": "```yaml\n# docker-compose.yml\nversion: '3.8'\n\nservices:\n  web:\n    build: .\n    ports:\n      - \"5000:5000\"\n    environment:\n      - DATABASE_URL=postgresql://user:pass@db:5432/mydb\n    depends_on:\n      - db\n      - redis\n\n  db:\n    image: postgres:15\n    environment:\n      - POSTGRES_USER=user\n      - POSTGRES_PASSWORD=pass\n      - POSTGRES_DB=mydb\n    volumes:\n      - pgdata:/var/lib/postgresql/data\n\n  redis:\n    image: redis:7-alpine\n    ports:\n      - \"6379:6379\"\n\nvolumes:\n  pgdata:\n```\n\n```bash\n# Çalıştır\ndocker-compose up -d\n\n# Loglar\ndocker-compose logs -f\n\n# Durdur\ndocker-compose down\n```"
        },
        {
            "instruction": "Git flow nedir, nasıl kullanılır?",
            "input": "",
            "output": "Git flow, dallanma ve birleştirme stratejisidir.\n\n**Dallar:**\n- `main`: Production kodu\n- `develop`: Geliştirme kodu\n- `feature/*`: Yeni özellikler\n- `release/*`: Sürüm hazırlığı\n- `hotfix/*`: Acil düzeltmeler\n\n**Kullanım:**\n```bash\n# Feature başlat\ngit flow feature start my-feature\n\n# Feature bitir\ngit flow feature finish my-feature\n\n# Release başlat\ngit flow release start v1.0.0\n\n# Release bitir\ngit flow release finish v1.0.0\n\n# Hotfix\ngit flow hotfix start fix-bug\ngit flow hotfix finish fix-bug\n```"
        },
        {
            "instruction": "Python'da bir singleton pattern implementasyonu yaz.",
            "input": "",
            "output": "```python\nclass Singleton:\n    _instance = None\n    \n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance\n\n# Kullanım\na = Singleton()\nb = Singleton()\nprint(a is b)  # True\n\n# Metaclass ile\nclass SingletonMeta(type):\n    _instances = {}\n    def __call__(cls, *args, **kwargs):\n        if cls not in cls._instances:\n            cls._instances[cls] = super().__call__(*args, **kwargs)\n        return cls._instances[cls]\n\nclass Database(metaclass=SingletonMeta):\n    pass\n```"
        },
        {
            "instruction": "Python'da bir retry decorator yaz: exponential backoff ile.",
            "input": "",
            "output": "```python\nimport time\nimport random\n\ndef retry(max_attempts=3, backoff_factor=2, exceptions=(Exception,)):\n    def decorator(func):\n        def wrapper(*args, **kwargs):\n            for attempt in range(max_attempts):\n                try:\n                    return func(*args, **kwargs)\n                except exceptions as e:\n                    if attempt == max_attempts - 1:\n                        raise\n                    wait = backoff_factor ** attempt + random.uniform(0, 1)\n                    print(f'Deneme {attempt+1} başarısız: {e}. {wait:.1f}s bekleniyor...')\n                    time.sleep(wait)\n        return wrapper\n    return decorator\n\n@retry(max_attempts=3, exceptions=(ConnectionError,))\ndef unreliable_api():\n    import random\n    if random.random() < 0.7:\n        raise ConnectionError('Bağlantı hatası')\n    return 'Başarılı'\n```"
        },
        {
            "instruction": "Python'da binary search implementasyonu yaz (iterative).",
            "input": "",
            "output": "```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n\n# Test\nprint(binary_search([1, 3, 5, 7, 9], 7))  # 3\nprint(binary_search([1, 3, 5, 7, 9], 4))  # -1\n```"
        },
        {
            "instruction": "Python'da merge sort implementasyonu yaz.",
            "input": "",
            "output": "```python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n\nprint(merge_sort([38, 27, 43, 3, 9, 82, 10]))\n```"
        },
        {
            "instruction": "Python'da graph BFS ve DFS implementasyonu yaz.",
            "input": "",
            "output": "```python\nfrom collections import deque, defaultdict\n\ndef bfs(graph, start):\n    visited = set()\n    queue = deque([start])\n    result = []\n    while queue:\n        node = queue.popleft()\n        if node not in visited:\n            visited.add(node)\n            result.append(node)\n            queue.extend(graph[node])\n    return result\n\ndef dfs(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    result = [start]\n    for neighbor in graph[start]:\n        if neighbor not in visited:\n            result.extend(dfs(graph, neighbor, visited))\n    return result\n\ngraph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['E'], 'D': [], 'E': []}\nprint('BFS:', bfs(graph, 'A'))  # ['A', 'B', 'C', 'D', 'E']\nprint('DFS:', dfs(graph, 'A'))  # ['A', 'B', 'D', 'C', 'E']\n```"
        },
        {
            "instruction": "Python'da trie (prefix tree) implementasyonu yaz.",
            "input": "",
            "output": "```python\nclass TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\n\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()\n    \n    def insert(self, word):\n        node = self.root\n        for char in word:\n            if char not in node.children:\n                node.children[char] = TrieNode()\n            node = node.children[char]\n        node.is_end = True\n    \n    def search(self, word):\n        node = self.root\n        for char in word:\n            if char not in node.children:\n                return False\n            node = node.children[char]\n        return node.is_end\n    \n    def starts_with(self, prefix):\n        node = self.root\n        for char in prefix:\n            if char not in node.children:\n                return False\n            node = node.children[char]\n        return True\n\ntrie = Trie()\ntrie.insert('merhaba')\nprint(trie.search('merhaba'))  # True\nprint(trie.starts_with('mer'))  # True\n```"
        },
        {
            "instruction": "Python'da linked list implementasyonu yaz.",
            "input": "",
            "output": "```python\nclass Node:\n    def __init__(self, data):\n        self.data = data\n        self.next = None\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n    \n    def append(self, data):\n        new_node = Node(data)\n        if not self.head:\n            self.head = new_node\n            return\n        current = self.head\n        while current.next:\n            current = current.next\n        current.next = new_node\n    \n    def display(self):\n        elements = []\n        current = self.head\n        while current:\n            elements.append(str(current.data))\n            current = current.next\n        print(' -> '.join(elements))\n    \n    def reverse(self):\n        prev = None\n        current = self.head\n        while current:\n            next_node = current.next\n            current.next = prev\n            prev = current\n            current = next_node\n        self.head = prev\n\nll = LinkedList()\nll.append(1)\nll.append(2)\nll.append(3)\nll.display()  # 1 -> 2 -> 3\nll.reverse()\nll.display()  # 3 -> 2 -> 1\n```"
        },
        {
            "instruction": "Python'da min-heap implementasyonu yaz.",
            "input": "",
            "output": "```python\nclass MinHeap:\n    def __init__(self):\n        self.heap = []\n    \n    def push(self, val):\n        self.heap.append(val)\n        self._bubble_up(len(self.heap) - 1)\n    \n    def pop(self):\n        if not self.heap:\n            return None\n        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]\n        val = self.heap.pop()\n        self._bubble_down(0)\n        return val\n    \n    def _bubble_up(self, i):\n        parent = (i - 1) // 2\n        if i > 0 and self.heap[i] < self.heap[parent]:\n            self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]\n            self._bubble_up(parent)\n    \n    def _bubble_down(self, i):\n        smallest = i\n        left, right = 2*i+1, 2*i+2\n        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:\n            smallest = left\n        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:\n            smallest = right\n        if smallest != i:\n            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]\n            self._bubble_down(smallest)\n\nh = MinHeap()\nfor v in [5, 3, 8, 1, 2]:\n    h.push(v)\nprint(h.pop())  # 1\nprint(h.pop())  # 2\n```"
        },
        {
            "instruction": "Python'da LRU cache implementasyonu yaz.",
            "input": "",
            "output": "```python\nfrom functools import lru_cache\n\n# Hazır kullanım\n@lru_cache(maxsize=128)\ndef fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(50))  # 12586269025\n\n# Manuel implementasyon\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    \n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    \n    def put(self, key, value):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)\n```"
        },
        {
            "instruction": "Python'da rate limiter implementasyonu yaz.",
            "input": "",
            "output": "```python\nimport time\nfrom collections import defaultdict\n\nclass RateLimiter:\n    def __init__(self, max_requests, window_seconds):\n        self.max_requests = max_requests\n        self.window = window_seconds\n        self.requests = defaultdict(list)\n    \n    def is_allowed(self, user_id):\n        now = time.time()\n        self.requests[user_id] = [\n            t for t in self.requests[user_id] if now - t < self.window\n        ]\n        if len(self.requests[user_id]) < self.max_requests:\n            self.requests[user_id].append(now)\n            return True\n        return False\n\nlimiter = RateLimiter(max_requests=5, window_seconds=60)\nprint(limiter.is_allowed('user1'))  # True\n```"
        },
        {
            "instruction": "Python'da event emitter implementasyonu yaz.",
            "input": "",
            "output": "```python\nclass EventEmitter:\n    def __init__(self):\n        self._listeners = {}\n    \n    def on(self, event, callback):\n        if event not in self._listeners:\n            self._listeners[event] = []\n        self._listeners[event].append(callback)\n    \n    def emit(self, event, *args, **kwargs):\n        for callback in self._listeners.get(event, []):\n            callback(*args, **kwargs)\n    \n    def off(self, event, callback):\n        if event in self._listeners:\n            self._listeners[event].remove(callback)\n\nemitter = EventEmitter()\nemitter.on('data', lambda x: print(f'Alındı: {x}'))\nemitter.emit('data', 'Merhaba')\n```"
        },
        {
            "instruction": "Python'da CLI aracı yaz: argparse ile dosya işlemleri.",
            "input": "",
            "output": "```python\nimport argparse\nimport os\nimport shutil\n\ndef main():\n    parser = argparse.ArgumentParser(description='Dosya yönetim aracı')\n    parser.add_argument('action', choices=['copy', 'move', 'delete', 'list'])\n    parser.add_argument('source')\n    parser.add_argument('--dest', help='Hedef dizin')\n    \n    args = parser.parse_args()\n    \n    if args.action == 'copy':\n        shutil.copy2(args.source, args.dest)\n        print(f'Kopyalandı: {args.source} -> {args.dest}')\n    elif args.action == 'move':\n        shutil.move(args.source, args.dest)\n        print(f'Taşındı: {args.source} -> {args.dest}')\n    elif args.action == 'delete':\n        os.remove(args.source)\n        print(f'Silindi: {args.source}')\n    elif args.action == 'list':\n        for f in os.listdir(args.source):\n            print(f)\n\nif __name__ == '__main__':\n    main()\n```"
        },
        {
            "instruction": "Python'da pytest ile bir test yaz: mock kullanarak.",
            "input": "",
            "output": "```python\nimport pytest\nfrom unittest.mock import Mock, patch\n\nclass UserService:\n    def __init__(self, db):\n        self.db = db\n    \n    def get_user(self, user_id):\n        return self.db.query(f'SELECT * FROM users WHERE id={user_id}')\n\ndef test_get_user():\n    mock_db = Mock()\n    mock_db.query.return_value = {'id': 1, 'name': 'Ali'}\n    \n    service = UserService(mock_db)\n    result = service.get_user(1)\n    \n    assert result['name'] == 'Ali'\n    mock_db.query.assert_called_once()\n\n@patch('requests.get')\ndef test_api_call(mock_get):\n    mock_get.return_value.json.return_value = {'status': 'ok'}\n    response = requests.get('https://api.example.com')\n    assert response.json()['status'] == 'ok'\n```"
        },
        {
            "instruction": "Python'da generator ile büyük dosyayı parça parça oku.",
            "input": "",
            "output": "```python\ndef read_chunks(filename, chunk_size=1024):\n    with open(filename, 'r') as f:\n        while True:\n            chunk = f.read(chunk_size)\n            if not chunk:\n                break\n            yield chunk\n\n# Kullanım\nfor chunk in read_chunks('buyuk_dosya.txt', chunk_size=4096):\n    process(chunk)\n\n# Satır satır okuma\ndef read_lines(filename):\n    with open(filename, 'r') as f:\n        for line in f:\n            yield line.strip()\n```"
        },
        {
            "instruction": "Python'da metaclass ile singleton + registry pattern yaz.",
            "input": "",
            "output": "```python\nclass PluginMeta(type):\n    _registry = {}\n    \n    def __new__(mcs, name, bases, attrs):\n        cls = super().__new__(mcs, name, bases, attrs)\n        if name != 'Plugin':\n            mcs._registry[name.lower()] = cls\n        return cls\n\nclass Plugin(metaclass=PluginMeta):\n    pass\n\nclass EmailPlugin(Plugin):\n    pass\n\nclass SMSPlugin(Plugin):\n    pass\n\nprint(PluginMeta._registry)  # {'emailplugin': ..., 'smsplugin': ...}\n```"
        },
    ]
    return samples

# ── 3. Akıl Yürütme Verileri ────────────────────────────────────────────────

def generate_reasoning_data():
    """Akıl yürütme ve problem çözme verileri"""
    samples = [
        {
            "instruction": "Bir işte 12 işçi 15 günde bitiriyor. 18 işçi kaç günde bitirir?",
            "input": "",
            "output": "Bu ters orantı problemidir.\n\n**Çözüm:**\n- 12 işçi × 15 gün = 180 işçi-gün (toplam iş)\n- 180 işçi-gün ÷ 18 işçi = **10 gün**\n\n18 işçi 10 günde bitirir."
        },
        {
            "instruction": "Bir sayının %20'si 60 ise bu sayı kaçtır?",
            "input": "",
            "output": "Sayı = x olsun.\n\nx × 20/100 = 60\nx × 0.2 = 60\nx = 60 / 0.2\n**x = 300**\n\nSayı 300'dür."
        },
        {
            "instruction": "3, 6, 12, 24, ? dizisinde sonraki sayı kaçtır?",
            "input": "",
            "output": "Diziyi inceleyelim:\n- 3 × 2 = 6\n- 6 × 2 = 12\n- 12 × 2 = 24\n- 24 × 2 = **48**\n\nHer sayı öncekinin 2 katıdır. Sonraki sayı **48**'dir."
        },
        {
            "instruction": "Bir dikdörtgenin uzun kenarı 12 cm, kısa kenarı 8 cm. Çevresi ve alanı kaç cm²?",
            "input": "",
            "output": "**Çevre:**\nC = 2 × (uzun + kısa) = 2 × (12 + 8) = 2 × 20 = **40 cm**\n\n**Alan:**\nA = uzun × kısa = 12 × 8 = **96 cm²**"
        },
        {
            "instruction": "Bir araba saatte 60 km hızla 3 saat, sonra saatte 80 km hızla 2 saat yol alırsa toplam kaç km yol almıştır?",
            "input": "",
            "output": "**İlk bölüm:**\n60 km/sa × 3 sa = 180 km\n\n**İkinci bölüm:**\n80 km/sa × 2 sa = 160 km\n\n**Toplam:**\n180 + 160 = **340 km**"
        },
        {
            "instruction": "5 kişilik bir grupta en az 2 kişinin doğum gününün aynı ayda olma olasılığı nedir?",
            "input": "",
            "output": "Bu bir çekmece ilkesi (pigeonhole principle) problemidir.\n\n12 ay var, 5 kişi var.\nEn kötü senaryo: 5 kişi farklı aylarda doğmuş.\nAma 5 < 12 olduğu için garantili değil.\n\n**Doğru yaklaşım:**\nTümünün farklı ayda olma olasılığı:\nP(farklı) = 12/12 × 11/12 × 10/12 × 9/12 × 8/12 ≈ 0.382\n\n**En az 2 kişinin aynı ayda olma olasılığı:**\nP(aynı) = 1 - 0.382 ≈ **0.618 (yaklaşık %62)**"
        },
        {
            "instruction": "Bir zarı 2 kez atıyoruz. İkisinin de 6 gelme olasılığı nedir?",
            "input": "",
            "output": "Bir zarda 6 gelme olasılığı: 1/6\n\nİki zarda da 6 gelme:\nP = 1/6 × 1/6 = **1/36 ≈ %2.78**"
        },
        {
            "instruction": "Bir şirketin kar marjı %25. Maliyeti 80 TL ise satış fiyatı kaç TL?",
            "input": "",
            "output": "Kar marjı = (Satış - Maliyet) / Satış = %25\n\n0.25 = (S - 80) / S\n0.25S = S - 80\n80 = S - 0.25S\n80 = 0.75S\nS = 80 / 0.75\n**S = 106.67 TL**\n\nSatış fiyatı **106.67 TL**'dir."
        },
        {
            "instruction": "Bir sınıfta 30 öğrenci var. %60'ı kız, %40'ı erkek. Kız öğrencilerin yarısı bilgisayar biliyor. Bilgisayar bilen kız öğrenci sayısı kaç?",
            "input": "",
            "output": "Kız öğrenci sayısı: 30 × 0.60 = 18\n\nBilgisayar bilen kız: 18 × 0.50 = **9 öğrenci**"
        },
        {
            "instruction": "Bir sayı 7'ye bölündüğünde kalan 3, 5'e bölündüğünde kalan 2. 100'den küçük en büyük böyle sayı kaç?",
            "input": "",
            "output": "Sayı = x olsun.\n- x ≡ 3 (mod 7) → x = 7k + 3\n- x ≡ 2 (mod 5)\n\n7k + 3 ≡ 2 (mod 5)\n7k ≡ -1 ≡ 4 (mod 5)\n2k ≡ 4 (mod 5)\nk ≡ 2 (mod 5)\nk = 5m + 2\n\nx = 7(5m + 2) + 3 = 35m + 17\n\n100'den küçük en büyük:\nm = 2 → x = 35×2 + 17 = **87**\n\nKontrol: 87/7 = 12 kalan 3 ✓, 87/5 = 17 kalan 2 ✓"
        },
    ]
    return samples

# ── 4. Türkçe Kültür & Genel Kültür ─────────────────────────────────────────

def generate_culture_data():
    """Türk kültürü ve genel kültür verileri"""
    samples = [
        {
            "instruction": "Türkiye'nin başkenti neresidir ve neden önemlidir?",
            "input": "",
            "output": "Türkiye'nin başkenti **Ankara**'dır. 13 Ekim 1923'te Türkiye Cumhuriyeti'nin kuruluşuyla birlikte başkent ilan edilmiştir.\n\n**Neden önemlidir:**\n- Kurtuluş Savaşı'nın merkezi olmuştur\n- Mustafa Kemal Atatürk'ün ikametgahıdır\n- Meclis ve devlet bakanlıkları buradadır\n- Türkiye'nin ikinci en büyük şehridir (~6 milyon nüfus)"
        },
        {
            "instruction": "Atatürk kimdir ve neden önemlidir?",
            "input": "",
            "output": "Mustafa Kemal Atatürk (1881-1938), Türkiye Cumhuriyeti'nin kurucusu ve ilk Cumhurbaşkanıdır.\n\n**Neden önemlidir:**\n- Kurtuluş Savaşı'nı önderlik ederek bağımsız Türkiye'yi kurdu\n- Çağdaşlaşma ve modernleşme reformları yaptı\n- Kadınlara seçme ve seçme hakkı verdi\n- Laiklik ilkesini getirdi\n- Latin alfabesine geçişi sağladı"
        },
        {
            "instruction": "Cumhuriyet ne zaman ilan edildi?",
            "input": "",
            "output": "Türkiye Cumhuriyeti **29 Ekim 1923**'te ilan edilmiştir. Bu tarihten itibaren Türkiye monarşiden cumhuriyete geçmiştir."
        },
        {
            "instruction": "Osmanlı İmparatorluğu ne zaman kuruldu ve ne zaman sona erdi?",
            "input": "",
            "output": "Osmanlı İmparatorluğu:\n- **Kuruluş:** 1299 (Osman Bey)\n- **Yükselme:** 1453 (İstanbul'un fethi)\n- **Doruk:** 1520-1566 (Kanuni Sultan Süleyman)\n- **Çöküş:** 1918 (I. Dünya Savaşı yenilgisi)\n- **Sona eriş:** 1 Kasım 1922 (Saltanatın kaldırılması)\n- **Cumhuriyet:** 29 Ekim 1923"
        },
        {
            "instruction": "Türkiye'nin en büyük 5 şehri nedir?",
            "input": "",
            "output": "Türkiye'nin en büyük şehirleri (nüfusa göre):\n1. **İstanbul** – ~16 milyon\n2. **Ankara** – ~6 milyon\n3. **İzmir** – ~4.5 milyon\n4. **Bursa** – ~3 milyon\n5. **Antalya** – ~2.5 milyon"
        },
        {
            "instruction": "Türk mutfağının önemli lezzetleri nelerdir?",
            "input": "",
            "output": "Türk mutfağının önemli lezzetleri:\n\n**Ana Yemekler:**\n- Kebap çeşitleri (Adana, Urfa, İskender)\n- Köfte, karnıyarık, mantı\n- Lahmacun, pide\n- Etli nohut, kuru fasulye\n\n**Tatlılar:**\n- Baklava, künefe, katmer\n- Lokum, helva, aşure\n- Sütlaç, kazandibi, tavuk göğsü\n\n**İçecekler:**\n- Türk kahvesi, çay\n- Ayran, boza, şalgam"
        },
        {
            "instruction": "Türkiye'nin UNESCO Dünya Mirası listesindeki yerleri nelerdir?",
            "input": "",
            "output": "Türkiye'nin UNESCO Dünya Mirası listesindeki bazı yerleri:\n\n1. **Kapadokya** (Nevşehir) – Peri bacaları, yeraltı şehirleri\n2. **Pamukkale** (Denizli) – Doğal travertenler\n3. **Efes** (İzmir) – Antik kent\n4. **Nemrut Dağı** (Adıyaman) – Dev heykeller\n5. **Safranbolu** (Karabük) – Osmanlı evleri\n6. **Trabzon Sümela Manastırı** – Ortodoks manastır\n7. **İstanbul Tarihi Alanlar** – Ayasofya, Topkapı, Süleymaniye\n8. **Göbeklitepe** (Şanlıurfa) – En eski tapınak"
        },
    ]
    return samples


# ── Ana Fonksiyon ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("KuroNeko TR - Veri Seti Hazırlama")
    print("=" * 60)
    
    all_samples = []
    
    # 1. HF veri setleri
    print("\n[1/4] HuggingFace veri setleri yükleniyor...")
    all_samples.extend(load_turkish_instructions())
    all_samples.extend(load_alpaca_tr())
    all_samples.extend(load_orca_tr())
    all_samples.extend(load_gsm8k_tr())
    all_samples.extend(load_no_robots_tr())
    all_samples.extend(load_lima_tr())
    
    # 2. Sentetik veri
    print("\n[2/4] Sentetik Türkçe veri oluşturuluyor...")
    all_samples.extend(generate_synthetic_turkish())
    
    # 3. Akıl yürütme
    print("\n[3/4] Akıl yürütme verileri oluşturuluyor...")
    all_samples.extend(generate_reasoning_data())
    
    # 4. Kültür
    print("\n[4/4] Kültür verileri oluşturuluyor...")
    all_samples.extend(generate_culture_data())
    
    # Karıştır
    random.shuffle(all_samples)
    
    # Train/Val split
    val_size = int(len(all_samples) * 0.05)
    val_data = all_samples[:val_size]
    train_data = all_samples[val_size:]
    
    # Kaydet
    with open(OUTPUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(OUTPUT_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # ChatML formatında da kaydet
    with open(OUTPUT_DIR / "train_chatml.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            if "instruction" in item:
                chat_item = format_chatml(item["instruction"], item.get("input", ""), item["output"])
                f.write(json.dumps(chat_item, ensure_ascii=False) + "\n")
    
    print(f"\n{'=' * 60}")
    print(f"Toplam: {len(all_samples)} örnek")
    print(f"Train: {len(train_data)} örnek")
    print(f"Val: {len(val_data)} örnek")
    print(f"Kaydedildi: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()

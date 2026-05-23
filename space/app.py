"""
KuroNeko TR - HuggingFace Space App
Phi-4 Mini Türkçe Asistan - Gradio Arayüzü

Donanım: CPU Basic (2 vCPU, 16GB RAM) - Ücretsiz
Model: 4-bit quantized Phi-4-mini-instruct (QLoRA merged)
"""

import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Konfigürasyon ────────────────────────────────────────────────────────────

MODEL_ID = os.environ.get("MODEL_ID", "KuroNeko1234t/phi4-mini-tr")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

SYSTEM_PROMPT = """Sen Türkçe konuşan zeki ve yardımcı bir yapay zeka asistanısın.
Türkçe sorulara her zaman akıcı, doğal ve bilgilendirici Türkçe cevaplar verirsin.
Mantıklı düşünür, gerekirse adım adım açıklarsın.
Kod yazarken açık ve temiz kod üretirsin.
Akıl yürütme gerektiren sorularda düşünme sürecini gösterirsin."""

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# ── Model Yükleme ────────────────────────────────────────────────────────────

print(f"Model yükleniyor: {MODEL_ID}")

# 4-bit quantization config (CPU için)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN or None,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN or None,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("✅ Model yüklendi!")

# ── Chat Fonksiyonu ──────────────────────────────────────────────────────────

def chat(message, history, system_prompt, temperature, max_tokens):
    """Chat fonksiyonu"""
    
    # Mesaj geçmişini oluştur
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Son 5 turn'ü al (bağlam penceresi için)
    for user_msg, assistant_msg in history[-5:]:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    
    # Tokenize
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    return response.strip()

# ── Gradio Arayüzü ───────────────────────────────────────────────────────────

with gr.Blocks(
    title="KuroNeko TR - Phi-4 Mini Türkçe Asistan",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container { max-width: 900px !important; margin: auto !important; }
    .chatbot { min-height: 400px; }
    """,
) as demo:
    
    gr.Markdown("""
    # 🔮 KuroNeko TR
    **Phi-4 Mini Türkçe Asistan**
    
    Türkçe sorulara yanıt veren, kod yazabilen, akıl yürütebilen yapay zeka asistanı.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Sohbet",
                height=400,
                show_label=False,
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Mesajınızı yazın...",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Gönder", variant="primary", scale=1)
            
            clear_btn = gr.Button("Sohbeti Temizle", variant="secondary")
        
        with gr.Column(scale=1):
            system_input = gr.Textbox(
                value=SYSTEM_PROMPT,
                label="System Prompt",
                lines=4,
            )
            
            temp_slider = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=TEMPERATURE,
                step=0.1,
                label="Temperature",
            )
            
            tokens_slider = gr.Slider(
                minimum=64,
                maximum=1024,
                value=MAX_NEW_TOKENS,
                step=64,
                label="Max Tokens",
            )
    
    # Örnek sorular
    gr.Examples(
        examples=[
            "Merhaba! Kendini tanıtır mısın?",
            "Türkiye'nin başkenti neresidir?",
            "Python'da bir liste nasıl sıralanır? Örnek ver.",
            "Fotosentez nedir? Basit bir dille açıkla.",
            "Bir markette elma 5 TL/kg, armut 7 TL/kg. 3 kg elma ve 2 kg armut alsam toplam kaç TL öderim?",
            "Flask ile basit bir REST API yaz.",
            "Docker container'ini arka planda çalıştırmak için komut nedir?",
            "Git'te bir branch'i silmek için komut nedir?",
        ],
        inputs=msg,
    )
    
    # Event handlers
    def respond(message, history, system, temp, tokens):
        if not message.strip():
            return "", history
        
        response = chat(message, history, system, temp, tokens)
        history.append((message, response))
        return "", history
    
    msg.submit(respond, [msg, chatbot, system_input, temp_slider, tokens_slider], [msg, chatbot])
    send_btn.click(respond, [msg, chatbot, system_input, temp_slider, tokens_slider], [msg, chatbot])
    clear_btn.click(lambda: None, None, chatbot)

# ── Başlat ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )

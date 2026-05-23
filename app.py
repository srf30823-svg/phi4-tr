"""
HuggingFace Space - Phi-4 Türkçe Asistan
==========================================
Bu dosyayı HuggingFace Space'e yükle.
Space ayarları: SDK = Gradio, Hardware = CPU Basic (ücretsiz) veya T4 Small

requirements.txt dosyasıyla birlikte kullan.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re

# ── Konfigürasyon ──────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("MODEL_ID", "KULLANICI_ADIN/phi4-mini-turkish")  # <-- DEĞİŞTİR
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

SYSTEM_PROMPT = """Sen Türkçe konuşan zeki ve yardımcı bir yapay zeka asistanısın.
Türkçe sorulara her zaman akıcı, doğal ve bilgilendirici Türkçe cevaplar verirsin.
Mantıklı düşünür, gerekirse adım adım açıklarsın. Kibarlık ve saygı temel değerlerindir."""

# ── Model yükleme ──────────────────────────────────────────────────────────────
print(f"Model yükleniyor: {MODEL_ID}")

# CPU'da çalışmak için (Space ücretsiz tier)
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
DTYPE = torch.float16 if USE_GPU else torch.float32

print(f"Cihaz: {DEVICE}, Dtype: {DTYPE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" if USE_GPU else None,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

if not USE_GPU:
    model = model.to(DEVICE)

model.eval()
print("✅ Model hazır!")


# ── Çıkarım fonksiyonu ─────────────────────────────────────────────────────────
def build_prompt(history: list, user_message: str) -> str:
    """Sohbet geçmişinden ChatML promptu oluştur"""
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    
    # Geçmiş turları ekle (son 5 tur)
    recent_history = history[-5:] if len(history) > 5 else history
    for user_msg, assistant_msg in recent_history:
        prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
    
    # Mevcut soru
    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    return prompt


def generate_response(
    message: str,
    history: list,
    system_override: str = "",
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_NEW_TOKENS,
):
    """Model yanıtı üret"""
    
    if not message.strip():
        return ""
    
    # System prompt override
    global SYSTEM_PROMPT
    active_system = system_override if system_override.strip() else SYSTEM_PROMPT
    
    prompt = f"<|im_start|>system\n{active_system}<|im_end|>\n"
    recent = history[-4:] if len(history) > 4 else history
    for h in recent:
        prompt += f"<|im_start|>user\n{h['content'] if isinstance(h, dict) else h[0]}<|im_end|>\n"
        if isinstance(h, dict) and h.get('role') == 'assistant':
            prompt += f"<|im_start|>assistant\n{h['content']}<|im_end|>\n"
        elif isinstance(h, (list, tuple)) and len(h) > 1:
            prompt += f"<|im_start|>assistant\n{h[1]}<|im_end|>\n"
    
    prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_length = inputs["input_ids"].shape[1]
    
    # Çok uzun promptları kırp
    if input_length > 2048:
        inputs["input_ids"] = inputs["input_ids"][:, -2048:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -2048:]
    
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tokenizer.eos_token_id]
    if im_end_id and im_end_id != tokenizer.eos_token_id:
        eos_ids.append(im_end_id)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(0.01, temperature),
            do_sample=temperature > 0.01,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Temizle
    response = response.replace("<|im_end|>", "").strip()
    
    return response


# ── Gradio Arayüzü ─────────────────────────────────────────────────────────────
EXAMPLES = [
    ["Türkiye'nin en güzel şehirleri hangileridir?"],
    ["Python'da bir liste nasıl sıralanır? Örnek ver."],
    ["Sağlıklı bir sabah rutini nasıl oluşturabilirim?"],
    ["Osmanlı İmparatorluğu ne zaman kuruldu ve yıkıldı?"],
    ["Bir üçgenin iç açıları toplamı kaç derecedir? Açıkla."],
    ["Depresyon ile üzüntü arasındaki fark nedir?"],
]

CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    font-family: 'Segoe UI', sans-serif;
}
.chat-message {
    padding: 12px;
    border-radius: 8px;
    margin: 4px 0;
}
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Phi-4 Türkçe Asistan", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🇹🇷 Phi-4 Türkçe Asistan
    **Microsoft Phi-4-mini** temel alınarak Türkçe için özel olarak eğitilmiş model.
    Türkçe sorularınızı sorabilir, yardım isteyebilirsiniz.
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Sohbet",
                height=500,
                show_copy_button=True,
                avatar_images=("👤", "🤖"),
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Mesajınızı buraya yazın...",
                    label="",
                    scale=5,
                    container=False,
                    lines=1,
                )
                send_btn = gr.Button("Gönder", variant="primary", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Ayarlar")
            
            system_input = gr.Textbox(
                value=SYSTEM_PROMPT,
                label="System Prompt",
                lines=4,
                info="Asistanın kişiliğini özelleştir"
            )
            
            temp_slider = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.7, step=0.05,
                label="Sıcaklık",
                info="Düşük=tutarlı, Yüksek=yaratıcı"
            )
            
            max_tokens_slider = gr.Slider(
                minimum=64, maximum=1024, value=512, step=64,
                label="Maks. Token"
            )
            
            clear_btn = gr.Button("🗑️ Sohbeti Temizle", variant="secondary")
    
    # Örnek sorular
    gr.Examples(
        examples=EXAMPLES,
        inputs=msg_input,
        label="💡 Örnek Sorular",
    )
    
    gr.Markdown("""
    ---
    **Model:** Phi-4-mini-instruct → Türkçe Fine-Tuning (QLoRA + DPO)  
    **Bağlam:** 128K token | **Dil:** Türkçe öncelikli
    """)
    
    # Event handlers
    def user_submit(message, history, system, temp, max_tok):
        if not message.strip():
            return history, ""
        history = history or []
        response = generate_response(message, history, system, temp, max_tok)
        history.append([message, response])
        return history, ""
    
    def clear_chat():
        return [], ""
    
    send_btn.click(
        user_submit,
        inputs=[msg_input, chatbot, system_input, temp_slider, max_tokens_slider],
        outputs=[chatbot, msg_input],
    )
    
    msg_input.submit(
        user_submit,
        inputs=[msg_input, chatbot, system_input, temp_slider, max_tokens_slider],
        outputs=[chatbot, msg_input],
    )
    
    clear_btn.click(clear_chat, outputs=[chatbot, msg_input])

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )

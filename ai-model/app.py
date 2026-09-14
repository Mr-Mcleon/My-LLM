import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template_string
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from main import (
    MODEL_CONFIG,
    SmallLLM,
    TiktokenWrapper,
    add_lora_adapters,
    load_lora_state_dict,
)

app = Flask(__name__)

MODEL_DIR = os.path.dirname(__file__)
FULL_CHECKPOINT_PATH = os.environ.get(
    "LLM_CHECKPOINT", os.path.join(MODEL_DIR, "small_llm.pth")
)
ADAPTER_CHECKPOINT_PATH = os.environ.get(
    "LLM_ADAPTER_CHECKPOINT", os.path.join(MODEL_DIR, "small_llm_lora.pth")
)
BASE_CHECKPOINT_PATH = os.environ.get(
    "LLM_BASE_CHECKPOINT",
    os.path.join(MODEL_DIR, "small_llm_pretrained_500m.pth"),
)
model = None
tokenizer = None
device = None

def load_model():
    global model, tokenizer, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = TiktokenWrapper.load(os.path.join(MODEL_DIR, "tokenizer.pkl"))

    adapter_checkpoint = None
    if os.path.exists(ADAPTER_CHECKPOINT_PATH):
        adapter_checkpoint = torch.load(ADAPTER_CHECKPOINT_PATH, map_location="cpu")
        if not os.path.exists(BASE_CHECKPOINT_PATH):
            raise FileNotFoundError(
                "Found a LoRA adapter but not its base model. "
                f"Set LLM_BASE_CHECKPOINT or add {os.path.basename(BASE_CHECKPOINT_PATH)}."
            )
        checkpoint = torch.load(BASE_CHECKPOINT_PATH, map_location="cpu")
        print(
            f"Loading LoRA adapter {ADAPTER_CHECKPOINT_PATH} "
            f"on base model {BASE_CHECKPOINT_PATH}"
        )
    else:
        checkpoint = torch.load(FULL_CHECKPOINT_PATH, map_location="cpu")

    model_config = {
        "n_embd": checkpoint.get("n_embd", MODEL_CONFIG["n_embd"]),
        "n_head": checkpoint.get("n_head", MODEL_CONFIG["n_head"]),
        "n_layer": checkpoint.get("n_layer", MODEL_CONFIG["n_layer"]),
        "block_size": checkpoint.get("block_size", MODEL_CONFIG["block_size"]),
    }
    model = SmallLLM(vocab_size=checkpoint["vocab_size"], **model_config)
    model.load_state_dict(checkpoint["model_state"])
    if adapter_checkpoint:
        lora_config = adapter_checkpoint.get("lora_config")
        if not lora_config:
            raise ValueError("LoRA adapter checkpoint is missing its lora_config.")
        add_lora_adapters(model, **lora_config)
        load_lora_state_dict(model, adapter_checkpoint["adapter_state"])
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, interactive-widget=resizes-content">
<title>My LLM Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f0f0f;
    color: #e8e8e8;
    height: 100dvh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    padding: 16px 20px;
    border-bottom: 1px solid #222;
    background: #141414;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  header .dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 8px #4ade80;
  }
  header h1 { font-size: 16px; font-weight: 600; color: #fff; }
  header span { font-size: 12px; color: #555; margin-left: auto; }
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .message {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 14px;
    line-height: 1.6;
    word-wrap: break-word;
    animation: fadeIn 0.2s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  .user {
    align-self: flex-end;
    background: #2563eb;
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .bot {
    align-self: flex-start;
    background: #1e1e1e;
    color: #e8e8e8;
    border: 1px solid #2a2a2a;
    border-bottom-left-radius: 4px;
  }
  .thinking {
    align-self: flex-start;
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-bottom-left-radius: 4px;
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 14px;
    color: #555;
    font-style: italic;
  }
  .thinking span { animation: blink 1.2s infinite; }
  .thinking span:nth-child(2) { animation-delay: 0.2s; }
  .thinking span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%, 80%, 100% { opacity: 0.2; } 40% { opacity: 1; } }
  #form {
    display: flex;
    gap: 10px;
    padding: 16px 20px;
    border-top: 1px solid #222;
    background: #141414;
  }
  #input {
    flex: 1;
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 12px 16px;
    color: #e8e8e8;
    font-size: 16px;
    outline: none;
    transition: border-color 0.2s;
    touch-action: manipulation;
    -webkit-user-select: text;
    user-select: text;
  }
  #input:focus { border-color: #2563eb; }
  #input::placeholder { color: #444; }
  #send {
    background: #2563eb;
    color: #fff;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }
  #send:hover { background: #1d4ed8; }
  #send:disabled { background: #1e3a6e; cursor: not-allowed; }
  #open-btn {
    display: block;
    text-align: center;
    font-size: 11px;
    color: #444;
    padding: 4px 0 0;
    text-decoration: none;
  }
  #open-btn:hover { color: #666; }
  .empty {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    color: #333;
  }
  .empty h2 { font-size: 20px; color: #444; }
  .empty p { font-size: 13px; }
</style>
</head>
<body>
<header>
  <div class="dot"></div>
  <h1>My LLM</h1>
  <span>trained on your data</span>
</header>
<div id="chat">
  <div class="empty" id="empty">
    <h2>Start a conversation</h2>
    <p>Type something below to chat with your model</p>
  </div>
</div>
<form id="form">
  <input id="input" type="text" placeholder="Type a message..." autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false" inputmode="text" enterkeyhint="send" />
  <button id="send" type="submit" touch-action="manipulation">Send</button>
</form>
<a id="open-btn" href="" target="_blank">↗ Open in browser for best experience</a>
<script>
  const chat = document.getElementById('chat');
  const form = document.getElementById('form');
  const input = document.getElementById('input');
  const send = document.getElementById('send');
  const empty = document.getElementById('empty');
  const openBtn = document.getElementById('open-btn');

  openBtn.href = window.location.href;

  input.addEventListener('touchstart', () => {
    input.focus();
  });

  function addMessage(text, role) {
    if (empty) empty.style.display = 'none';
    const div = document.createElement('div');
    div.className = 'message ' + role;
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  function addThinking() {
    if (empty) empty.style.display = 'none';
    const div = document.createElement('div');
    div.className = 'thinking';
    div.innerHTML = '<span>●</span><span>●</span><span>●</span>';
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    addMessage(text, 'user');
    input.value = '';
    send.disabled = true;

    const thinking = addThinking();

    try {
      const res = await fetch('/chat/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text })
      });
      const data = await res.json();
      thinking.remove();
      addMessage(data.response || data.error, 'bot');
    } catch (err) {
      thinking.remove();
      addMessage('Something went wrong. Try again.', 'bot');
    }

    send.disabled = false;
    input.focus();
  });
</script>
</body>
</html>"""

@app.route("/chat")
def index():
    return render_template_string(HTML)

@app.route("/chat/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    temperature = float(data.get("temperature", 0.85))
    max_new_tokens = int(data.get("max_new_tokens", 150))

    try:
        prompt_ids = tokenizer.encode("<|bos|>\nUser: " + prompt + "\nAssistant:")
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_len = len(prompt_ids)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                x_cond = x[:, -model.block_size:]
                logits = model(x_cond)[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_id], dim=1)
                if next_id.item() == tokenizer.special_tokens.get("<|eos|>", -1):
                    break

        # Only decode the newly generated tokens, skip special tokens
        generated_ids = x[0].tolist()[prompt_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not response:
            response = tokenizer.decode(x[0].tolist(), skip_special_tokens=True).strip()
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)

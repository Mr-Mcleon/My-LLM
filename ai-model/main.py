import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math
import pickle
from typing import List
import time
import os

# ~505M parameters with the GPT-2 tokenizer vocabulary.
MODEL_CONFIG = {
    "n_embd": 1024,
    "n_head": 16,
    "n_layer": 36,
    "block_size": 64,
}

# ------------------- Tokenizer (tiktoken-based) -------------------
import tiktoken

class TiktokenWrapper:
    """Wrapper around tiktoken GPT-2 tokenizer with our special tokens."""

    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        # Special tokens appended after tiktoken vocab
        self.special_tokens = {
            "<|pad|>": self.enc.n_vocab,
            "<|unk|>": self.enc.n_vocab + 1,
            "<|bos|>": self.enc.n_vocab + 2,
            "<|eos|>": self.enc.n_vocab + 3,
        }
        self.vocab_size = self.enc.n_vocab + len(self.special_tokens)
        # Simple decode mapping for special tokens
        self._special_decode = {v: k for k, v in self.special_tokens.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text; special tokens in the text are preserved."""
        if not add_special_tokens or not self.special_tokens:
            return self.enc.encode(text, allowed_special="all")

        # Split text on special tokens
        import re
        special_pattern = "(" + "|".join(
            re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)
        ) + ")"
        parts = re.split(special_pattern, text)

        ids = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.enc.encode(part, allowed_special="all"))
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text, preserving order of special and regular tokens."""
        parts = []
        regular_run = []
        for i in ids:
            if i in self._special_decode:
                # flush any pending regular tokens first (preserves order)
                if regular_run:
                    parts.append(self.enc.decode(regular_run))
                    regular_run = []
                if not skip_special_tokens:
                    parts.append(self._special_decode[i])
            elif i < self.enc.n_vocab:
                regular_run.append(i)
        # flush remaining regular tokens
        if regular_run:
            parts.append(self.enc.decode(regular_run))
        return "".join(parts)

    @property
    def vocab(self):
        # Compatibility property
        return {i: i for i in range(self.vocab_size)}

    def save(self, path="tokenizer.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"type": "tiktoken", "special_tokens": self.special_tokens}, f)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path="tokenizer.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("type") == "tiktoken":
            return cls()
        # Legacy fallback: old BPETokenizer pickle
        tok = cls()
        return tok


# ------------------- Model -------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

class SmallLLM(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, n_embd),
            'wpe': nn.Embedding(block_size, n_embd),
            'drop': nn.Dropout(dropout),
            'h': nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]),
            'ln_f': nn.LayerNorm(n_embd),
        })
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Sequence length {t} exceeds block size {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


# ------------------- LoRA adapters -------------------
class LoRALinear(nn.Module):
    """A frozen Linear layer with a trainable low-rank update."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be greater than zero")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.base = linear
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(
            rank,
            self.in_features,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        ))
        self.lora_B = nn.Parameter(torch.zeros(
            self.out_features,
            rank,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        ))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)

    def forward(self, x):
        base_output = self.base(x)
        adapter_output = F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),
            self.lora_B,
        )
        return base_output + adapter_output * self.scaling


def _replace_lora_targets(module, rank, alpha, dropout, target_modules):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_modules:
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
        else:
            _replace_lora_targets(child, rank, alpha, dropout, target_modules)


def add_lora_adapters(
    model,
    rank=16,
    alpha=32.0,
    dropout=0.05,
    target_modules=("c_attn", "c_proj", "c_fc"),
):
    """Freeze the base model and attach LoRA to transformer Linear layers."""
    if rank <= 0:
        raise ValueError("LoRA rank must be greater than zero")
    if alpha <= 0:
        raise ValueError("LoRA alpha must be greater than zero")
    if not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in the range [0, 1)")
    for parameter in model.parameters():
        parameter.requires_grad = False
    _replace_lora_targets(model, rank, float(alpha), dropout, set(target_modules))
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError(f"No LoRA target modules found. Targets: {target_modules}")
    return model


def lora_state_dict(model):
    """Return only adapter tensors, keeping checkpoints small and portable."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("lora_A" in name or "lora_B" in name)
    }


def load_lora_state_dict(model, adapter_state):
    parameters = dict(model.named_parameters())
    missing = []
    loaded = 0
    for name, tensor in adapter_state.items():
        parameter = parameters.get(name)
        if parameter is None:
            missing.append(name)
            continue
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
        loaded += 1
    if missing:
        raise ValueError(f"LoRA checkpoint contains unknown adapter tensors: {missing[:3]}")
    if loaded == 0:
        raise ValueError("LoRA checkpoint did not contain any adapter tensors")


# ------------------- Data Loaders -------------------
def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_jsonl_file(path: str) -> str:
    import json
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = conv.get("messages", [])
            if not messages:
                continue
            parts = ["<|bos|>"]
            for msg in messages:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "").strip()
                if content:
                    parts.append(f"{role}: {content}")
            parts.append("<|eos|>")
            lines.append("\n".join(parts))
    return "\n\n".join(lines)

def load_training_data(data_path: str) -> str:
    if data_path.endswith(".jsonl"):
        return load_jsonl_file(data_path)
    return load_text_file(data_path)


def load_chat_samples(path: str, tokenizer: TiktokenWrapper, block_size: int):
    """Build fixed-length prompt/answer samples with loss only on answers."""
    import json
    import random

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                conversation = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = conversation.get("messages", [])
            for assistant_index, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue
                answer = str(message.get("content", "")).strip()
                if not answer:
                    continue

                context_parts = ["<|bos|>"]
                for prior in messages[:assistant_index]:
                    content = str(prior.get("content", "")).strip()
                    if content:
                        role = str(prior.get("role", "user")).capitalize()
                        context_parts.append(f"{role}: {content}")
                context_parts.append("Assistant:")
                context_ids = tokenizer.encode("\n".join(context_parts))
                answer_ids = tokenizer.encode(" " + answer) + [
                    tokenizer.special_tokens["<|eos|>"]
                ]

                # Keep the prompt visible while reserving half the context for
                # the answer. Long examples are cropped at the left.
                answer_limit = max(8, block_size // 2)
                answer_ids = answer_ids[:answer_limit]
                context_limit = block_size - len(answer_ids)
                context_ids = context_ids[-context_limit:]
                ids = context_ids + answer_ids
                if len(ids) < 2:
                    continue

                target_ids = ids[1:] + [tokenizer.special_tokens["<|pad|>"]]
                target_mask = [0] * max(0, len(context_ids) - 1)
                target_mask.extend([1] * len(answer_ids))
                target_mask = target_mask[:block_size]

                pad_id = tokenizer.special_tokens["<|pad|>"]
                ids = ids[:block_size] + [pad_id] * (block_size - len(ids))
                target_ids = target_ids[:block_size] + [
                    pad_id
                ] * (block_size - len(target_ids))
                target_mask += [0] * (block_size - len(target_mask))
                samples.append((ids, target_ids, target_mask))

    random.seed(42)
    random.shuffle(samples)
    return samples


# ------------------- Training -------------------
def atomic_torch_save(payload, path: str):
    """Write checkpoints atomically so an interruption cannot corrupt latest."""
    temp_path = f"{path}.tmp"
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def make_checkpoint(
    model,
    optimizer,
    tokenizer,
    model_config,
    epoch,
    step,
    global_step,
    best_loss,
    scaler=None,
    adapter_only=False,
    lora_config=None,
    base_checkpoint_path=None,
):
    checkpoint = {
        "optimizer_state": optimizer.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        **model_config,
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "best_loss": best_loss,
        "torch_rng_state": torch.get_rng_state(),
    }
    if adapter_only:
        checkpoint.update({
            "checkpoint_type": "lora",
            "adapter_state": lora_state_dict(model),
            "lora_config": lora_config or {},
            "base_checkpoint_path": base_checkpoint_path,
        })
    else:
        checkpoint["model_state"] = model.state_dict()
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return checkpoint


def train_llm(
    data_path: str = "training_data.txt",
    epochs=1,
    pretrained_path: str = None,
    lr: float = None,
    steps: int = None,
    checkpoint_prefix: str = None,
    checkpoint_every: int = 100,
    resume: bool = True,
    output_path: str = "small_llm.pth",
    checkpoint_dir: str = ".",
    tokenizer_path: str = None,
    device_name: str = None,
    dtype_name: str = "auto",
    lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
):
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    def checkpoint_path(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(checkpoint_dir, path)

    data_path = os.path.abspath(data_path)
    pretrained_path = checkpoint_path(pretrained_path) if pretrained_path else None
    output_path = checkpoint_path(output_path)
    tokenizer_path = checkpoint_path(tokenizer_path or "tokenizer.pkl")

    text = load_training_data(data_path)
    print(f"Loaded {len(text):,} characters")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    is_finetune = data_path.endswith(".jsonl")
    if checkpoint_prefix is None:
        if lora:
            checkpoint_prefix = "lora_500m"
        else:
            checkpoint_prefix = "finetune_500m" if is_finetune else "pretrain_500m"
    latest_path = checkpoint_path(f"{checkpoint_prefix}_latest.pth")
    best_path = checkpoint_path(f"{checkpoint_prefix}_best.pth")

    tokenizer = (
        TiktokenWrapper.load(tokenizer_path)
        if os.path.exists(tokenizer_path)
        else TiktokenWrapper()
    )
    print(f"Tokenizer ready: {tokenizer.vocab_size:,} tokens (tiktoken GPT-2 + specials)")

    base_checkpoint = None
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}...")
        base_checkpoint = torch.load(pretrained_path, map_location="cpu")

    resume_checkpoint = None
    if resume and os.path.exists(latest_path):
        print(f"Resuming from {latest_path}...")
        resume_checkpoint = torch.load(latest_path, map_location="cpu")

    if lora and base_checkpoint is None:
        raise FileNotFoundError(
            "LoRA conversation tuning requires a pretrained checkpoint. "
            "Pass --pretrained or place pretrain_500m_latest.pth in the checkpoint directory."
        )
    source_checkpoint = base_checkpoint if lora else (resume_checkpoint or base_checkpoint)
    if source_checkpoint:
        model_config = {
            "n_embd": source_checkpoint.get("n_embd", MODEL_CONFIG["n_embd"]),
            "n_head": source_checkpoint.get("n_head", MODEL_CONFIG["n_head"]),
            "n_layer": source_checkpoint.get("n_layer", MODEL_CONFIG["n_layer"]),
            "block_size": source_checkpoint.get("block_size", MODEL_CONFIG["block_size"]),
        }
        vocab_size = source_checkpoint["vocab_size"]
    else:
        model_config = MODEL_CONFIG.copy()
        vocab_size = tokenizer.vocab_size

    model = SmallLLM(vocab_size=vocab_size, **model_config)
    if source_checkpoint:
        model.load_state_dict(source_checkpoint["model_state"])
    model.to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    if model_config["n_embd"] >= 768:
        if dtype_name == "auto":
            if device.type == "cuda":
                dtype_name = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
            else:
                dtype_name = "bf16"
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported dtype '{dtype_name}'. Choose auto, fp32, fp16, or bf16.")
        model = model.to(dtype=dtype_map[dtype_name])
        print(f"Using {dtype_name} weights for the large-model memory budget")
    else:
        dtype_name = "fp32"

    lora_config = None
    if lora:
        lora_config = {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": ["c_attn", "c_proj", "c_fc"],
        }
        if resume_checkpoint and resume_checkpoint.get("lora_config"):
            saved_lora_config = resume_checkpoint["lora_config"]
            for key in ("rank", "alpha", "dropout", "target_modules"):
                if saved_lora_config.get(key) != lora_config[key]:
                    raise ValueError(
                        f"LoRA configuration changed since the latest checkpoint "
                        f"({key}: {saved_lora_config.get(key)!r} != {lora_config[key]!r}). "
                        "Use the same adapter settings when resuming."
                    )
        add_lora_adapters(model, **lora_config)
        if resume_checkpoint:
            load_lora_state_dict(model, resume_checkpoint["adapter_state"])
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"LoRA config: rank={lora_rank}, alpha={lora_alpha}, "
            f"dropout={lora_dropout}, trainable={trainable:,}/{total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    tokenizer.save(tokenizer_path)

    if lr is None:
        lr = 1e-4 if is_finetune else 6e-4
    if steps is None:
        steps = 2000 if is_finetune else 5000
    total_steps = steps * epochs

    print(f"Training config: lr={lr}, steps={total_steps}, fine-tune={is_finetune}")
    if is_finetune:
        chat_samples = load_chat_samples(data_path, tokenizer, model.block_size)
        if not chat_samples:
            raise RuntimeError("No usable chat samples found")
        print(f"Chat samples: {len(chat_samples):,}")
    else:
        print("Encoding data for training...")
        data = torch.tensor(tokenizer.encode(text, add_special_tokens=True), dtype=torch.long)
        print(f"Total tokens: {len(data):,}")

    # AdamW keeps two extra full-size tensors per parameter and is too
    # memory-heavy for the 500M configuration on this CPU environment.
    if lora:
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=lr,
            weight_decay=0.01,
        )
        batch_size = 1
        print("Using AdamW on trainable LoRA parameters with batch size 1")
    elif model_config["n_embd"] >= 768:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        batch_size = 1
        print("Using memory-light SGD with batch size 1 for the 500M model")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        batch_size = 4

    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and dtype_name == "fp16"
    )
    completed_steps = int(resume_checkpoint.get("global_step", 0)) if resume_checkpoint else 0
    best_loss = float(resume_checkpoint.get("best_loss", float("inf"))) if resume_checkpoint else float("inf")
    if resume_checkpoint and resume_checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
    if resume_checkpoint and resume_checkpoint.get("scaler_state"):
        scaler.load_state_dict(resume_checkpoint["scaler_state"])
    if resume_checkpoint and resume_checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(resume_checkpoint["torch_rng_state"])
    if resume_checkpoint and resume_checkpoint.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng_state_all"])

    if completed_steps >= total_steps:
        print(f"Checkpoint already completed {completed_steps} steps; nothing to run.")
        return model, tokenizer

    model.train()
    start_time = time.time()
    for global_step in range(completed_steps, total_steps):
        step = global_step % steps
        if is_finetune:
            idx = torch.randint(0, len(chat_samples), (batch_size,)).tolist()
            batch = [chat_samples[i] for i in idx]
            x = torch.tensor([item[0] for item in batch], dtype=torch.long, device=device)
            y = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
            mask = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
            logits = model(x)
            token_loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                y.view(-1),
                reduction="none",
            ).view(batch_size, model.block_size)
            loss = (token_loss * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            if len(data) < model.block_size + 1:
                break
            idx = torch.randint(0, len(data) - model.block_size, (batch_size,))
            x = torch.stack([data[i:i+model.block_size] for i in idx]).to(device)
            y = torch.stack([data[i+1:i+model.block_size+1] for i in idx]).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_value = float(loss.item())
        if global_step % 100 == 0:
            print(f"Epoch {global_step // steps + 1} | Step {step}/{steps} | Loss: {loss_value:.4f}")

        next_step = global_step + 1
        if next_step % checkpoint_every == 0:
            if loss_value < best_loss:
                best_loss = loss_value
                atomic_torch_save(
                    make_checkpoint(
                        model, optimizer, tokenizer, model_config,
                        global_step // steps + 1, step + 1, next_step, best_loss, scaler,
                        adapter_only=lora,
                        lora_config=lora_config,
                        base_checkpoint_path=pretrained_path,
                    ),
                    best_path,
                )
                print(f"  ✅ Best checkpoint saved ({best_loss:.4f})")

            atomic_torch_save(
                make_checkpoint(
                    model, optimizer, tokenizer, model_config,
                    global_step // steps + 1, step + 1, next_step, best_loss, scaler,
                    adapter_only=lora,
                    lora_config=lora_config,
                    base_checkpoint_path=pretrained_path,
                ),
                latest_path,
            )
            print(f"  ✅ Latest checkpoint saved at step {next_step}")

    print(f"Training completed in {time.time() - start_time:.1f}s")
    atomic_torch_save(
        make_checkpoint(
            model, optimizer, tokenizer, model_config,
            total_steps // steps, total_steps % steps, total_steps, best_loss, scaler,
            adapter_only=lora,
            lora_config=lora_config,
            base_checkpoint_path=pretrained_path,
        ),
        output_path,
    )
    print(f"✅ Final checkpoint saved to {output_path}")
    return model, tokenizer


# ------------------- Chat Interface -------------------
@torch.no_grad()
def chat(model, tokenizer, max_new_tokens=150, temperature=0.85):
    model.eval()
    device = next(model.parameters()).device
    print("\n=== Chat with your LLM (type 'quit' to exit) ===")
    while True:
        prompt = input("\nYou: ")
        if prompt.lower() in ['quit', 'exit']:
            break
        ids = tokenizer.encode("<|bos|>\nUser: " + prompt + "\nAssistant:")
        x = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            x_cond = x[:, -model.block_size:]
            logits = model(x_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)
            if next_id.item() == tokenizer.special_tokens.get("<|eos|>", -1):
                break
        full = tokenizer.decode(x[0].tolist(), skip_special_tokens=True)
        response = full.split("Assistant:", 1)[-1].strip() if "Assistant:" in full else full
        print("Bot:", response)


# ===================== RUN =====================
if __name__ == "__main__":
    mode = os.environ.get("LLM_TRAIN_MODE", "").lower()
    if mode == "pretrain":
        train_llm(
            data_path="training_data.txt",
            epochs=1,
            lr=6e-4,
            steps=5000,
            checkpoint_prefix="pretrain_500m",
            checkpoint_every=100,
            resume=True,
            output_path="small_llm_pretrained_500m.pth",
            checkpoint_dir=os.environ.get("LLM_CHECKPOINT_DIR", "."),
            device_name=os.environ.get("LLM_DEVICE"),
            dtype_name=os.environ.get("LLM_DTYPE", "auto"),
        )
    elif mode == "finetune":
        train_llm(
            data_path="chat_dataset.jsonl",
            epochs=1,
            pretrained_path="pretrain_500m_latest.pth",
            lr=5e-5,
            steps=2000,
            checkpoint_prefix="finetune_500m",
            checkpoint_every=100,
            resume=True,
            output_path="small_llm.pth",
            checkpoint_dir=os.environ.get("LLM_CHECKPOINT_DIR", "."),
            device_name=os.environ.get("LLM_DEVICE"),
            dtype_name=os.environ.get("LLM_DTYPE", "auto"),
        )
    elif mode == "lora":
        train_llm(
            data_path="chat_dataset.jsonl",
            epochs=1,
            pretrained_path=os.environ.get(
                "LLM_BASE_CHECKPOINT", "pretrain_500m_latest.pth"
            ),
            lr=float(os.environ.get("LLM_LORA_LR", "2e-4")),
            steps=2000,
            checkpoint_prefix="lora_500m",
            checkpoint_every=100,
            resume=True,
            output_path="small_llm_lora.pth",
            checkpoint_dir=os.environ.get("LLM_CHECKPOINT_DIR", "."),
            device_name=os.environ.get("LLM_DEVICE"),
            dtype_name=os.environ.get("LLM_DTYPE", "auto"),
            lora=True,
            lora_rank=int(os.environ.get("LLM_LORA_RANK", "16")),
            lora_alpha=float(os.environ.get("LLM_LORA_ALPHA", "32")),
            lora_dropout=float(os.environ.get("LLM_LORA_DROPOUT", "0.05")),
        )
    else:
        print("Training is paused. Set LLM_TRAIN_MODE=pretrain, finetune, or lora to start explicitly.")

import json
import random
from pathlib import Path

# -------------------- Config --------------------
MIN_MSG_LEN = 3
MAX_TURNS = 10
OUTPUT_FILE = "chat_dataset.jsonl"
TARGET_CONVS = 100000

# -------------------- Helpers --------------------
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = ' '.join(text.split())
    return text

def is_good_message(msg: str) -> bool:
    if len(msg) < MIN_MSG_LEN:
        return False
    alpha = sum(c.isalpha() for c in msg)
    if alpha / max(len(msg), 1) < 0.4:
        return False
    return True

def make_conv(messages: list) -> dict | None:
    if not messages or len(messages) < 2:
        return None
    filtered = [m for m in messages if is_good_message(m.get("content", ""))]
    if len(filtered) < 2:
        return None
    result = []
    roles = ["user", "assistant"]
    for i, m in enumerate(filtered[:MAX_TURNS * 2]):
        result.append({"role": roles[i % 2], "content": clean_text(m["content"])})
    return {"messages": result}

# -------------------- Dataset processors --------------------

def process_oasst1(dataset_split):
    from collections import defaultdict
    by_parent = defaultdict(list)
    by_id = {}
    for item in dataset_split:
        mid = item["message_id"]
        pid = item["parent_id"]
        by_id[mid] = item
        by_parent[pid].append(mid)

    def extract_chain(node_id, depth=0):
        node = by_id.get(node_id)
        if not node or depth > MAX_TURNS * 2:
            return [[]]
        children = by_parent.get(node_id, [])
        if not children:
            return [[node]]
        chains = []
        for child_id in children[:2]:
            for sub in extract_chain(child_id, depth + 1):
                chains.append([node] + sub)
        return chains or [[node]]

    conversations = []
    roots = [mid for mid, item in by_id.items() if item["parent_id"] is None]
    for root_id in roots:
        for chain in extract_chain(root_id):
            messages = [{"role": n["role"], "content": n["text"]} for n in chain]
            conv = make_conv(messages)
            if conv:
                conversations.append(conv)
    return conversations

def process_dolly(dataset_split):
    conversations = []
    for item in dataset_split:
        instruction = item.get("instruction", "").strip()
        context = item.get("context", "").strip()
        response = item.get("response", "").strip()
        if not instruction or not response:
            continue
        user_msg = f"{context}\n{instruction}".strip() if context else instruction
        conv = make_conv([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response}
        ])
        if conv:
            conversations.append(conv)
    return conversations

def process_alpaca(dataset_split):
    conversations = []
    for item in dataset_split:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output = item.get("output", "").strip()
        user_msg = f"{instruction} {input_text}".strip() if input_text else instruction
        if not user_msg or not output:
            continue
        conv = make_conv([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output}
        ])
        if conv:
            conversations.append(conv)
    return conversations

# -------------------- Main --------------------
def main():
    from datasets import load_dataset

    print("📥 Loading conversational datasets...")
    all_convs = []

    # 1. OpenAssistant oasst1 - high quality multi-turn Q&A
    try:
        print("  Loading OpenAssistant/oasst1...")
        oasst = load_dataset("OpenAssistant/oasst1", split="train")
        oasst_convs = process_oasst1(oasst)
        print(f"   oasst1: {len(oasst_convs):,} conversations")
        all_convs.extend(oasst_convs)
    except Exception as e:
        print(f"   ✗ oasst1 failed: {e}")

    # 2. Databricks dolly-15k - instruction following
    try:
        print("  Loading databricks/databricks-dolly-15k...")
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        dolly_convs = process_dolly(dolly)
        print(f"   dolly-15k: {len(dolly_convs):,} conversations")
        all_convs.extend(dolly_convs)
    except Exception as e:
        print(f"   ✗ dolly-15k failed: {e}")

    # 3. Alpaca - instruction following (single-turn, large & clean)
    try:
        print("  Loading tatsu-lab/alpaca...")
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        alpaca_convs = process_alpaca(alpaca)
        print(f"   Alpaca: {len(alpaca_convs):,} conversations")
        all_convs.extend(alpaca_convs)
    except Exception as e:
        print(f"   ✗ Alpaca failed: {e}")

    if not all_convs:
        print("❌ No conversations loaded. Check your internet connection.")
        return

    print(f"\n  Combined: {len(all_convs):,} conversations before deduplication")

    # Deduplicate
    seen = set()
    unique_convs = []
    for conv in all_convs:
        key = "|||".join(m["content"] for m in conv["messages"])
        if key not in seen:
            seen.add(key)
            unique_convs.append(conv)
    print(f"  After dedup: {len(unique_convs):,} unique conversations")

    # Shuffle and limit
    random.seed(42)
    random.shuffle(unique_convs)
    if len(unique_convs) > TARGET_CONVS:
        unique_convs = unique_convs[:TARGET_CONVS]

    # Write JSONL
    output_path = Path(__file__).parent / OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in unique_convs:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Done! Wrote {len(unique_convs):,} conversations ({size_mb:.1f} MB) to {output_path}")
    print("👉 The Train LLM workflow will automatically use chat_dataset.jsonl next run.")

if __name__ == "__main__":
    main()

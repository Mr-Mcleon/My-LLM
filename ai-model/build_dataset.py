import os, re, requests, gzip, io, argparse
from pathlib import Path
from typing import Iterator

# -------------------- Configuration --------------------
TARGET_SIZE_MB = 500        # Final cleaned text size
OUTPUT_FILE = "training_data.txt"   # writes directly to training_data.txt
GUTENBERG_IDS = [
    # Well-known simple English books (all public domain)
    11,    # Alice's Adventures in Wonderland
    12,    # Through the Looking-Glass
    120,   # Treasure Island
    43,    # The Strange Case of Dr. Jekyll and Mr. Hyde
    74,    # The Adventures of Tom Sawyer
    76,    # Adventures of Huckleberry Finn
    1661,  # The Adventures of Sherlock Holmes
    2701,  # Moby Dick
    4300,  # The Wonderful Wizard of Oz
    345,   # Dracula
]

# -------------------- Helper functions --------------------
def clean_text(text: str) -> str:
    """Aggressively clean text for a small LM."""
    text = text.lower()
    # Remove Project Gutenberg headers/footers
    text = re.sub(r'\*\*\* start of this project gutenberg ebook.*?\*\*\*', '', text, flags=re.I|re.DOTALL)
    text = re.sub(r'\*\*\* end of this project gutenberg ebook.*?\*\*\*', '', text, flags=re.I|re.DOTALL)
    # Remove illustrations, chapter headings, etc.
    text = re.sub(r'\[illustration.*?\]', '', text)
    text = re.sub(r'chapter [ivxlcdm]+\.?', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Split into sentences and keep only lines that are mostly alphabetic
    lines = []
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        sentence = sentence.strip()
        if len(sentence) >= 10:
            alpha_ratio = sum(c.isalpha() for c in sentence) / max(len(sentence), 1)
            if alpha_ratio > 0.6:
                lines.append(sentence)
    return ' '.join(lines)

def download_gutenberg(book_id: int) -> str:
    """Fetch a book from Project Gutenberg in plain text."""
    urls = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                print(f"    ✓ Got book {book_id} from {url}")
                return resp.text
        except Exception as e:
            continue
    raise ValueError(f"Could not download book {book_id} from any URL")

def generate_simple_wiki_lines() -> Iterator[str]:
    """Use a pre-filtered Simple Wikipedia dump (manually downloaded)."""
    wiki_file = Path("simplewiki.txt")
    if not wiki_file.exists():
        print("  ℹ️  No simplewiki.txt found — skipping Wikipedia content.")
        return iter([])
    with open(wiki_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

# -------------------- Main pipeline --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_mb", type=int, default=TARGET_SIZE_MB)
    args = parser.parse_args()

    print(f"📚 Building dataset (target: {args.target_mb}MB)...")
    final_docs = []

    # 1. Add Gutenberg books
    for bid in GUTENBERG_IDS:
        print(f"  Downloading Gutenberg ID {bid}...")
        try:
            raw = download_gutenberg(bid)
            cleaned = clean_text(raw)
            paragraphs = [p.strip() for p in cleaned.split('. ') if len(p) > 30]
            final_docs.extend(paragraphs)
            current_mb = sum(len(d.encode('utf-8')) for d in final_docs) / (1024*1024)
            print(f"    → {len(paragraphs):,} paragraphs added | total so far: {current_mb:.1f}MB")
            if current_mb > args.target_mb * 0.7:
                print("  Reached 70% of target from Gutenberg, stopping early.")
                break
        except Exception as e:
            print(f"    ✗ Skipping book {bid}: {e}")

    # 2. Add Simple Wikipedia (optional)
    print("  Checking for Simple Wikipedia content...")
    wiki_docs = list(generate_simple_wiki_lines())[:50000]
    if wiki_docs:
        final_docs.extend(wiki_docs)
        print(f"  Added {len(wiki_docs):,} Wikipedia lines")

    # 3. Deduplicate and filter
    print("  Deduplicating and filtering...")
    final_docs = list(dict.fromkeys(final_docs))
    final_docs = [d for d in final_docs if 20 < len(d) < 1000]

    # 4. Write output
    output_path = Path(__file__).parent / OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as out:
        for doc in final_docs:
            out.write(doc + "\n")

    size_mb = output_path.stat().st_size / (1024*1024)
    print(f"\n✅ Done! Wrote {len(final_docs):,} documents ({size_mb:.1f} MB) to {output_path}")
    print("👉 Run the Train LLM workflow to retrain on this new dataset.")

if __name__ == "__main__":
    main()

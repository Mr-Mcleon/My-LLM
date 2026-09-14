# My LLM

A small GPT-style language model written in PyTorch with a GPT-2 `tiktoken`
tokenizer, a local Flask chat interface, and resumable training scripts.

The current architecture is configured at approximately 505M parameters.
The included checkpoint was an infrastructure test, not a capable base model;
meaningful chat quality requires broader pretraining followed by conversation
fine-tuning.

## Repository layout

- `ai-model/main.py` — model, tokenizer, training loop, checkpointing, and CLI
- `ai-model/colab_train.py` — Colab/GPU entrypoint with Drive-backed checkpoints
- `ai-model/app.py` — local/Replit Flask chat interface
- `ai-model/build_dataset.py` — optional public-domain text dataset builder
- `ai-model/build_chat_dataset.py` — conversation dataset builder
- `colab/Colab_Train_My_LLM.ipynb` — guided Google Colab workflow
- `requirements-colab.txt` — dependencies that are not normally included in Colab

## Run locally

The existing Replit workflows remain unchanged:

```bash
cd ai-model
python main.py       # paused unless LLM_TRAIN_MODE is set
python app.py        # starts the chat server
```

Training is intentionally paused by default. To run a local training mode:

```bash
cd ai-model
LLM_TRAIN_MODE=pretrain python main.py
# or
LLM_TRAIN_MODE=finetune python main.py
```

## Google Colab workflow

1. Push the code to a GitHub repository.
2. Open `colab/Colab_Train_My_LLM.ipynb` in Google Colab.
3. Select a GPU runtime and set `REPO_URL` in the first configuration cell.
4. Mount Google Drive when prompted.
5. Put the larger, license-appropriate corpus in the Drive dataset directory.
6. Run pretraining, then conversation fine-tuning.

See [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) for the complete setup,
dataset, licensing, loading, smoke-test, resume, and LoRA procedure.

Checkpoints are written atomically every 100 steps. The notebook stores them
under `My Drive/my-llm/checkpoints`, so a disconnected Colab runtime can resume
from the latest checkpoint. Use `--no-resume` only when deliberately starting
over.

The current model uses a 64-token context window. Increase that architecture
setting only as a separate experiment after confirming the available GPU
memory.

## Large files and licensing

Model checkpoints and generated datasets should normally live in Google Drive,
Hugging Face Hub, or Git LFS rather than ordinary Git history. The repository
contains existing large local artifacts for continuity, but new checkpoints
are ignored under `ai-model/checkpoints/`.

Before publishing a public repository, review the license and redistribution
terms for every dataset. The Project Gutenberg builder is intended for
public-domain texts; the conversation datasets have their own terms.

## Backup

Before the Colab preparation changes, a verified local backup was created at
`backups/pre-colab-prep-2026-08-16.tar.gz` with a matching `.sha256` file.
Backups are ignored so they cannot be accidentally pushed to GitHub.
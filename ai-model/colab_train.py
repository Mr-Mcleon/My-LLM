"""Colab-friendly command-line entrypoint for the custom GPT-style model.

The training implementation lives in main.py. This wrapper keeps data in the
repository (or Drive) while placing all resumable checkpoints in a directory
that can be backed by Google Drive.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from main import train_llm


def resolve_path(value: str, repo_dir: Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else repo_dir / path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train My LLM from Colab or a local GPU.")
    parser.add_argument("--mode", choices=("pretrain", "finetune", "lora"), required=True)
    parser.add_argument("--data", help="Training text or JSONL file.")
    parser.add_argument("--checkpoint-dir", default=os.environ.get("LLM_CHECKPOINT_DIR"))
    parser.add_argument("--pretrained", help="Base checkpoint for conversation fine-tuning.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cuda or cpu.")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_dir = Path(__file__).resolve().parent
    checkpoint_dir = Path(
        args.checkpoint_dir or (repo_dir / "checkpoints")
    ).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.data:
        data_path = resolve_path(args.data, repo_dir)
    else:
        data_path = str(repo_dir / (
            "training_data.txt" if args.mode == "pretrain" else "chat_dataset.jsonl"
        ))

    if args.mode == "pretrain":
        steps = args.steps if args.steps is not None else 5000
        lr = args.lr if args.lr is not None else 6e-4
        pretrained_path = None
        checkpoint_prefix = "pretrain_500m"
        output_path = "small_llm_pretrained_500m.pth"
        lora = False
    elif args.mode == "lora":
        steps = args.steps if args.steps is not None else 2000
        lr = args.lr if args.lr is not None else 2e-4
        pretrained_path = (
            resolve_path(args.pretrained, repo_dir)
            if args.pretrained
            else str(checkpoint_dir / "pretrain_500m_latest.pth")
        )
        checkpoint_prefix = "lora_500m"
        output_path = "small_llm_lora.pth"
        lora = True
    else:
        steps = args.steps if args.steps is not None else 2000
        lr = args.lr if args.lr is not None else 5e-5
        pretrained_path = (
            resolve_path(args.pretrained, repo_dir)
            if args.pretrained
            else str(checkpoint_dir / "pretrain_500m_latest.pth")
        )
        checkpoint_prefix = "finetune_500m"
        output_path = "small_llm.pth"
        lora = False

    print(f"Mode: {args.mode}")
    print(f"Data: {data_path}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    train_llm(
        data_path=data_path,
        epochs=args.epochs,
        pretrained_path=pretrained_path,
        lr=lr,
        steps=steps,
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        output_path=output_path,
        checkpoint_dir=str(checkpoint_dir),
        device_name=args.device,
        dtype_name=args.dtype,
        lora=lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


if __name__ == "__main__":
    main()
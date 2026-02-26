#!/usr/bin/env python3
"""
Standalone HeartMuLa LoRA Training Script

This script trains a LoRA adapter for HeartMuLa without requiring SimpleTuner's
full pipeline, bypassing validation requirements.

Usage:
    python train_heartmula.py --data_dir ./datasets/max_richter --output_dir ./output
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class AudioTokenDataset(Dataset):
    """Dataset for pre-tokenized audio files."""

    def __init__(self, data_dir: str, max_seq_len: int = 6000):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len

        # Load metadata
        metadata_path = self.data_dir / "metadata.jsonl"
        self.samples = []

        if metadata_path.exists():
            with open(metadata_path) as f:
                for line in f:
                    entry = json.loads(line)
                    token_path = self.data_dir / entry["audio_tokens_path"]
                    if token_path.exists():
                        self.samples.append({
                            "token_path": str(token_path),
                            "tags": entry.get("tags", "music"),
                            "lyrics": entry.get("lyrics", ""),
                        })
        else:
            # Fallback: find all .npy files
            for npy_file in self.data_dir.glob("*.npy"):
                txt_file = npy_file.with_suffix(".txt")
                tags = txt_file.read_text().strip() if txt_file.exists() else "music"
                self.samples.append({
                    "token_path": str(npy_file),
                    "tags": tags,
                    "lyrics": "",
                })

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load tokens [num_codebooks, frames]
        tokens = np.load(sample["token_path"])
        tokens = torch.from_numpy(tokens).long()

        return {
            "audio_tokens": tokens,
            "tags": sample["tags"],
            "lyrics": sample["lyrics"],
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    audio_tokens = [item["audio_tokens"] for item in batch]
    tags = [item["tags"] for item in batch]
    lyrics = [item["lyrics"] for item in batch]

    # Pad tokens to same length
    max_len = max(t.shape[1] for t in audio_tokens)
    num_codebooks = audio_tokens[0].shape[0]

    padded_tokens = torch.zeros(len(batch), num_codebooks, max_len, dtype=torch.long)
    for i, t in enumerate(audio_tokens):
        padded_tokens[i, :, :t.shape[1]] = t

    return {
        "audio_tokens": padded_tokens,
        "tags": tags,
        "lyrics": lyrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train HeartMuLa LoRA")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with tokenized audio")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--model_name", type=str, default="HeartMuLa/HeartMuLa-oss-3B")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--quantize", action="store_true", help="Use int8 quantization")
    parser.add_argument("--max_seq_len", type=int, default=750, help="Max sequence length per chunk")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set dtype
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading HeartMuLa model: {args.model_name}")

    # Load HeartMuLa using SimpleTuner's model class
    try:
        from simpletuner.helpers.models.heartmula.modeling_heartmula import HeartMuLaModel
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download

        # Resolve model path - download if needed
        model_path = args.model_name
        if not os.path.isdir(model_path):
            # Try to download from HuggingFace
            print(f"Downloading model from HuggingFace: {args.model_name}")
            model_path = snapshot_download(
                repo_id=args.model_name,
                local_files_only=False,
            )
            print(f"Model downloaded to: {model_path}")

        # Load the model
        model = HeartMuLaModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )
        model = model.to(device)

        # Load tokenizer - HeartMuLa uses LLaMA tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        except Exception:
            # Fallback to tiktoken-based tokenizer
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Model loaded: {model.__class__.__name__}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Apply LoRA
    print(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    except Exception as e:
        print(f"Error applying LoRA: {e}")
        print("Training full model instead...")

    # Create dataset and dataloader
    dataset = AudioTokenDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # Get model config (handle PEFT wrapper)
    if hasattr(model, 'get_base_model'):
        base_model = model.get_base_model()
        config = base_model.config
    else:
        config = model.config

    num_codebooks = config.audio_num_codebooks  # 8
    audio_vocab_size = config.audio_vocab_size  # 8192
    text_vocab_size = config.text_vocab_size

    print(f"Model config: {num_codebooks} codebooks, audio vocab {audio_vocab_size}, text vocab {text_vocab_size}")

    # Training loop
    model.train()
    global_step = 0
    total_loss = 0.0

    print(f"\nStarting training for {args.max_steps} steps...")

    progress_bar = tqdm(total=args.max_steps, desc="Training")

    while global_step < args.max_steps:
        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            # Prepare inputs
            # audio_tokens: [batch, num_codebooks, frames] from dataset
            audio_tokens = batch["audio_tokens"].to(device)
            tags = batch["tags"]
            lyrics = batch["lyrics"]

            batch_size = audio_tokens.shape[0]
            frames = audio_tokens.shape[2]

            # Format prompt and tokenize
            prompts = []
            for t, l in zip(tags, lyrics):
                if l:
                    prompt = f"<|tags|>{t}<|lyrics|>{l}<|audio|>"
                else:
                    prompt = f"<|tags|>{t}<|audio|>"
                prompts.append(prompt)

            try:
                text_inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)

                # Get text token IDs (use a single token per frame for simplicity)
                # In full training, text tokens are interleaved with audio
                # For overfitting, we use the last token ID repeated
                text_token_id = text_inputs.input_ids[:, -1].unsqueeze(1).expand(-1, frames)  # [batch, frames]

                # Format tokens for HeartMuLa: [batch, frames, num_codebooks + 1]
                # Where the last column is text token, first num_codebooks are audio
                audio_tokens_transposed = audio_tokens.permute(0, 2, 1)  # [batch, frames, num_codebooks]

                # Clamp audio tokens to valid range
                audio_tokens_transposed = torch.clamp(audio_tokens_transposed, 0, audio_vocab_size - 1)

                # Clamp text tokens to valid range
                text_token_id = torch.clamp(text_token_id, 0, text_vocab_size - 1)

                # Combine: [batch, frames, num_codebooks + 1]
                tokens = torch.cat([
                    audio_tokens_transposed,
                    text_token_id.unsqueeze(-1)
                ], dim=-1)

                # Create mask: all ones since we have valid tokens everywhere
                tokens_mask = torch.ones_like(tokens, dtype=torch.float32)

                with torch.autocast(device_type="cuda", dtype=dtype):
                    outputs = model(
                        tokens=tokens,
                        tokens_mask=tokens_mask,
                    )

                    # Compute loss from logits
                    # codebook0_logits: [batch, frames-1, audio_vocab_size]
                    # codebook_logits: [batch, frames-1, num_codebooks-1, audio_vocab_size]
                    codebook0_logits = outputs["codebook0_logits"]
                    codebook_logits = outputs["codebook_logits"]

                    # Target: shifted audio tokens
                    target_cb0 = audio_tokens_transposed[:, 1:, 0]  # [batch, frames-1]
                    target_other = audio_tokens_transposed[:, 1:, 1:]  # [batch, frames-1, num_codebooks-1]

                    # Cross-entropy loss for codebook 0
                    loss_cb0 = torch.nn.functional.cross_entropy(
                        codebook0_logits.reshape(-1, audio_vocab_size),
                        target_cb0.reshape(-1),
                        reduction="mean"
                    )

                    # Cross-entropy loss for other codebooks
                    loss_other = torch.nn.functional.cross_entropy(
                        codebook_logits.reshape(-1, audio_vocab_size),
                        target_other.reshape(-1),
                        reduction="mean"
                    )

                    # Combined loss
                    loss = (loss_cb0 + loss_other) / 2.0
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                total_loss += loss.item()

                if (global_step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            except Exception as e:
                print(f"\nError in training step {global_step}: {e}")
                import traceback
                traceback.print_exc()
                continue

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"})

            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                print(f"\nSaving checkpoint to {checkpoint_path}")
                model.save_pretrained(checkpoint_path)

    progress_bar.close()

    # Save final model
    final_path = os.path.join(args.output_dir, "final")
    print(f"\nSaving final model to {final_path}")
    model.save_pretrained(final_path)

    avg_loss = total_loss / global_step if global_step > 0 else 0
    print(f"\nTraining complete! Average loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()

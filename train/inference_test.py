#!/usr/bin/env python3
"""
Inference test script for trained HeartMuLa model.

This script loads the trained model and generates audio from the training data
to verify the model learned the audio patterns.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Test HeartMuLa inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with tokenized audio")
    parser.add_argument("--output_dir", type=str, default="./inference_output", help="Output directory")
    parser.add_argument("--codec_name", type=str, default="HeartMuLa/HeartCodec-oss-20260123")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Using device: {device}")

    # Load trained HeartMuLa model
    print(f"Loading trained HeartMuLa model from: {args.model_path}")
    try:
        from simpletuner.helpers.models.heartmula.modeling_heartmula import HeartMuLaModel
        model = HeartMuLaModel.from_pretrained(args.model_path, torch_dtype=dtype)
        model = model.to(device)
        model.eval()
        print(f"HeartMuLa loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"Error loading HeartMuLa: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load HeartCodec for detokenization
    print(f"Loading HeartCodec from: {args.codec_name}")
    try:
        from huggingface_hub import snapshot_download
        codec_path = snapshot_download(repo_id=args.codec_name, local_files_only=False)
        print(f"HeartCodec downloaded to: {codec_path}")

        # Try SimpleTuner's HeartCodec
        try:
            from simpletuner.helpers.models.heartmula.codec.modeling_heartcodec import HeartCodec
            codec = HeartCodec.from_pretrained(codec_path, torch_dtype=dtype)
        except Exception:
            # Fall back to local implementation
            from heartlib.heartcodec.modeling_heartcodec import HeartCodec
            codec = HeartCodec.from_pretrained(codec_path, torch_dtype=dtype)

        codec = codec.to(device)
        codec.eval()
        print(f"HeartCodec loaded")
    except Exception as e:
        print(f"Error loading HeartCodec: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing without codec - will only test forward pass...")
        codec = None

    # Load test tokens
    data_dir = Path(args.data_dir)
    npy_files = sorted(data_dir.glob("*.npy"))[:args.num_samples]

    if not npy_files:
        print(f"No .npy files found in {args.data_dir}")
        return

    config = model.config
    num_codebooks = config.audio_num_codebooks
    audio_vocab_size = config.audio_vocab_size
    text_vocab_size = config.text_vocab_size

    print(f"\nModel config: {num_codebooks} codebooks, audio vocab {audio_vocab_size}")
    print(f"Testing {len(npy_files)} samples...\n")

    for i, npy_file in enumerate(npy_files):
        print(f"Processing: {npy_file.name}")

        # Load original tokens
        tokens_np = np.load(npy_file)
        audio_tokens = torch.from_numpy(tokens_np).long().to(device)
        print(f"  Original tokens shape: {audio_tokens.shape}")

        # Format for model: [batch, frames, num_codebooks + 1]
        audio_tokens_transposed = audio_tokens.permute(1, 0).unsqueeze(0)  # [1, frames, 8]
        audio_tokens_transposed = torch.clamp(audio_tokens_transposed, 0, audio_vocab_size - 1)

        # Add text token (use a simple tag token)
        frames = audio_tokens_transposed.shape[1]
        text_token = torch.ones(1, frames, 1, dtype=torch.long, device=device) * 1000

        tokens = torch.cat([audio_tokens_transposed, text_token], dim=-1)
        tokens_mask = torch.ones_like(tokens, dtype=torch.float32)

        print(f"  Input tokens shape: {tokens.shape}")

        # Run forward pass
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            outputs = model(tokens=tokens, tokens_mask=tokens_mask)

        codebook0_logits = outputs["codebook0_logits"]
        codebook_logits = outputs["codebook_logits"]

        print(f"  codebook0_logits: {codebook0_logits.shape}")
        print(f"  codebook_logits: {codebook_logits.shape}")

        # Predict tokens (greedy decoding)
        pred_cb0 = codebook0_logits.argmax(dim=-1)  # [batch, frames-1]
        pred_other = codebook_logits.argmax(dim=-1)  # [batch, frames-1, 7]

        # Compare with targets
        target_cb0 = audio_tokens_transposed[:, 1:, 0]
        target_other = audio_tokens_transposed[:, 1:, 1:]

        acc_cb0 = (pred_cb0 == target_cb0).float().mean().item()
        acc_other = (pred_other == target_other).float().mean().item()

        print(f"  Codebook 0 accuracy: {acc_cb0:.2%}")
        print(f"  Other codebooks accuracy: {acc_other:.2%}")

        # Reconstruct full token sequence
        pred_tokens = torch.cat([
            pred_cb0.unsqueeze(-1),  # [1, frames-1, 1]
            pred_other,              # [1, frames-1, 7]
        ], dim=-1)  # [1, frames-1, 8]

        # Add first frame from original
        first_frame = audio_tokens_transposed[:, :1, :]  # [1, 1, 8]
        full_pred_tokens = torch.cat([first_frame, pred_tokens], dim=1)  # [1, frames, 8]

        # Transpose to codec format: [8, frames]
        pred_for_codec = full_pred_tokens.squeeze(0).permute(1, 0)  # [8, frames]
        orig_for_codec = audio_tokens  # [8, frames]

        print(f"  Predicted tokens for codec: {pred_for_codec.shape}")

        # Decode to audio if codec available
        if codec is not None:
            print("  Decoding audio...")
            try:
                duration = frames / 12.5  # ~12.5 frames per second

                with torch.no_grad():
                    # Decode predicted tokens
                    pred_audio = codec.detokenize(
                        pred_for_codec,
                        duration=duration,
                        num_steps=10,
                        disable_progress=True,
                    )

                    # Decode original tokens
                    orig_audio = codec.detokenize(
                        orig_for_codec,
                        duration=duration,
                        num_steps=10,
                        disable_progress=True,
                    )

                # Save audio
                pred_path = os.path.join(args.output_dir, f"{npy_file.stem}_predicted.wav")
                orig_path = os.path.join(args.output_dir, f"{npy_file.stem}_reconstructed.wav")

                torchaudio.save(pred_path, pred_audio.float().cpu(), 48000)
                torchaudio.save(orig_path, orig_audio.float().cpu(), 48000)

                print(f"  Saved: {pred_path}")
                print(f"  Saved: {orig_path}")

            except Exception as e:
                print(f"  Error decoding: {e}")
                import traceback
                traceback.print_exc()

        print()

    print("Inference test complete!")


if __name__ == "__main__":
    main()

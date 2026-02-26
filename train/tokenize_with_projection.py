#!/usr/bin/env python3
"""
Tokenize audio using the trained encoder projection.

This script uses the learned 128→512 projection to properly encode audio to tokens.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm


class EncoderProjection(nn.Module):
    """Trained projection from ScalarModel latents (128-dim) to VQ input (512-dim)."""

    def __init__(self, input_dim=128, output_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def encode_audio_to_tokens(
    waveform: torch.Tensor,
    codec,
    projection: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode audio waveform to VQ tokens using trained projection.

    Args:
        waveform: [1, samples] audio tensor at 48kHz
        codec: HeartCodec model
        projection: Trained EncoderProjection
        device: torch device

    Returns:
        tokens: [8, frames] tensor of token indices
    """
    audio_input = waveform.unsqueeze(0).to(device)  # [1, 1, samples]

    with torch.no_grad():
        # Step 1: Encode with ScalarModel
        x = audio_input
        for layer in codec.scalar_model.encoder:
            x = layer(x)
        latent_128 = x.permute(0, 2, 1)  # [1, frames, 128]

        # Step 2: Project to 512-dim using trained projection
        latent_512 = projection(latent_128)  # [1, frames, 512]

        # Step 3: Quantize through VQ
        vq = codec.flow_matching.vq_embed
        x_proj = vq.project_in(latent_512.squeeze(0))  # [frames, 32]

        # Residual quantization
        residual = x_proj
        tokens = []
        for layer in vq.layers:
            codebook = layer._codebook.embed.squeeze(0)  # [8192, 32]
            distances = torch.cdist(residual.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
            indices = distances.argmin(dim=-1)
            tokens.append(indices)
            quantized = codebook[indices]
            residual = residual - quantized

        tokens = torch.stack(tokens, dim=0)  # [8, frames]

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Tokenize audio with trained projection")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tokens")
    parser.add_argument("--projection_path", type=str, default="train/encoder_projection/best_projection.pt")
    parser.add_argument("--codec_name", type=str, default="HeartMuLa/HeartCodec-oss-20260123")
    parser.add_argument("--max_duration", type=float, default=60.0, help="Max duration per file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_sr = 48000

    print(f"Device: {device}")

    # Load codec
    print(f"Loading HeartCodec: {args.codec_name}")
    from simpletuner.helpers.models.heartmula.codec.modeling_heartcodec import HeartCodec
    codec = HeartCodec.from_pretrained(args.codec_name, torch_dtype=torch.float32)
    codec = codec.to(device)
    codec.eval()

    # Load trained projection
    print(f"Loading projection: {args.projection_path}")
    checkpoint = torch.load(args.projection_path, map_location=device, weights_only=False)
    projection = EncoderProjection(
        input_dim=128,
        output_dim=512,
        hidden_dim=checkpoint.get("args", {}).get("hidden_dim", 256)
    )
    projection.load_state_dict(checkpoint["projection_state_dict"])
    projection = projection.to(device)
    projection.eval()
    print(f"Projection loaded (trained loss: {checkpoint.get('loss', 'N/A'):.4f})")

    # Find audio files
    input_dir = Path(args.input_dir)
    audio_extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f"**/*{ext}"))

    print(f"Found {len(audio_files)} audio files")

    metadata = []
    for audio_file in tqdm(audio_files, desc="Tokenizing"):
        try:
            # Load audio
            waveform, sr = torchaudio.load(str(audio_file))

            # Resample if needed
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Truncate to max duration
            max_samples = int(args.max_duration * target_sr)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

            # Encode to tokens
            tokens = encode_audio_to_tokens(waveform, codec, projection, device)

            # Save tokens
            rel_path = audio_file.relative_to(input_dir)
            output_name = rel_path.with_suffix(".npy")
            output_path = Path(args.output_dir) / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(output_path, tokens.cpu().numpy())

            duration_sec = waveform.shape[1] / target_sr
            metadata.append({
                "audio_path": str(rel_path),
                "token_path": str(output_name),
                "num_frames": tokens.shape[1],
                "duration_sec": duration_sec,
                "fps": tokens.shape[1] / duration_sec,
            })

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save metadata
    metadata_path = Path(args.output_dir) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTokenized {len(metadata)} files")
    print(f"Tokens saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

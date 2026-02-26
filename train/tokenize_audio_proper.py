#!/usr/bin/env python3
"""
Proper HeartCodec Audio Tokenization

This script properly encodes audio using HeartCodec's ScalarModel encoder
and ResidualVQ quantizer, producing tokens that can be decoded back to audio.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def tokenize_audio(
    audio_path: str,
    codec,
    device: torch.device,
    max_duration: float = 60.0,
    target_sr: int = 48000,
) -> torch.Tensor:
    """
    Tokenize audio file using HeartCodec's encoder and VQ.

    Returns:
        tokens: [8, frames] tensor of token indices
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Truncate to max duration
    max_samples = int(max_duration * target_sr)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    waveform = waveform.to(device)
    audio_input = waveform.unsqueeze(0)  # [1, 1, samples]

    with torch.no_grad():
        # Step 1: Encode with ScalarModel
        x = audio_input
        for layer in codec.scalar_model.encoder:
            x = layer(x)
        latent = x  # [1, 128, frames]

        # Step 2: Prepare for VQ (need 512-dim input)
        latent_t = latent.permute(0, 2, 1)  # [1, frames, 128]
        latent_padded = torch.nn.functional.pad(latent_t, (0, 512 - 128))  # [1, frames, 512]

        # Step 3: Quantize with ResidualVQ
        vq = codec.flow_matching.vq_embed
        x_proj = vq.project_in(latent_padded)  # [1, frames, 32]

        residual = x_proj
        all_indices = []
        for layer in vq.layers:
            quantized, indices, _ = layer(residual)
            all_indices.append(indices)
            residual = residual - quantized

        # Stack indices: [1, frames, 8] -> [frames, 8] -> [8, frames]
        tokens = torch.stack(all_indices, dim=-1).squeeze(0)  # [frames, 8]
        tokens = tokens.permute(1, 0)  # [8, frames]

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Tokenize audio with HeartCodec")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tokens")
    parser.add_argument("--max_duration", type=float, default=60.0, help="Max duration per file in seconds")
    parser.add_argument("--codec_name", type=str, default="HeartMuLa/HeartCodec-oss-20260123")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading HeartCodec: {args.codec_name}")

    from simpletuner.helpers.models.heartmula.codec.modeling_heartcodec import HeartCodec
    codec = HeartCodec.from_pretrained(args.codec_name, torch_dtype=torch.float32)
    codec = codec.to(device)
    codec.eval()
    print("HeartCodec loaded")

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
            tokens = tokenize_audio(
                str(audio_file),
                codec,
                device,
                max_duration=args.max_duration,
            )

            # Save tokens
            rel_path = audio_file.relative_to(input_dir)
            output_name = rel_path.with_suffix(".npy")
            output_path = Path(args.output_dir) / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(output_path, tokens.cpu().numpy())

            metadata.append({
                "audio_path": str(rel_path),
                "token_path": str(output_name),
                "num_frames": tokens.shape[1],
                "duration_sec": tokens.shape[1] / 12.5,
            })

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue

    # Save metadata
    import json
    metadata_path = Path(args.output_dir) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTokenized {len(metadata)} files")
    print(f"Tokens saved to: {args.output_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()

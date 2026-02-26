#!/usr/bin/env python3
"""
Simple HeartCodec Audio Tokenization

Uses ScalarModel encoder + tiled projection to VQ.
This is a fallback approach since HeartCodec's full encoder is not public.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def encode_audio_to_tokens(
    waveform: torch.Tensor,
    codec,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode audio waveform to VQ tokens using ScalarModel + tiled projection.

    Args:
        waveform: [1, samples] audio tensor at 48kHz
        codec: HeartCodec model
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
        latent = x  # [1, 128, frames]

        # Step 2: Prepare for VQ - tile 128 -> 512
        latent_t = latent.permute(0, 2, 1)  # [1, frames, 128]
        latent_tiled = latent_t.repeat(1, 1, 4)  # [1, frames, 512]

        # Step 3: Quantize through VQ
        vq = codec.flow_matching.vq_embed
        x_proj = vq.project_in(latent_tiled.squeeze(0))  # [frames, 32]

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
    parser = argparse.ArgumentParser(description="Tokenize audio with HeartCodec (simple approach)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tokens")
    parser.add_argument("--max_duration", type=float, default=60.0, help="Max duration per file in seconds")
    parser.add_argument("--codec_name", type=str, default="HeartMuLa/HeartCodec-oss-20260123")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_sr = 48000

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
            tokens = encode_audio_to_tokens(waveform, codec, device)

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
            continue

    # Save metadata
    metadata_path = Path(args.output_dir) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTokenized {len(metadata)} files")
    print(f"Tokens saved to: {args.output_dir}")
    print(f"Metadata saved to: {metadata_path}")

    # Print stats
    if metadata:
        avg_fps = np.mean([m["fps"] for m in metadata])
        print(f"Average frame rate: {avg_fps:.1f} fps")


if __name__ == "__main__":
    main()

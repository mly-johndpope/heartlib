#!/usr/bin/env python3
"""
Decode HeartCodec tokens back to audio.

Uses manual decode path since official detokenize has a bug with newer PyTorch.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def manual_decode_tokens(codec, tokens, device, dtype=torch.float32, num_steps=20):
    """
    Manually decode tokens through VQ -> FlowMatching -> ScalarModel.

    Args:
        codec: HeartCodec model
        tokens: [num_codebooks, frames] tensor of token indices
        device: torch device
        dtype: torch dtype
        num_steps: number of diffusion steps

    Returns:
        audio: [1, samples] tensor
    """
    num_codebooks, num_frames = tokens.shape
    flow = codec.flow_matching

    with torch.no_grad():
        # Prepare codes in the expected format: [batch, num_codebooks, frames]
        codes = tokens.unsqueeze(0).to(device)  # [1, 8, frames]

        # Get conditioning from VQ
        conditioning = flow._quantize_condition(codes)  # [1, frames*2, 512]

        total_frames = conditioning.shape[1]
        latent_dim = flow.latent_dim  # 256

        # Initialize noise
        latents = torch.randn(1, total_frames, latent_dim, device=device, dtype=dtype)

        # Use zero incontext (no context)
        incontext_latents = torch.zeros_like(latents)

        # Prepare mask (all positions are "to generate")
        mask = torch.ones(1, total_frames, dtype=torch.int64, device=device) * 2
        mask_float = (mask > 0.5).unsqueeze(-1).float()

        # Apply zero conditioning for masked positions (none in this case)
        zero_cond = flow.zero_cond_embedding1.view(1, 1, -1)
        conditioning = mask_float * conditioning + (1 - mask_float) * zero_cond

        # Time span for Euler solver
        t_span = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)

        # Euler solver
        t = t_span[0]
        dt = t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            stacked = torch.cat([latents, incontext_latents, conditioning], dim=2)
            velocity = flow.estimator(stacked, timestep=t.unsqueeze(-1))

            latents = latents + dt * velocity
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        # Reshape for ScalarModel (256 -> 128 with stereo trick)
        batch = latents.shape[0]
        time_steps = latents.shape[1]
        feat = latents.shape[2]

        reshaped = latents.reshape(batch, time_steps, 2, feat // 2)
        reshaped = reshaped.permute(0, 2, 1, 3)
        reshaped = reshaped.reshape(batch * 2, time_steps, feat // 2)

        # Decode with ScalarModel
        decoded = codec.scalar_model.decode(reshaped.transpose(1, 2))
        audio = decoded.mean(dim=0)

    return audio


def main():
    parser = argparse.ArgumentParser(description="Decode HeartCodec tokens to audio")
    parser.add_argument("--input", type=str, required=True, help="Token file (.npy) or directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for audio")
    parser.add_argument("--codec_name", type=str, default="HeartMuLa/HeartCodec-oss-20260123")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of diffusion steps")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Using device: {device}")
    print(f"Loading HeartCodec: {args.codec_name}")

    from simpletuner.helpers.models.heartmula.codec.modeling_heartcodec import HeartCodec
    codec = HeartCodec.from_pretrained(args.codec_name, torch_dtype=dtype)
    codec = codec.to(device)
    codec.eval()
    print("HeartCodec loaded")

    # Collect input files
    input_path = Path(args.input)
    if input_path.is_file():
        token_files = [input_path]
    else:
        token_files = sorted(input_path.glob("**/*.npy"))

    print(f"Found {len(token_files)} token files")

    for token_file in tqdm(token_files, desc="Decoding"):
        try:
            # Load tokens
            tokens_np = np.load(token_file)
            tokens = torch.from_numpy(tokens_np).long()

            print(f"\n{token_file.name}: {tokens.shape}")

            # Decode
            audio = manual_decode_tokens(codec, tokens, device, dtype, args.num_steps)

            # Save
            if input_path.is_dir():
                rel_path = token_file.relative_to(input_path)
                output_name = rel_path.with_suffix(".wav")
            else:
                output_name = Path(token_file.stem + ".wav")

            output_path = Path(args.output_dir) / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torchaudio.save(str(output_path), audio.cpu(), 48000)
            print(f"  Saved: {output_path}")

            # Print stats
            duration = audio.shape[1] / 48000
            print(f"  Duration: {duration:.1f}s, samples: {audio.shape[1]}")

        except Exception as e:
            print(f"Error processing {token_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone! Audio saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

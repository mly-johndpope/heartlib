#!/usr/bin/env python3
"""
Test the trained encoder projection using OFFICIAL HeartCodec detokenize.
"""

import argparse
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path


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


def encode_to_tokens(waveform, codec, projection, device):
    """Encode audio to VQ tokens using trained projection."""
    audio = waveform.to(device)
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)  # [1, 1, samples]

    with torch.no_grad():
        # Step 1: Encode with ScalarModel
        x = audio
        for layer in codec.scalar_model.encoder:
            x = layer(x)
        latent_128 = x.permute(0, 2, 1)  # [1, frames, 128]

        # Step 2: Project to 512-dim
        latent_512 = projection(latent_128)  # [1, frames, 512]

        # Step 3: Quantize through VQ
        vq = codec.flow_matching.vq_embed
        x_proj = vq.project_in(latent_512)  # [1, frames, 32]

        # Residual quantization
        residual = x_proj
        tokens = []

        for layer in vq.layers:
            codebook = layer._codebook.embed.squeeze(0)  # [8192, 32]
            distances = torch.cdist(residual, codebook.unsqueeze(0).expand(residual.shape[0], -1, -1))
            indices = distances.argmin(dim=-1)
            tokens.append(indices)
            quantized = codebook[indices.view(-1)].view_as(residual)
            residual = residual - quantized

        tokens = torch.stack(tokens, dim=0)  # [8, 1, frames]

    return tokens.squeeze(1)  # [8, frames]


def main():
    parser = argparse.ArgumentParser(description="Test encoder projection with official detokenize")
    parser.add_argument("--audio", type=str, required=True, help="Input audio file")
    parser.add_argument("--projection", type=str, default="train/encoder_projection_fast/best_projection.pt")
    parser.add_argument("--output", type=str, default="test_official.wav")
    parser.add_argument("--codec_name", type=str, default="HeartMuLa/HeartCodec-oss-20260123")
    parser.add_argument("--num_steps", type=int, default=10, help="Diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=1.25, help="CFG scale")
    parser.add_argument("--max_duration", type=float, default=10.0, help="Max duration")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load codec
    print(f"Loading HeartCodec: {args.codec_name}")
    from simpletuner.helpers.models.heartmula.codec.modeling_heartcodec import HeartCodec
    codec = HeartCodec.from_pretrained(args.codec_name, torch_dtype=torch.float32)
    codec = codec.to(device)
    codec.eval()

    # Load projection
    print(f"Loading projection: {args.projection}")
    checkpoint = torch.load(args.projection, map_location=device, weights_only=False)

    projection = EncoderProjection(
        input_dim=128,
        output_dim=512,
        hidden_dim=checkpoint.get("args", {}).get("hidden_dim", 256)
    )
    projection.load_state_dict(checkpoint["projection_state_dict"])
    projection = projection.to(device)
    projection.eval()

    print(f"Checkpoint from epoch {checkpoint.get('epoch', 'N/A')}, loss={checkpoint.get('loss', 'N/A'):.4f}")

    # Load audio
    print(f"Loading audio: {args.audio}")
    waveform, sr = torchaudio.load(args.audio)

    # Resample to 48kHz if needed
    if sr != 48000:
        print(f"Resampling from {sr} to 48000")
        resampler = torchaudio.transforms.Resample(sr, 48000)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Limit duration
    max_samples = int(args.max_duration * 48000)
    if waveform.shape[1] > max_samples:
        print(f"Truncating to {args.max_duration} seconds")
        waveform = waveform[:, :max_samples]

    duration = waveform.shape[1] / 48000
    print(f"Input: {waveform.shape}, duration: {duration:.2f}s")

    # Encode to tokens
    print("Encoding to tokens with trained projection...")
    tokens = encode_to_tokens(waveform, codec, projection, device)
    print(f"Tokens shape: {tokens.shape} (8 codebooks × {tokens.shape[1]} frames)")
    print(f"Token rate: {tokens.shape[1] / duration:.1f} tokens/sec")

    # Decode using OFFICIAL detokenize
    print(f"Decoding with official detokenize (steps={args.num_steps}, guidance={args.guidance_scale})...")
    audio_out = codec.detokenize(
        codes=tokens,
        duration=duration,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        device=device,
        disable_progress=False,
    )
    print(f"Output shape: {audio_out.shape}")

    # Save
    output_path = Path(args.output)
    torchaudio.save(str(output_path), audio_out.cpu(), 48000)
    print(f"Saved reconstruction to: {output_path}")

    # Save original
    orig_path = output_path.with_suffix('.orig.wav')
    torchaudio.save(str(orig_path), waveform.cpu(), 48000)
    print(f"Saved original to: {orig_path}")

    # Metrics
    min_len = min(waveform.shape[1], audio_out.shape[1])
    mse = ((waveform[:, :min_len].cpu() - audio_out[:, :min_len].cpu()) ** 2).mean().item()

    print(f"\n=== Results ===")
    print(f"MSE: {mse:.6f}")
    print(f"Token statistics:")
    for i in range(tokens.shape[0]):
        unique = tokens[i].unique().numel()
        print(f"  Codebook {i}: {unique} unique codes")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test the trained encoder projection on an audio file.
Encodes audio → projection → VQ → decode and saves reconstruction.
"""

import argparse
import torch
import torch.nn as nn
import torchaudio
import numpy as np
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


class ProjectionDecoder(nn.Module):
    """Small decoder to bridge VQ output to ScalarModel decoder."""

    def __init__(self, input_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def encode_decode_with_projection(waveform, codec, projection, proj_decoder, device):
    """
    Full encode-decode with trained projection (no diffusion).
    """
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
        quantized_sum = torch.zeros_like(x_proj)
        residual = x_proj
        tokens = []

        for layer in vq.layers:
            codebook = layer._codebook.embed.squeeze(0)  # [8192, 32]
            distances = torch.cdist(residual, codebook.unsqueeze(0).expand(residual.shape[0], -1, -1))
            indices = distances.argmin(dim=-1)
            tokens.append(indices)
            quantized = codebook[indices.view(-1)].view_as(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized

        tokens = torch.stack(tokens, dim=0)  # [8, 1, frames]

        # Step 4: Project back to 512-dim
        latent_512_out = vq.project_out(quantized_sum)  # [1, frames, 512]

        # Step 5: Use projection decoder to get back to 128-dim
        latent_128_out = proj_decoder(latent_512_out)  # [1, frames, 128]

        # Step 6: Decode with ScalarModel
        decoder_input = latent_128_out.permute(0, 2, 1)  # [1, 128, frames]
        audio_out = codec.scalar_model.decode(decoder_input)  # [1, 1, samples]

    return audio_out.squeeze(0), tokens.squeeze(1), latent_128, latent_512


def main():
    parser = argparse.ArgumentParser(description="Test encoder projection")
    parser.add_argument("--audio", type=str, required=True, help="Input audio file")
    parser.add_argument("--projection", type=str, default="train/encoder_projection_fast/best_projection.pt")
    parser.add_argument("--output", type=str, default="test_reconstruction.wav")
    parser.add_argument("--codec_name", type=str, default="HeartMuLa/HeartCodec-oss-20260123")
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

    # Load projection decoder if available
    proj_decoder = ProjectionDecoder(input_dim=512, output_dim=128)
    if "proj_decoder_state_dict" in checkpoint:
        proj_decoder.load_state_dict(checkpoint["proj_decoder_state_dict"])
        print("Loaded projection decoder from checkpoint")
    else:
        print("Warning: No projection decoder in checkpoint, using random init")
    proj_decoder = proj_decoder.to(device)
    proj_decoder.eval()

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

    # Limit duration for testing (30 seconds)
    max_samples = 30 * 48000
    if waveform.shape[1] > max_samples:
        print(f"Truncating to 30 seconds")
        waveform = waveform[:, :max_samples]

    print(f"Input shape: {waveform.shape}, duration: {waveform.shape[1]/48000:.2f}s")

    # Encode and decode
    print("Encoding and decoding...")
    audio_out, tokens, latent_128, latent_512 = encode_decode_with_projection(
        waveform, codec, projection, proj_decoder, device
    )

    # Match lengths
    min_len = min(waveform.shape[1], audio_out.shape[1])
    waveform_trim = waveform[:, :min_len]
    audio_out_trim = audio_out[:, :min_len]

    # Calculate metrics
    mse = ((waveform_trim.cpu() - audio_out_trim.cpu()) ** 2).mean().item()

    # Save reconstruction
    output_path = Path(args.output)
    torchaudio.save(str(output_path), audio_out_trim.cpu(), 48000)
    print(f"Saved reconstruction to: {output_path}")

    # Save original for comparison
    orig_path = output_path.with_suffix('.orig.wav')
    torchaudio.save(str(orig_path), waveform_trim.cpu(), 48000)
    print(f"Saved original to: {orig_path}")

    # Print stats
    print(f"\n=== Results ===")
    print(f"Input duration: {waveform.shape[1]/48000:.2f}s")
    print(f"Output duration: {audio_out.shape[1]/48000:.2f}s")
    print(f"Tokens shape: {tokens.shape} (8 codebooks × {tokens.shape[1]} frames)")
    print(f"Token rate: {tokens.shape[1] / (waveform.shape[1]/48000):.1f} tokens/sec")
    print(f"MSE: {mse:.6f}")
    print(f"Latent 128 stats: mean={latent_128.mean():.4f}, std={latent_128.std():.4f}")
    print(f"Latent 512 stats: mean={latent_512.mean():.4f}, std={latent_512.std():.4f}")

    # Token statistics
    print(f"\nToken statistics per codebook:")
    for i in range(tokens.shape[0]):
        unique = tokens[i].unique().numel()
        print(f"  Codebook {i}: {unique} unique codes used")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test the trained encoder projection with FULL diffusion decoding.
Encodes audio → projection → VQ → FlowMatching diffusion → decode
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    return tokens.squeeze(1), quantized_sum, latent_512


def decode_with_diffusion(tokens, codec, device, num_steps=16):
    """Decode tokens using full FlowMatching diffusion."""
    with torch.no_grad():
        vq = codec.flow_matching.vq_embed
        flow = codec.flow_matching

        # Reconstruct quantized from tokens
        batch_size = 1
        num_frames = tokens.shape[1]

        quantized_sum = torch.zeros(batch_size, num_frames, 32, device=device)
        for i, layer in enumerate(vq.layers):
            codebook = layer._codebook.embed.squeeze(0)  # [8192, 32]
            indices = tokens[i]  # [frames]
            quantized = codebook[indices]  # [frames, 32]
            quantized_sum = quantized_sum + quantized.unsqueeze(0)

        # Project to 512-dim and apply conditioning
        latent_512_out = vq.project_out(quantized_sum)  # [1, frames, 512]
        conditioning = flow.cond_feature_emb(latent_512_out)
        conditioning = F.interpolate(
            conditioning.permute(0, 2, 1), scale_factor=2, mode="nearest"
        ).permute(0, 2, 1)  # [1, frames*2, 512]

        total_frames = conditioning.shape[1]
        latent_dim = flow.latent_dim
        batch = conditioning.shape[0]

        # FlowMatching diffusion
        torch.manual_seed(42)
        latents = torch.randn(batch, total_frames, latent_dim, device=device)
        incontext_latents = torch.zeros_like(latents)

        t_span = torch.linspace(0, 1, num_steps + 1, device=device)

        for step in range(1, len(t_span)):
            t = t_span[step - 1]
            dt = t_span[step] - t_span[step - 1]

            stacked = torch.cat([latents, incontext_latents, conditioning], dim=2)
            velocity = flow.estimator(stacked, timestep=t.unsqueeze(0).expand(batch))
            latents = latents + dt * velocity

        # Reshape and decode with ScalarModel
        reshaped = latents.reshape(batch, total_frames, 2, latent_dim // 2)
        reshaped = reshaped.permute(0, 2, 1, 3)
        reshaped = reshaped.reshape(batch * 2, total_frames, latent_dim // 2)

        decoded = codec.scalar_model.decode(reshaped.transpose(1, 2))
        audio_out = decoded.view(batch, 2, -1).mean(dim=1)  # [batch, samples]

    return audio_out


def main():
    parser = argparse.ArgumentParser(description="Test encoder projection with full diffusion")
    parser.add_argument("--audio", type=str, required=True, help="Input audio file")
    parser.add_argument("--projection", type=str, default="train/encoder_projection_fast/best_projection.pt")
    parser.add_argument("--output", type=str, default="test_diffusion_reconstruction.wav")
    parser.add_argument("--codec_name", type=str, default="HeartMuLa/HeartCodec-oss-20260123")
    parser.add_argument("--diffusion_steps", type=int, default=16, help="Number of diffusion steps")
    parser.add_argument("--max_duration", type=float, default=10.0, help="Max duration in seconds")
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

    print(f"Input shape: {waveform.shape}, duration: {waveform.shape[1]/48000:.2f}s")

    # Encode to tokens
    print("Encoding to tokens...")
    tokens, quantized, latent_512 = encode_to_tokens(waveform, codec, projection, device)
    print(f"Tokens shape: {tokens.shape}")

    # Decode with diffusion
    print(f"Decoding with {args.diffusion_steps} diffusion steps...")
    audio_out = decode_with_diffusion(tokens, codec, device, num_steps=args.diffusion_steps)

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
    print(f"Tokens: {tokens.shape[0]} codebooks × {tokens.shape[1]} frames")
    print(f"Token rate: {tokens.shape[1] / (waveform.shape[1]/48000):.1f} tokens/sec")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print(f"MSE: {mse:.6f}")

    # Token statistics
    print(f"\nToken statistics per codebook:")
    for i in range(tokens.shape[0]):
        unique = tokens[i].unique().numel()
        print(f"  Codebook {i}: {unique} unique codes used")


if __name__ == "__main__":
    main()

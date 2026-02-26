#!/usr/bin/env python3
"""
Test the trained encoder projection with direct FlowMatching call (bypass buggy detokenize).
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pathlib import Path
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


def encode_to_tokens(waveform, codec, projection, device):
    """Encode audio to VQ tokens using trained projection."""
    audio = waveform.to(device)
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)

    with torch.no_grad():
        x = audio
        for layer in codec.scalar_model.encoder:
            x = layer(x)
        latent_128 = x.permute(0, 2, 1)

        latent_512 = projection(latent_128)

        vq = codec.flow_matching.vq_embed
        x_proj = vq.project_in(latent_512)

        residual = x_proj
        tokens = []

        for layer in vq.layers:
            codebook = layer._codebook.embed.squeeze(0)
            distances = torch.cdist(residual, codebook.unsqueeze(0).expand(residual.shape[0], -1, -1))
            indices = distances.argmin(dim=-1)
            tokens.append(indices)
            quantized = codebook[indices.view(-1)].view_as(residual)
            residual = residual - quantized

        tokens = torch.stack(tokens, dim=0)

    return tokens.squeeze(1)


def decode_tokens_direct(tokens, codec, device, num_steps=16, guidance_scale=1.25):
    """
    Decode tokens using FlowMatching directly (bypass buggy detokenize).

    This implements the core diffusion loop without the buggy incontext_mask code.
    """
    flow = codec.flow_matching
    vq = flow.vq_embed

    batch = 1
    num_frames = tokens.shape[1]
    dtype = next(codec.parameters()).dtype

    with torch.no_grad():
        # Dequantize tokens to get VQ embeddings
        tokens = tokens.to(device)  # [8, frames]

        quantized_sum = torch.zeros(batch, num_frames, 32, device=device, dtype=dtype)
        for i, layer in enumerate(vq.layers):
            codebook = layer._codebook.embed.squeeze(0)  # [8192, 32]
            indices = tokens[i]  # [frames]
            quantized = codebook[indices]  # [frames, 32]
            quantized_sum = quantized_sum + quantized.unsqueeze(0)

        # Project to conditioning space
        latent_512 = vq.project_out(quantized_sum)  # [batch, frames, 512]
        conditioning = flow.cond_feature_emb(latent_512)  # [batch, frames, 512]

        # Upsample conditioning 2x
        conditioning = F.interpolate(
            conditioning.permute(0, 2, 1),
            scale_factor=2,
            mode="nearest"
        ).permute(0, 2, 1)  # [batch, frames*2, 512]

        total_frames = conditioning.shape[1]
        latent_dim = flow.latent_dim

        # Initialize noise
        latents = torch.randn(batch, total_frames, latent_dim, device=device, dtype=dtype)

        # No incontext (we're doing pure generation from tokens)
        incontext_latents = torch.zeros_like(latents)

        # CFG: duplicate for guidance
        if guidance_scale != 1.0:
            latents = torch.cat([latents, latents], dim=0)
            incontext_latents = torch.cat([incontext_latents, incontext_latents], dim=0)
            # Unconditioned uses zero embedding
            uncond = flow.zero_cond_embedding1.view(1, 1, -1).expand(batch, total_frames, -1)
            conditioning = torch.cat([conditioning, uncond], dim=0)

        # Euler solve
        t_span = torch.linspace(0, 1, num_steps + 1, device=device)

        for i in tqdm(range(len(t_span) - 1), desc="Diffusion"):
            t = t_span[i]
            dt = t_span[i + 1] - t

            stacked = torch.cat([latents, incontext_latents, conditioning], dim=2)

            # Get velocity
            timestep = t.expand(latents.shape[0])
            velocity = flow.estimator(stacked, timestep=timestep)

            # CFG
            if guidance_scale != 1.0:
                v_cond, v_uncond = velocity.chunk(2, dim=0)
                velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
                velocity = torch.cat([velocity, velocity], dim=0)

            latents = latents + dt * velocity

        # Take conditioned output
        if guidance_scale != 1.0:
            latents = latents[:batch]

        # Decode with ScalarModel
        # Reshape: [batch, frames*2, latent_dim] -> [batch*2, frames, latent_dim//2]
        reshaped = latents.reshape(batch, total_frames, 2, latent_dim // 2)
        reshaped = reshaped.permute(0, 2, 1, 3)
        reshaped = reshaped.reshape(batch * 2, total_frames, latent_dim // 2)

        decoded = codec.scalar_model.decode(reshaped.transpose(1, 2))
        audio_out = decoded.view(batch, 2, -1).mean(dim=1)  # [batch, samples]

    return audio_out


def main():
    parser = argparse.ArgumentParser(description="Test encoder projection with direct FlowMatching")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--projection", type=str, default="train/encoder_projection_fast/best_projection.pt")
    parser.add_argument("--output", type=str, default="test_direct.wav")
    parser.add_argument("--codec_name", type=str, default="HeartMuLa/HeartCodec-oss-20260123")
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=1.25)
    parser.add_argument("--max_duration", type=float, default=10.0)
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

    print(f"Checkpoint: epoch {checkpoint.get('epoch', 'N/A')}, loss={checkpoint.get('loss', 'N/A'):.4f}")

    # Load audio
    print(f"Loading: {args.audio}")
    waveform, sr = torchaudio.load(args.audio)

    if sr != 48000:
        waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    max_samples = int(args.max_duration * 48000)
    if waveform.shape[1] > max_samples:
        print(f"Truncating to {args.max_duration}s")
        waveform = waveform[:, :max_samples]

    duration = waveform.shape[1] / 48000
    print(f"Input: {duration:.2f}s")

    # Encode
    print("Encoding...")
    tokens = encode_to_tokens(waveform, codec, projection, device)
    print(f"Tokens: {tokens.shape}")

    # Decode
    print(f"Decoding (steps={args.num_steps}, guidance={args.guidance_scale})...")
    audio_out = decode_tokens_direct(tokens, codec, device, args.num_steps, args.guidance_scale)
    print(f"Output: {audio_out.shape}")

    # Save
    output_path = Path(args.output)
    torchaudio.save(str(output_path), audio_out.cpu(), 48000)
    print(f"Saved: {output_path}")

    orig_path = output_path.with_suffix('.orig.wav')
    torchaudio.save(str(orig_path), waveform.cpu(), 48000)
    print(f"Saved original: {orig_path}")

    # Metrics
    min_len = min(waveform.shape[1], audio_out.shape[1])
    mse = ((waveform[:, :min_len].cpu() - audio_out[:, :min_len].cpu()) ** 2).mean().item()
    print(f"\nMSE: {mse:.6f}")


if __name__ == "__main__":
    main()

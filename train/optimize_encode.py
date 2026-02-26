#!/usr/bin/env python3
"""
Optimization-based audio encoding for HeartCodec.

Since HeartCodec doesn't have a public encoder (the 256->512 projection is missing),
this script uses gradient-based optimization to find tokens that decode to the target audio.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    vq = flow.vq_embed

    with torch.no_grad():
        # Prepare codes in the expected format: [batch, num_codebooks, frames]
        codes = tokens.unsqueeze(0)  # [1, 8, frames]

        # Get conditioning from VQ
        # _quantize_condition expects [batch, num_codebooks, frames]
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
        noise = latents.clone()

        for step in range(1, len(t_span)):
            # Forward pass through estimator
            # Input: [latents, incontext_latents, conditioning] concatenated on dim=2
            stacked = torch.cat([latents, incontext_latents, conditioning], dim=2)
            velocity = flow.estimator(stacked, timestep=t.unsqueeze(-1))

            latents = latents + dt * velocity
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        # Reshape for ScalarModel (256 -> 128 with stereo trick)
        batch = latents.shape[0]
        time_steps = latents.shape[1]
        feat = latents.shape[2]  # 256

        reshaped = latents.reshape(batch, time_steps, 2, feat // 2)  # [1, frames, 2, 128]
        reshaped = reshaped.permute(0, 2, 1, 3)  # [1, 2, frames, 128]
        reshaped = reshaped.reshape(batch * 2, time_steps, feat // 2)  # [2, frames, 128]

        # Decode with ScalarModel
        decoded = codec.scalar_model.decode(reshaped.transpose(1, 2))  # [2, 1, samples]
        audio = decoded.mean(dim=0)  # [1, samples]

    return audio


def decode_soft_embeds(codec, soft_embeds, device, dtype=torch.float32, num_steps=8):
    """
    Decode soft embeddings (differentiable) through FlowMatching -> ScalarModel.

    Args:
        codec: HeartCodec model
        soft_embeds: [num_codebooks, frames, 32] soft VQ embeddings
        device: torch device
        dtype: torch dtype
        num_steps: number of diffusion steps (fewer for optimization)

    Returns:
        audio: [1, samples] tensor (differentiable)
    """
    flow = codec.flow_matching
    vq = flow.vq_embed

    # Sum embeddings across codebooks (simulating VQ decode)
    quantized = soft_embeds.sum(dim=0).unsqueeze(0)  # [1, frames, 32]

    # Project out to 512-dim
    latent_512 = vq.project_out(quantized.squeeze(0))  # [frames, 512]
    latent_512 = latent_512.unsqueeze(0)  # [1, frames, 512]

    # Apply conditioning embedding
    conditioning = flow.cond_feature_emb(latent_512)  # [1, frames, 512]

    # Upsample by 2x (as done in _quantize_condition)
    conditioning = F.interpolate(
        conditioning.permute(0, 2, 1), scale_factor=2, mode="nearest"
    ).permute(0, 2, 1)  # [1, frames*2, 512]

    total_frames = conditioning.shape[1]
    latent_dim = flow.latent_dim  # 256

    # Initialize noise (use fixed seed for stability during optimization)
    torch.manual_seed(42)
    latents = torch.randn(1, total_frames, latent_dim, device=device, dtype=dtype)
    noise_copy = latents.clone()

    # Use zero incontext
    incontext_latents = torch.zeros_like(latents)

    # Time span
    t_span = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)

    # Euler solver (differentiable)
    t = t_span[0]
    dt = t_span[1] - t_span[0]

    for step in range(1, len(t_span)):
        stacked = torch.cat([latents, incontext_latents, conditioning], dim=2)
        velocity = flow.estimator(stacked, timestep=t.unsqueeze(-1))
        latents = latents + dt * velocity
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t

    # Reshape for ScalarModel
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


def optimize_encode(
    audio_path: str,
    codec,
    device: torch.device,
    num_steps: int = 300,
    lr: float = 0.1,
    target_sr: int = 48000,
    max_duration: float = 30.0,
):
    """
    Find tokens that decode to the target audio using gradient optimization.

    Returns:
        tokens: [num_codebooks, frames] tensor of token indices
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

    # Truncate/pad to max duration
    max_samples = int(max_duration * target_sr)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    target_audio = waveform.to(device)

    # Calculate number of token frames (~12.5 fps, but output is 2x due to interpolation)
    duration_sec = target_audio.shape[1] / target_sr
    token_frames = int(duration_sec * 12.5) + 1

    print(f"Audio: {duration_sec:.2f}s, {target_audio.shape[1]} samples -> {token_frames} token frames")

    # Get VQ components
    vq = codec.flow_matching.vq_embed
    num_codebooks = len(vq.layers)

    # Initialize soft embeddings
    soft_embeds = torch.randn(
        num_codebooks, token_frames, 32,
        device=device, dtype=torch.float32
    ) * 0.1
    soft_embeds = soft_embeds.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([soft_embeds], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

    best_loss = float('inf')
    best_embeds = None

    pbar = tqdm(range(num_steps), desc="Optimizing")
    for step in pbar:
        optimizer.zero_grad()

        # Decode soft embeddings to audio
        pred_audio = decode_soft_embeds(
            codec, soft_embeds, device,
            dtype=torch.float32, num_steps=8
        )

        # Match lengths
        min_len = min(pred_audio.shape[1], target_audio.shape[1])
        pred_trimmed = pred_audio[:, :min_len]
        target_trimmed = target_audio[:, :min_len]

        # Multi-scale loss
        loss = F.mse_loss(pred_trimmed, target_trimmed)

        # Add spectral loss for perceptual quality
        if min_len >= 2048:
            pred_spec = torch.stft(
                pred_trimmed.squeeze(), n_fft=2048, hop_length=512,
                return_complex=True, window=torch.hann_window(2048, device=device)
            )
            target_spec = torch.stft(
                target_trimmed.squeeze(), n_fft=2048, hop_length=512,
                return_complex=True, window=torch.hann_window(2048, device=device)
            )
            spec_loss = F.mse_loss(pred_spec.abs(), target_spec.abs())
            loss = loss + 0.1 * spec_loss

        # Regularization to keep embeddings in distribution
        reg_loss = 0.001 * (soft_embeds ** 2).mean()
        total_loss = loss + reg_loss

        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([soft_embeds], 1.0)

        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_embeds = soft_embeds.detach().clone()

        if step % 50 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", best=f"{best_loss:.4f}")

    print(f"\nBest loss: {best_loss:.4f}")

    # Quantize to discrete tokens
    print("Quantizing to discrete tokens...")
    tokens = []

    with torch.no_grad():
        residual = best_embeds.clone()  # [num_codebooks, frames, 32]

        # For each codebook, find nearest neighbors
        for cb_idx in range(num_codebooks):
            layer = vq.layers[cb_idx]
            codebook = layer._codebook.embed.squeeze(0)  # [8192, 32]

            # Get residual for this codebook
            cb_embed = residual[cb_idx]  # [frames, 32]

            # Find nearest codebook entry
            distances = torch.cdist(cb_embed.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
            indices = distances.argmin(dim=-1)  # [frames]

            tokens.append(indices)

            # Subtract quantized from residual for next codebook
            quantized = codebook[indices]  # [frames, 32]
            if cb_idx < num_codebooks - 1:
                # Propagate residual to next codebook
                residual[cb_idx + 1] = residual[cb_idx + 1] + (cb_embed - quantized)

    tokens = torch.stack(tokens, dim=0)  # [num_codebooks, frames]

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Optimization-based HeartCodec encoding")
    parser.add_argument("--input", type=str, required=True, help="Input audio file or directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tokens")
    parser.add_argument("--codec_name", type=str, default="HeartMuLa/HeartCodec-oss-20260123")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Max duration per file")
    parser.add_argument("--num_steps", type=int, default=300, help="Optimization steps")
    parser.add_argument("--verify", action="store_true", help="Verify by decoding tokens back to audio")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Loading HeartCodec: {args.codec_name}")

    from simpletuner.helpers.models.heartmula.codec.modeling_heartcodec import HeartCodec
    codec = HeartCodec.from_pretrained(args.codec_name, torch_dtype=torch.float32)
    codec = codec.to(device)
    codec.eval()

    # Collect input files
    input_path = Path(args.input)
    if input_path.is_file():
        audio_files = [input_path]
    else:
        audio_extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f"**/*{ext}"))

    print(f"Found {len(audio_files)} audio files")

    metadata = []

    for audio_file in audio_files:
        print(f"\n{'='*60}")
        print(f"Processing: {audio_file.name}")

        try:
            # Optimize encoding
            tokens = optimize_encode(
                str(audio_file),
                codec,
                device,
                num_steps=args.num_steps,
                max_duration=args.max_duration,
            )

            print(f"Tokens shape: {tokens.shape}")

            # Save tokens
            if input_path.is_dir():
                rel_path = audio_file.relative_to(input_path)
                output_name = rel_path.with_suffix(".npy")
            else:
                output_name = Path(audio_file.stem + ".npy")

            output_path = Path(args.output_dir) / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(output_path, tokens.cpu().numpy())
            print(f"Saved: {output_path}")

            # Verify by decoding
            if args.verify:
                print("Verifying decode...")
                decoded_audio = manual_decode_tokens(codec, tokens, device, num_steps=20)

                verify_path = output_path.with_suffix(".verify.wav")
                torchaudio.save(str(verify_path), decoded_audio.cpu(), 48000)
                print(f"Saved verification: {verify_path}")

            metadata.append({
                "audio_path": str(audio_file.name if input_path.is_file() else audio_file.relative_to(input_path)),
                "token_path": str(output_name),
                "num_frames": tokens.shape[1],
                "duration_sec": tokens.shape[1] / 12.5,
            })

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save metadata
    if metadata:
        import json
        metadata_path = Path(args.output_dir) / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nMetadata saved: {metadata_path}")

    print(f"\nProcessed {len(metadata)} files successfully")


if __name__ == "__main__":
    main()

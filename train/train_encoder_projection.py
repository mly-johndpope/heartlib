#!/usr/bin/env python3
"""
Train the missing 128→512 encoder projection for HeartCodec.

Based on recommendations:
1. Train a learnable projection network (MLP or Conv1D)
2. Use end-to-end reconstruction loss (audio → encode → project → VQ → decode → audio)
3. Multi-scale STFT loss for perceptual quality
4. Optional: Use pseudo-inverse of project_out for initialization
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class EncoderProjection(nn.Module):
    """
    Learnable projection from ScalarModel latents (128-dim) to VQ input (512-dim).

    Options:
    - MLP: Simple multi-layer perceptron
    - Conv1D: 1D convolutions for temporal modeling
    """

    def __init__(self, input_dim=128, output_dim=512, hidden_dim=256, model_type="mlp"):
        super().__init__()
        self.model_type = model_type

        if model_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            )
        elif model_type == "conv1d":
            # Temporal modeling with 1D convolutions
            self.net = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.GroupNorm(8, hidden_dim),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.GroupNorm(8, hidden_dim),
                nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x):
        """
        Args:
            x: [batch, frames, 128] or [batch, 128, frames] for conv1d
        Returns:
            [batch, frames, 512]
        """
        if self.model_type == "mlp":
            return self.net(x)
        else:
            # Conv1D expects [batch, channels, length]
            if x.dim() == 3 and x.shape[-1] == 128:
                x = x.permute(0, 2, 1)  # [batch, 128, frames]
            out = self.net(x)  # [batch, 512, frames]
            return out.permute(0, 2, 1)  # [batch, frames, 512]

    def init_from_pinv(self, project_out_weight, project_in_weight):
        """
        Initialize using pseudo-inverse of project_out to create a meaningful starting point.
        project_out: Linear(32, 512)
        project_in: Linear(512, 32)
        """
        # We want to learn: 128 → 512
        # project_out goes: 32 → 512
        # We can use pinv(project_out) to get 512 → 32 approximation
        # Then chain: 128 → expansion → 512 such that project_in(our_output) ≈ meaningful

        with torch.no_grad():
            if self.model_type == "mlp":
                # Initialize last layer to roughly match project_out structure
                W_out = project_out_weight  # [512, 32]
                W_in = project_in_weight  # [32, 512]

                # Last layer: hidden_dim → 512
                # Initialize to spread the signal across 512-dim in a way compatible with project_in
                # Use SVD to find a good initialization
                U, S, Vt = torch.linalg.svd(W_out, full_matrices=False)

                # Scale the last layer to have similar norm
                scale = W_out.norm() / self.net[-1].weight.norm()
                self.net[-1].weight.data *= scale

                print(f"Initialized projection with pseudo-inverse guidance (scale={scale:.4f})")


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss for perceptual quality."""

    def __init__(self, n_ffts=[512, 1024, 2048], hop_lengths=[128, 256, 512]):
        super().__init__()
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths

    def forward(self, pred, target):
        loss = 0.0
        for n_fft, hop_length in zip(self.n_ffts, self.hop_lengths):
            window = torch.hann_window(n_fft, device=pred.device)

            pred_spec = torch.stft(
                pred.squeeze(1), n_fft=n_fft, hop_length=hop_length,
                window=window, return_complex=True
            )
            target_spec = torch.stft(
                target.squeeze(1), n_fft=n_fft, hop_length=hop_length,
                window=window, return_complex=True
            )

            # Log magnitude loss
            pred_mag = torch.log1p(pred_spec.abs())
            target_mag = torch.log1p(target_spec.abs())
            loss += F.l1_loss(pred_mag, target_mag)

            # Spectral convergence
            loss += torch.norm(target_spec.abs() - pred_spec.abs(), p='fro') / (torch.norm(target_spec.abs(), p='fro') + 1e-8)

        return loss / len(self.n_ffts)


class AudioDataset(Dataset):
    """Simple audio dataset for training."""

    def __init__(self, audio_dir, sample_rate=48000, max_duration=10.0):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)

        # Find all audio files
        extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
        self.files = []
        for ext in extensions:
            self.files.extend(self.audio_dir.glob(f"**/*{ext}"))

        print(f"Found {len(self.files)} audio files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]

        # Load audio
        waveform, sr = torchaudio.load(str(audio_path))

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Truncate or pad
        if waveform.shape[1] > self.max_samples:
            # Random crop
            start = torch.randint(0, waveform.shape[1] - self.max_samples, (1,)).item()
            waveform = waveform[:, start:start + self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            # Pad with zeros
            pad_len = self.max_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_len))

        return waveform


def encode_decode_with_projection(audio, codec, projection, device, num_diffusion_steps=8):
    """
    Full encode-decode roundtrip with learned projection.

    audio → ScalarModel.encode → projection → VQ → FlowMatching → ScalarModel.decode → audio
    """
    # Step 1: Encode with ScalarModel
    x = audio.unsqueeze(1)  # [batch, 1, samples]
    for layer in codec.scalar_model.encoder:
        x = layer(x)
    latent_128 = x.permute(0, 2, 1)  # [batch, frames, 128]

    # Step 2: Project to 512-dim
    latent_512 = projection(latent_128)  # [batch, frames, 512]

    # Step 3: Quantize through VQ
    vq = codec.flow_matching.vq_embed
    x_proj = vq.project_in(latent_512)  # [batch, frames, 32]

    # Residual quantization with straight-through estimator
    quantized_sum = torch.zeros_like(x_proj)
    residual = x_proj
    for layer in vq.layers:
        codebook = layer._codebook.embed.squeeze(0)  # [8192, 32]

        # Compute distances and find nearest
        distances = torch.cdist(residual, codebook.unsqueeze(0).expand(residual.shape[0], -1, -1))
        indices = distances.argmin(dim=-1)

        # Get quantized values
        quantized = codebook[indices.view(-1)].view_as(residual)

        # Straight-through estimator: gradient flows through but values are quantized
        quantized_st = residual + (quantized - residual).detach()
        quantized_sum = quantized_sum + quantized_st
        residual = residual - quantized.detach()

    # Step 4: Project back to 512-dim and apply conditioning
    flow = codec.flow_matching
    latent_512_out = vq.project_out(quantized_sum)  # [batch, frames, 512]
    conditioning = flow.cond_feature_emb(latent_512_out)
    conditioning = F.interpolate(
        conditioning.permute(0, 2, 1), scale_factor=2, mode="nearest"
    ).permute(0, 2, 1)  # [batch, frames*2, 512]

    total_frames = conditioning.shape[1]
    latent_dim = flow.latent_dim
    batch = conditioning.shape[0]

    # Step 5: FlowMatching diffusion
    torch.manual_seed(42)  # Fixed seed for stability
    latents = torch.randn(batch, total_frames, latent_dim, device=device)
    incontext_latents = torch.zeros_like(latents)

    t_span = torch.linspace(0, 1, num_diffusion_steps + 1, device=device)
    t = t_span[0]
    dt = t_span[1] - t_span[0]

    for step in range(1, len(t_span)):
        stacked = torch.cat([latents, incontext_latents, conditioning], dim=2)
        velocity = flow.estimator(stacked, timestep=t.unsqueeze(-1))
        latents = latents + dt * velocity
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t

    # Step 6: Reshape and decode with ScalarModel
    reshaped = latents.reshape(batch, total_frames, 2, latent_dim // 2)
    reshaped = reshaped.permute(0, 2, 1, 3)
    reshaped = reshaped.reshape(batch * 2, total_frames, latent_dim // 2)

    decoded = codec.scalar_model.decode(reshaped.transpose(1, 2))
    audio_out = decoded.view(batch, 2, -1).mean(dim=1, keepdim=True).squeeze(1)

    return audio_out, latent_512


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Enable mixed precision for memory efficiency
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Load codec
    print(f"Loading HeartCodec: {args.codec_name}")
    from simpletuner.helpers.models.heartmula.codec.modeling_heartcodec import HeartCodec
    codec = HeartCodec.from_pretrained(args.codec_name, torch_dtype=torch.float32)
    codec = codec.to(device)
    codec.eval()

    # Freeze codec weights
    for param in codec.parameters():
        param.requires_grad = False

    # Create projection network
    projection = EncoderProjection(
        input_dim=128,
        output_dim=512,
        hidden_dim=args.hidden_dim,
        model_type=args.projection_type
    ).to(device)

    # Optional: Initialize from pseudo-inverse
    if args.init_pinv:
        vq = codec.flow_matching.vq_embed
        projection.init_from_pinv(
            vq.project_out.weight.data,
            vq.project_in.weight.data
        )

    print(f"Projection params: {sum(p.numel() for p in projection.parameters()):,}")

    # Dataset and dataloader
    dataset = AudioDataset(args.data_dir, max_duration=args.max_duration)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(projection.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(dataloader))

    # Loss functions
    stft_loss = MultiScaleSTFTLoss().to(device)

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        projection.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, audio in enumerate(pbar):
            audio = audio.to(device)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=use_amp):
                audio_reconstructed, latent_512 = encode_decode_with_projection(
                    audio.squeeze(1), codec, projection, device,
                    num_diffusion_steps=args.diffusion_steps
                )

                # Match lengths (decode may produce 2x due to upsampling)
                min_len = min(audio.shape[-1], audio_reconstructed.shape[-1])
                audio_trim = audio[:, :, :min_len].squeeze(1)
                recon_trim = audio_reconstructed[:, :min_len]

                # Losses
                loss_stft = stft_loss(recon_trim, audio_trim)
                loss_mse = F.mse_loss(recon_trim, audio_trim)

                # Regularization on projection output
                loss_reg = 0.001 * (latent_512 ** 2).mean()

                loss = loss_stft + 0.1 * loss_mse + loss_reg

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", stft=f"{loss_stft.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                "epoch": epoch,
                "projection_state_dict": projection.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_projection.pt"))
            print(f"Saved best checkpoint (loss={avg_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "epoch": epoch,
                "projection_state_dict": projection.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt"))

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train encoder projection for HeartCodec")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--output_dir", type=str, default="./encoder_projection", help="Output directory")
    parser.add_argument("--codec_name", type=str, default="HeartMuLa/HeartCodec-oss-20260123")
    parser.add_argument("--projection_type", type=str, default="mlp", choices=["mlp", "conv1d"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_duration", type=float, default=10.0)
    parser.add_argument("--diffusion_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--init_pinv", action="store_true", help="Initialize from pseudo-inverse")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

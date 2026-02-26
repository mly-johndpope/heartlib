#!/usr/bin/env python3
"""
Train the missing 128→512 encoder projection for HeartCodec - V2.

Improvements over v1:
1. Gumbel-Softmax with temperature annealing (replaces STE)
2. Code revival for dead codebook entries
3. Commitment loss + entropy regularization
4. Dithering before quantization
5. Better gradient flow through VQ bottleneck
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
        if self.model_type == "mlp":
            return self.net(x)
        else:
            if x.dim() == 3 and x.shape[-1] == 128:
                x = x.permute(0, 2, 1)
            out = self.net(x)
            return out.permute(0, 2, 1)

    def init_from_pinv(self, project_out_weight, project_in_weight):
        with torch.no_grad():
            if self.model_type == "mlp":
                W_out = project_out_weight
                scale = W_out.norm() / self.net[-1].weight.norm()
                self.net[-1].weight.data *= scale
                print(f"Initialized projection with pseudo-inverse guidance (scale={scale:.4f})")


class CodebookTracker:
    """Track codebook utilization and perform code revival."""

    def __init__(self, num_codebooks=8, num_entries=8192, revival_threshold=0.001, device='cuda'):
        self.num_codebooks = num_codebooks
        self.num_entries = num_entries
        self.revival_threshold = revival_threshold
        self.device = device

        # Track usage counts
        self.usage_counts = [torch.zeros(num_entries, device=device) for _ in range(num_codebooks)]
        self.total_counts = [0 for _ in range(num_codebooks)]

    def update(self, codebook_idx, indices):
        """Update usage counts for a codebook."""
        flat_indices = indices.view(-1)
        for idx in flat_indices:
            self.usage_counts[codebook_idx][idx] += 1
        self.total_counts[codebook_idx] += flat_indices.numel()

    def get_dead_codes(self, codebook_idx):
        """Get indices of dead codes (usage below threshold)."""
        if self.total_counts[codebook_idx] == 0:
            return []

        usage_freq = self.usage_counts[codebook_idx] / self.total_counts[codebook_idx]
        dead_mask = usage_freq < self.revival_threshold
        return torch.where(dead_mask)[0].tolist()

    def get_utilization(self, codebook_idx):
        """Get percentage of codes being used."""
        if self.total_counts[codebook_idx] == 0:
            return 0.0
        usage_freq = self.usage_counts[codebook_idx] / self.total_counts[codebook_idx]
        return (usage_freq > self.revival_threshold).float().mean().item() * 100

    def reset(self):
        """Reset counters (call at start of epoch)."""
        for i in range(self.num_codebooks):
            self.usage_counts[i].zero_()
            self.total_counts[i] = 0


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
                pred.squeeze(1) if pred.dim() > 2 else pred,
                n_fft=n_fft, hop_length=hop_length,
                window=window, return_complex=True
            )
            target_spec = torch.stft(
                target.squeeze(1) if target.dim() > 2 else target,
                n_fft=n_fft, hop_length=hop_length,
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

        extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
        self.files = []
        for ext in extensions:
            self.files.extend(self.audio_dir.glob(f"**/*{ext}"))

        print(f"Found {len(self.files)} audio files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        waveform, sr = torchaudio.load(str(audio_path))

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.shape[1] > self.max_samples:
            start = torch.randint(0, waveform.shape[1] - self.max_samples, (1,)).item()
            waveform = waveform[:, start:start + self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            pad_len = self.max_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_len))

        return waveform


def gumbel_softmax_quantize(residual, codebook, temperature, hard=True, top_k=64):
    """
    Memory-efficient Gumbel-Softmax quantization with top-k approximation.

    Args:
        residual: [batch, frames, dim] tensor to quantize
        codebook: [num_entries, dim] codebook embeddings
        temperature: softmax temperature (lower = harder)
        hard: if True, use straight-through for forward pass
        top_k: only consider top-k nearest codes (memory optimization)

    Returns:
        quantized: [batch, frames, dim] quantized values
        indices: [batch, frames] selected indices
        entropy: scalar entropy estimate
    """
    batch, frames, dim = residual.shape
    num_entries = codebook.shape[0]

    # Compute distances: [batch, frames, num_entries]
    distances = torch.cdist(residual, codebook.unsqueeze(0).expand(batch, -1, -1))

    # Get hard indices (always needed for tracking)
    indices = distances.argmin(dim=-1)

    # Memory-efficient: only use top-k for soft computation
    if top_k < num_entries:
        # Get top-k smallest distances
        topk_dists, topk_indices = distances.topk(top_k, dim=-1, largest=False)

        # Gumbel-Softmax on top-k only
        logits = -topk_dists / (temperature + 1e-8)
        soft_probs = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)

        # Gather top-k codebook entries and compute weighted sum
        # topk_indices: [batch, frames, top_k]
        topk_codebook = codebook[topk_indices.view(-1)].view(batch, frames, top_k, dim)
        quantized = torch.einsum('bfk,bfkd->bfd', soft_probs, topk_codebook)

        # Estimate entropy from top-k distribution
        avg_probs = soft_probs.mean(dim=[0, 1])
        entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
    else:
        # Full softmax (original behavior)
        logits = -distances / (temperature + 1e-8)
        soft_probs = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        quantized = torch.einsum('bfn,nd->bfd', soft_probs, codebook)

        avg_probs = soft_probs.mean(dim=[0, 1])
        entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()

    return quantized, indices, entropy


def encode_decode_with_projection_v2(
    audio, codec, projection, device,
    num_diffusion_steps=8,
    temperature=1.0,
    dither_scale=0.0,
    codebook_tracker=None,
    commitment_weight=0.25
):
    """
    Full encode-decode roundtrip with learned projection - V2 with improvements.

    Returns:
        audio_out: reconstructed audio
        latent_512: projected latents
        vq_losses: dict of VQ-related losses
    """
    # Step 1: Encode with ScalarModel
    x = audio.unsqueeze(1)
    for layer in codec.scalar_model.encoder:
        x = layer(x)
    latent_128 = x.permute(0, 2, 1)  # [batch, frames, 128]

    # Step 2: Project to 512-dim
    latent_512 = projection(latent_128)  # [batch, frames, 512]

    # Step 3: Quantize through VQ with improvements
    vq = codec.flow_matching.vq_embed
    x_proj = vq.project_in(latent_512)  # [batch, frames, 32]

    # Add dithering before quantization (training only)
    if dither_scale > 0:
        x_proj = x_proj + dither_scale * torch.randn_like(x_proj)

    # Residual quantization with Gumbel-Softmax
    quantized_sum = torch.zeros_like(x_proj)
    residual = x_proj
    commitment_loss = 0.0
    entropy_loss = 0.0

    for layer_idx, layer in enumerate(vq.layers):
        codebook = layer._codebook.embed.squeeze(0)  # [8192, 32]

        # Gumbel-Softmax quantization (memory-efficient with top-k=64)
        quantized, indices, entropy = gumbel_softmax_quantize(
            residual, codebook, temperature, hard=True, top_k=64
        )

        # Track codebook usage
        if codebook_tracker is not None:
            codebook_tracker.update(layer_idx, indices)

        # Commitment loss: encourage encoder output to be close to codebook
        commitment_loss += F.mse_loss(residual, quantized.detach())

        # Entropy regularization: encourage diverse code usage
        # Higher entropy = more uniform distribution = better utilization
        max_entropy = np.log(64)  # Max entropy for top-k=64
        entropy_loss += (max_entropy - entropy) / max_entropy  # Normalized, lower is better

        quantized_sum = quantized_sum + quantized
        residual = residual - quantized.detach()

    # Normalize losses
    commitment_loss = commitment_loss / len(vq.layers)
    entropy_loss = entropy_loss / len(vq.layers)

    # Step 4: Project back to 512-dim and apply conditioning
    flow = codec.flow_matching
    latent_512_out = vq.project_out(quantized_sum)
    conditioning = flow.cond_feature_emb(latent_512_out)
    conditioning = F.interpolate(
        conditioning.permute(0, 2, 1), scale_factor=2, mode="nearest"
    ).permute(0, 2, 1)

    total_frames = conditioning.shape[1]
    latent_dim = flow.latent_dim
    batch = conditioning.shape[0]

    # Step 5: FlowMatching diffusion
    torch.manual_seed(42)
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

    vq_losses = {
        'commitment': commitment_loss,
        'entropy': entropy_loss,
    }

    return audio_out, latent_512, vq_losses


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Load codec
    print(f"Loading HeartCodec: {args.codec_name}")
    from simpletuner.helpers.models.heartmula.codec.modeling_heartcodec import HeartCodec
    codec = HeartCodec.from_pretrained(args.codec_name, torch_dtype=torch.float32)
    codec = codec.to(device)
    codec.eval()

    for param in codec.parameters():
        param.requires_grad = False

    # Create projection network
    projection = EncoderProjection(
        input_dim=128,
        output_dim=512,
        hidden_dim=args.hidden_dim,
        model_type=args.projection_type
    ).to(device)

    # Load from checkpoint if resuming
    start_epoch = 0
    best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        projection.load_state_dict(checkpoint["projection_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("loss", float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")
    elif args.init_pinv:
        vq = codec.flow_matching.vq_embed
        projection.init_from_pinv(
            vq.project_out.weight.data,
            vq.project_in.weight.data
        )

    print(f"Projection params: {sum(p.numel() for p in projection.parameters()):,}")

    # Codebook tracker for utilization monitoring and code revival
    codebook_tracker = CodebookTracker(
        num_codebooks=8,
        num_entries=8192,
        revival_threshold=args.revival_threshold,
        device=device
    )

    # Dataset and dataloader
    dataset = AudioDataset(args.data_dir, max_duration=args.max_duration)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(projection.parameters(), lr=args.lr, weight_decay=0.01)

    # Scheduler with warmup
    total_steps = args.epochs * len(dataloader)
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss functions
    stft_loss = MultiScaleSTFTLoss().to(device)

    # Temperature annealing schedule
    temp_start = args.temp_start
    temp_end = args.temp_end

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        projection.train()
        codebook_tracker.reset()

        # Anneal temperature: high at start (soft), low at end (hard)
        progress = epoch / max(args.epochs - 1, 1)
        temperature = temp_start * (temp_end / temp_start) ** progress

        # Anneal dithering: high at start, zero at end
        dither_scale = args.dither_start * (1 - progress)

        total_loss = 0.0
        total_stft = 0.0
        total_commit = 0.0
        total_entropy = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, audio in enumerate(pbar):
            audio = audio.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                audio_reconstructed, latent_512, vq_losses = encode_decode_with_projection_v2(
                    audio.squeeze(1), codec, projection, device,
                    num_diffusion_steps=args.diffusion_steps,
                    temperature=temperature,
                    dither_scale=dither_scale,
                    codebook_tracker=codebook_tracker,
                    commitment_weight=args.commitment_weight
                )

                min_len = min(audio.shape[-1], audio_reconstructed.shape[-1])
                audio_trim = audio[:, :, :min_len].squeeze(1)
                recon_trim = audio_reconstructed[:, :min_len]

                # Losses
                loss_stft = stft_loss(recon_trim, audio_trim)
                loss_mse = F.mse_loss(recon_trim, audio_trim)
                loss_commit = vq_losses['commitment']
                loss_entropy = vq_losses['entropy']

                # Regularization on projection output
                loss_reg = 0.001 * (latent_512 ** 2).mean()

                # Combined loss
                loss = (
                    loss_stft +
                    0.1 * loss_mse +
                    args.commitment_weight * loss_commit +
                    args.entropy_weight * loss_entropy +
                    loss_reg
                )

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
            total_stft += loss_stft.item()
            total_commit += loss_commit.item()
            total_entropy += loss_entropy.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                stft=f"{loss_stft.item():.4f}",
                temp=f"{temperature:.3f}"
            )

        avg_loss = total_loss / len(dataloader)
        avg_stft = total_stft / len(dataloader)
        avg_commit = total_commit / len(dataloader)
        avg_entropy = total_entropy / len(dataloader)

        # Get codebook utilization
        utilizations = [codebook_tracker.get_utilization(i) for i in range(8)]
        avg_util = sum(utilizations) / len(utilizations)

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, stft={avg_stft:.4f}, "
              f"commit={avg_commit:.4f}, entropy={avg_entropy:.4f}, "
              f"temp={temperature:.3f}, util={avg_util:.1f}%")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                "epoch": epoch,
                "projection_state_dict": projection.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "stft_loss": avg_stft,
                "temperature": temperature,
                "utilization": avg_util,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_projection.pt"))
            print(f"Saved best checkpoint (loss={avg_loss:.4f}, util={avg_util:.1f}%)")

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
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
    parser = argparse.ArgumentParser(description="Train encoder projection for HeartCodec - V2")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--output_dir", type=str, default="./encoder_projection_v2", help="Output directory")
    parser.add_argument("--codec_name", type=str, default="HeartMuLa/HeartCodec-oss-20260123")
    parser.add_argument("--projection_type", type=str, default="mlp", choices=["mlp", "conv1d"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_duration", type=float, default=10.0)
    parser.add_argument("--diffusion_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--init_pinv", action="store_true", help="Initialize from pseudo-inverse")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # V2 specific args
    parser.add_argument("--temp_start", type=float, default=1.0, help="Starting temperature for Gumbel-Softmax")
    parser.add_argument("--temp_end", type=float, default=0.1, help="Ending temperature for Gumbel-Softmax")
    parser.add_argument("--dither_start", type=float, default=0.1, help="Starting dither scale")
    parser.add_argument("--commitment_weight", type=float, default=0.25, help="Commitment loss weight")
    parser.add_argument("--entropy_weight", type=float, default=0.1, help="Entropy regularization weight")
    parser.add_argument("--revival_threshold", type=float, default=0.001, help="Threshold for dead code detection")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

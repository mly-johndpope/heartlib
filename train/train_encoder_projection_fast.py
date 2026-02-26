#!/usr/bin/env python3
"""
Fast training for encoder projection - bypasses diffusion.

UPGRADED VERSION with:
- Stronger gradient clipping (0.5) and better NaN handling
- Proper warmup + cosine decay scheduler
- Validation callback with real detokenize every N epochs
- Higher diffusion steps during validation
- Robust checkpoint/resume with scheduler state
- Safer AMP usage
- Optional phase loss

Pipeline: audio → encode → project → VQ → decode (no diffusion)
"""

import argparse
import os
import random
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm


class EncoderProjection(nn.Module):
    """Learnable projection from encoder latents (128-dim) to VQ input (512-dim)."""

    def __init__(self, input_dim=128, output_dim=512, hidden_dim=256, projection_type="linear"):
        super().__init__()
        self.projection_type = projection_type

        if projection_type == "linear":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            )
        elif projection_type == "conv1d":
            # Conv1D variant that may capture temporal structure better
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
            raise ValueError(f"Unknown projection_type: {projection_type}")

    def forward(self, x):
        if self.projection_type == "conv1d":
            # x: [batch, frames, channels] -> [batch, channels, frames]
            x = x.permute(0, 2, 1)
            x = self.net(x)
            x = x.permute(0, 2, 1)
            return x
        return self.net(x)


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss for perceptual quality."""

    def __init__(self, n_ffts=[512, 1024, 2048], hop_lengths=[128, 256, 512],
                 include_phase=False, highfreq_weight=0.0, sample_rate=48000):
        super().__init__()
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.include_phase = include_phase
        self.highfreq_weight = highfreq_weight  # Extra weight on 4-8 kHz (shimmer zone)
        self.sample_rate = sample_rate

    def forward(self, pred, target):
        loss = 0.0
        phase_loss = 0.0

        for n_fft, hop_length in zip(self.n_ffts, self.hop_lengths):
            window = torch.hann_window(n_fft, device=pred.device)

            # Handle batch dimension
            if pred.dim() == 2:
                pred_in = pred
                target_in = target
            else:
                pred_in = pred.squeeze(1)
                target_in = target.squeeze(1)

            pred_spec = torch.stft(
                pred_in, n_fft=n_fft, hop_length=hop_length,
                window=window, return_complex=True
            )
            target_spec = torch.stft(
                target_in, n_fft=n_fft, hop_length=hop_length,
                window=window, return_complex=True
            )

            # Log magnitude loss
            pred_mag = torch.log1p(pred_spec.abs())
            target_mag = torch.log1p(target_spec.abs())
            loss += F.l1_loss(pred_mag, target_mag)

            # Spectral convergence
            sc_loss = torch.norm(target_spec.abs() - pred_spec.abs(), p='fro') / (torch.norm(target_spec.abs(), p='fro') + 1e-8)
            loss += sc_loss

            # Optional phase loss (weighted lower)
            if self.include_phase:
                pred_phase = torch.angle(pred_spec)
                target_phase = torch.angle(target_spec)
                # Use instantaneous frequency difference (more stable than raw phase)
                phase_diff = torch.abs(pred_phase - target_phase)
                # Wrap to [-pi, pi]
                phase_diff = torch.remainder(phase_diff + torch.pi, 2 * torch.pi) - torch.pi
                phase_loss += phase_diff.abs().mean()

            # Extra loss on high-frequency "shimmer zone" (4-8 kHz)
            # This targets the exact artifacts deshimmer fixes
            if self.highfreq_weight > 0:
                freq_bins = n_fft // 2 + 1
                freqs = torch.linspace(0, self.sample_rate / 2, freq_bins, device=pred.device)
                # Create mask for 4-8 kHz range (shimmer zone)
                hf_mask = ((freqs >= 4000) & (freqs <= 8000)).float()
                hf_mask = hf_mask.view(1, -1, 1)  # [1, freq_bins, 1] for broadcasting

                # Weighted high-freq magnitude loss
                hf_pred_mag = pred_mag * hf_mask
                hf_target_mag = target_mag * hf_mask
                loss += self.highfreq_weight * F.l1_loss(hf_pred_mag, hf_target_mag)

        total_loss = loss / len(self.n_ffts)
        if self.include_phase:
            total_loss = total_loss + 0.1 * (phase_loss / len(self.n_ffts))

        return total_loss


class MelSpectrogramLoss(nn.Module):
    """Mel-spectrogram loss for perceptual quality."""

    def __init__(self, sample_rate=48000, n_mels=80, n_fft=1024, hop_length=256):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def forward(self, pred, target):
        # Move transform to correct device
        self.mel_transform = self.mel_transform.to(pred.device)

        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)

        pred_mel = self.mel_transform(pred)
        target_mel = self.mel_transform(target)

        # Log mel loss
        pred_log = torch.log1p(pred_mel + 1e-8)
        target_log = torch.log1p(target_mel + 1e-8)

        loss = F.l1_loss(pred_log, target_log)
        return torch.clamp(loss, 0, 100)  # Prevent NaN propagation


class AudioDataset(Dataset):
    """Audio dataset with error handling and balanced sampling option."""

    def __init__(self, audio_dir, sample_rate=48000, max_duration=5.0,
                 genre_balanced=False, min_file_duration=1.0):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.min_samples = int(min_file_duration * sample_rate)

        extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
        self.files = []
        for root, dirs, files in os.walk(audio_dir, followlinks=True):
            for f in files:
                if Path(f).suffix.lower() in extensions:
                    self.files.append(Path(root) / f)

        # Shuffle files to improve diversity within epochs
        random.shuffle(self.files)
        print(f"Found {len(self.files)} audio files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        for attempt in range(3):
            try:
                audio_path = self.files[idx]
                waveform, sr = torchaudio.load(str(audio_path))

                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Skip very short files
                if waveform.shape[1] < self.min_samples:
                    idx = torch.randint(0, len(self.files), (1,)).item()
                    continue

                if waveform.shape[1] > self.max_samples:
                    start = torch.randint(0, waveform.shape[1] - self.max_samples, (1,)).item()
                    waveform = waveform[:, start:start + self.max_samples]
                elif waveform.shape[1] < self.max_samples:
                    pad_len = self.max_samples - waveform.shape[1]
                    waveform = F.pad(waveform, (0, pad_len))

                # Normalize audio to prevent extreme values
                max_val = waveform.abs().max()
                if max_val > 1.0:
                    waveform = waveform / max_val

                return waveform
            except Exception as e:
                idx = torch.randint(0, len(self.files), (1,)).item()

        return torch.zeros(1, self.max_samples)


def encode_with_vq_fast(audio, codec, projection, device):
    """
    Fast encode through VQ without diffusion.

    Returns quantized latents and losses for training.
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

    # Residual quantization with commitment loss
    quantized_sum = torch.zeros_like(x_proj)
    residual = x_proj
    commitment_loss = 0.0
    codebook_usage = []

    for layer_idx, layer in enumerate(vq.layers):
        codebook = layer._codebook.embed.squeeze(0)  # [8192, 32]

        # Find nearest codebook entry
        distances = torch.cdist(residual, codebook.unsqueeze(0).expand(residual.shape[0], -1, -1))
        indices = distances.argmin(dim=-1)

        # Track codebook usage
        unique_codes = indices.unique().numel()
        codebook_usage.append(unique_codes)

        # Get quantized values
        quantized = codebook[indices.view(-1)].view_as(residual)

        # Commitment loss: encoder output should commit to codebook
        commitment_loss += F.mse_loss(residual, quantized.detach())

        # Straight-through estimator
        quantized_st = residual + (quantized - residual).detach()
        quantized_sum = quantized_sum + quantized_st
        residual = residual - quantized.detach()

    commitment_loss = commitment_loss / len(vq.layers)
    avg_usage = sum(codebook_usage) / len(codebook_usage)

    # Step 4: Project back to 512-dim
    latent_512_out = vq.project_out(quantized_sum)  # [batch, frames, 512]

    return latent_512, latent_512_out, commitment_loss, avg_usage


def decode_fast(latent_512_out, codec, batch_size):
    """
    Fast decode without diffusion.

    Uses a simple linear projection to ScalarModel decoder input space.
    """
    frames = latent_512_out.shape[1]

    # Simple approach: reshape and use decoder
    latent_128 = latent_512_out[..., :128]  # Take first 128 dims as approximation

    # Decoder expects [batch, 128, frames]
    latent_for_decoder = latent_128.permute(0, 2, 1)

    # Decode
    audio_out = codec.scalar_model.decode(latent_for_decoder)

    return audio_out.squeeze(1)  # [batch, samples]


class ProjectionDecoder(nn.Module):
    """Small decoder to go from VQ output (512) back to audio-compatible space."""

    def __init__(self, input_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def apply_deshimmer(audio_np: "np.ndarray", sr: int = 48000) -> "np.ndarray":
    """
    Apply deshimmer post-processing to remove 5-7 kHz shimmer artifacts.

    Requires deshimmer to be installed. Falls back to no-op if unavailable.
    """
    try:
        # Try to import from the deshimmer repo (add to PYTHONPATH or install)
        import sys
        deshimmer_paths = [
            "/Users/johndpope/Documents/GitHub/deshimmer",
            os.path.expanduser("~/deshimmer"),
            "./deshimmer",
        ]
        for p in deshimmer_paths:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        from deshimmer import process, Params

        # Use conservative settings for training validation
        params = Params(
            start_hz=5100.0,
            end_hz=7200.0,
            thr_db=8.0,
            slope=0.6,
            mix=1.0,
        )
        return process(audio_np, sr, params)
    except ImportError:
        return audio_np  # No deshimmer available, return unchanged
    except Exception as e:
        print(f"  [Deshimmer] Warning: {e}")
        return audio_np


def run_validation_detokenize(
    codec, projection, vq_decoder, val_audio_paths: List[str],
    device, output_dir: str, epoch: int,
    num_diffusion_steps: int = 16, num_detokenize_steps: int = 20,
    apply_deshimmer_postproc: bool = False
):
    """
    Run validation by encoding→decoding a few reference clips using real detokenize.

    This gives a much more realistic assessment of quality than just looking at loss numbers.

    Args:
        apply_deshimmer_postproc: If True, also saves a deshimmer-processed version
                                   for A/B comparison (helps diagnose shimmer issues)
    """
    import numpy as np

    projection.eval()
    vq_decoder.eval()

    val_output_dir = os.path.join(output_dir, "validation")
    os.makedirs(val_output_dir, exist_ok=True)

    results = []

    for i, audio_path in enumerate(val_audio_paths):
        try:
            # Load validation audio
            waveform, sr = torchaudio.load(audio_path)
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(sr, 48000)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Limit duration for validation (30 seconds max)
            max_samples = 30 * 48000
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

            audio = waveform.squeeze(0).to(device)
            duration = audio.shape[-1] / 48000

            with torch.no_grad():
                # Encode
                x = audio.unsqueeze(0).unsqueeze(1)
                for layer in codec.scalar_model.encoder:
                    x = layer(x)
                latent_128 = x.permute(0, 2, 1)

                # Project
                latent_512 = projection(latent_128)

                # VQ
                vq = codec.flow_matching.vq_embed
                x_proj = vq.project_in(latent_512)

                quantized_sum = torch.zeros_like(x_proj)
                residual = x_proj
                all_indices = []

                for layer in vq.layers:
                    codebook = layer._codebook.embed.squeeze(0)
                    distances = torch.cdist(residual, codebook.unsqueeze(0).expand(residual.shape[0], -1, -1))
                    indices = distances.argmin(dim=-1)
                    all_indices.append(indices)
                    quantized = codebook[indices.view(-1)].view_as(residual)
                    quantized_sum = quantized_sum + quantized
                    residual = residual - quantized

                # Stack tokens: [batch, num_layers, frames]
                tokens = torch.stack(all_indices, dim=1)

                # Detokenize using official method with more steps for better quality
                audio_out = codec.detokenize(
                    tokens,
                    duration=duration,
                    num_steps=num_detokenize_steps,
                    guidance_scale=1.0
                )

            # Save outputs
            base_name = Path(audio_path).stem

            # Save original (first time only)
            orig_path = os.path.join(val_output_dir, f"{base_name}_original.wav")
            if not os.path.exists(orig_path):
                torchaudio.save(orig_path, waveform.cpu(), 48000)

            # Save reconstruction
            recon_path = os.path.join(val_output_dir, f"{base_name}_epoch{epoch:04d}.wav")
            torchaudio.save(recon_path, audio_out.cpu(), 48000)

            result_entry = {
                "file": base_name,
                "duration": duration,
                "tokens_shape": tokens.shape,
                "output_path": recon_path
            }

            # Optionally apply deshimmer and save for A/B comparison
            if apply_deshimmer_postproc:
                audio_np = audio_out.cpu().numpy()
                if audio_np.ndim == 1:
                    audio_np = audio_np[np.newaxis, :]
                # deshimmer expects (samples, channels)
                audio_np_t = audio_np.T if audio_np.shape[0] < audio_np.shape[1] else audio_np

                deshimmered = apply_deshimmer(audio_np_t, sr=48000)

                # Convert back to tensor format
                if deshimmered.ndim == 1:
                    deshimmered = deshimmered[np.newaxis, :]
                elif deshimmered.shape[1] > deshimmered.shape[0]:
                    pass  # Already (channels, samples)
                else:
                    deshimmered = deshimmered.T

                deshimmer_path = os.path.join(val_output_dir, f"{base_name}_epoch{epoch:04d}_deshimmered.wav")
                torchaudio.save(deshimmer_path, torch.from_numpy(deshimmered).float(), 48000)
                result_entry["deshimmer_path"] = deshimmer_path
                print(f"  [Val {i+1}] {base_name}: saved {recon_path} + deshimmered")
            else:
                print(f"  [Val {i+1}] {base_name}: saved {recon_path}")

            results.append(result_entry)

        except Exception as e:
            print(f"  [Val {i+1}] Error processing {audio_path}: {e}")

    projection.train()
    vq_decoder.train()

    return results


def create_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
    """
    Create a proper warmup + cosine decay scheduler.

    Args:
        warmup_steps: Number of steps for linear warmup
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial LR (default 0.01 = 1%)
    """
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6 / optimizer.param_groups[0]['lr'],  # Start from near-zero
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=optimizer.param_groups[0]['lr'] * min_lr_ratio
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    return scheduler


def check_gradients(model, name="model"):
    """Check for NaN/Inf gradients and return stats."""
    total_norm = 0.0
    has_nan = False
    has_inf = False

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            if torch.isnan(p.grad).any():
                has_nan = True
            if torch.isinf(p.grad).any():
                has_inf = True

    total_norm = total_norm ** 0.5
    return total_norm, has_nan, has_inf


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}")
    print(f"Warmup steps: {args.warmup_steps}, Grad clip: {args.grad_clip}")
    print(f"High-freq weight: {args.highfreq_weight}, Phase loss: {args.include_phase}")

    use_amp = device.type == "cuda" and args.use_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"AMP enabled: {use_amp}")

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
        projection_type=args.projection_type,
    ).to(device)

    # Create small decoder for VQ output → ScalarModel decoder input
    vq_decoder = ProjectionDecoder(input_dim=512, output_dim=128).to(device)

    total_params = sum(p.numel() for p in projection.parameters()) + sum(p.numel() for p in vq_decoder.parameters())
    print(f"Trainable params: {total_params:,}")
    print(f"Projection type: {args.projection_type}")

    # Dataset and dataloader
    dataset = AudioDataset(args.data_dir, max_duration=args.max_duration)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    # Optimizer for both projection and vq_decoder
    params = list(projection.parameters()) + list(vq_decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999))

    # Calculate total steps and create scheduler
    total_steps = args.epochs * len(dataloader)
    warmup_steps = min(args.warmup_steps, total_steps // 10)  # Cap warmup at 10% of training
    scheduler = create_scheduler(optimizer, warmup_steps, total_steps)
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    # Loss functions
    stft_loss_fn = MultiScaleSTFTLoss(
        include_phase=args.include_phase,
        highfreq_weight=args.highfreq_weight
    ).to(device)
    mel_loss_fn = MelSpectrogramLoss().to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')
    global_step = 0
    nan_count = 0
    max_nan_per_epoch = len(dataloader) // 10  # Allow up to 10% NaN steps before warning

    # Validation audio paths
    val_audio_paths = []
    if args.val_dir and os.path.isdir(args.val_dir):
        extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
        for f in os.listdir(args.val_dir):
            if Path(f).suffix.lower() in extensions:
                val_audio_paths.append(os.path.join(args.val_dir, f))
        val_audio_paths = val_audio_paths[:args.num_val_samples]  # Limit validation samples
        print(f"Validation samples: {len(val_audio_paths)}")

    # Resume if checkpoint exists
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        projection.load_state_dict(ckpt["projection_state_dict"])
        if "vq_decoder_state_dict" in ckpt:
            vq_decoder.load_state_dict(ckpt["vq_decoder_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", ckpt.get("loss", float('inf')))
        global_step = ckpt.get("global_step", start_epoch * len(dataloader))
        print(f"Resumed from epoch {start_epoch}, step {global_step}, best_loss={best_loss:.4f}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        projection.train()
        vq_decoder.train()

        total_loss = 0.0
        total_commit = 0.0
        total_recon = 0.0
        total_usage = 0.0
        epoch_nan_count = 0
        valid_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, audio in enumerate(pbar):
            audio = audio.to(device).squeeze(1)  # [batch, samples]

            optimizer.zero_grad()

            # === SAFER AMP: Only wrap forward pass and loss computation ===
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Encode through projection and VQ
                latent_512, latent_512_out, commitment_loss, codebook_usage = encode_with_vq_fast(
                    audio, codec, projection, device
                )

                # Decode: VQ output → small decoder → ScalarModel decoder
                latent_128_reconstructed = vq_decoder(latent_512_out)  # [batch, frames, 128]
                latent_for_decoder = latent_128_reconstructed.permute(0, 2, 1)  # [batch, 128, frames]

                audio_reconstructed = codec.scalar_model.decode(latent_for_decoder)
                audio_reconstructed = audio_reconstructed.squeeze(1)  # [batch, samples]

                # Match lengths
                min_len = min(audio.shape[-1], audio_reconstructed.shape[-1])
                audio_trim = audio[..., :min_len]
                recon_trim = audio_reconstructed[..., :min_len]

                # Losses
                loss_stft = stft_loss_fn(recon_trim, audio_trim)
                loss_mel = mel_loss_fn(recon_trim, audio_trim)

                # Regularization on projection output
                loss_reg = 0.001 * (latent_512 ** 2).mean()

                # Clamp individual losses to prevent NaN
                loss_mel = torch.clamp(loss_mel, 0, 10)

                # Combined loss
                loss = (
                    loss_stft +
                    0.5 * loss_mel +
                    args.commitment_weight * commitment_loss +
                    loss_reg
                )

            # === NaN/Inf detection BEFORE backward ===
            if torch.isnan(loss) or torch.isinf(loss):
                epoch_nan_count += 1
                if epoch_nan_count <= 3:  # Only print first few
                    print(f"\n[Step {global_step}] NaN/Inf loss detected - skipping step")
                optimizer.zero_grad()
                continue

            # === Backward pass OUTSIDE autocast for safety ===
            if use_amp:
                scaler.scale(loss).backward()

                # Unscale before gradient clipping
                scaler.unscale_(optimizer)

                # Check gradients for NaN/Inf
                grad_norm, has_nan, has_inf = check_gradients(projection, "projection")
                grad_norm2, has_nan2, has_inf2 = check_gradients(vq_decoder, "vq_decoder")

                if has_nan or has_nan2 or has_inf or has_inf2:
                    epoch_nan_count += 1
                    if epoch_nan_count <= 3:
                        print(f"\n[Step {global_step}] NaN/Inf gradient detected - skipping step")
                    optimizer.zero_grad()
                    scaler.update()
                    continue

                # Gradient clipping (STRONGER: 0.5 instead of 1.0)
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                # Check gradients
                grad_norm, has_nan, has_inf = check_gradients(projection, "projection")
                grad_norm2, has_nan2, has_inf2 = check_gradients(vq_decoder, "vq_decoder")

                if has_nan or has_nan2 or has_inf or has_inf2:
                    epoch_nan_count += 1
                    optimizer.zero_grad()
                    continue

                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1
            valid_batches += 1

            total_loss += loss.item()
            total_commit += commitment_loss.item()
            total_recon += loss_stft.item()
            total_usage += codebook_usage

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                stft=f"{loss_stft.item():.3f}",
                commit=f"{commitment_loss.item():.3f}",
                usage=f"{codebook_usage:.0f}",
                lr=f"{current_lr:.2e}"
            )

        # Epoch summary
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            avg_commit = total_commit / valid_batches
            avg_recon = total_recon / valid_batches
            avg_usage = total_usage / valid_batches
        else:
            avg_loss = float('inf')
            avg_commit = 0
            avg_recon = 0
            avg_usage = 0

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, stft={avg_recon:.4f}, commit={avg_commit:.4f}, usage={avg_usage:.0f}")

        if epoch_nan_count > 0:
            print(f"  NaN/Inf steps skipped: {epoch_nan_count}/{len(dataloader)}")
            if epoch_nan_count > max_nan_per_epoch:
                print(f"  WARNING: High NaN rate ({epoch_nan_count/len(dataloader)*100:.1f}%) - consider reducing learning rate")

        # === Validation with real detokenize ===
        if val_audio_paths and (epoch + 1) % args.val_interval == 0:
            print(f"\nRunning validation with detokenize (steps={args.val_detokenize_steps})...")
            run_validation_detokenize(
                codec, projection, vq_decoder, val_audio_paths,
                device, args.output_dir, epoch + 1,
                num_diffusion_steps=args.val_diffusion_steps,
                num_detokenize_steps=args.val_detokenize_steps,
                apply_deshimmer_postproc=args.deshimmer
            )

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "projection_state_dict": projection.state_dict(),
                "vq_decoder_state_dict": vq_decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_projection.pt"))
            print(f"Saved best checkpoint (loss={avg_loss:.4f})")

        # Periodic checkpoint (with full state for resume)
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "projection_state_dict": projection.state_dict(),
                "vq_decoder_state_dict": vq_decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt"))
            print(f"Saved periodic checkpoint at epoch {epoch+1}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fast encoder projection training (upgraded)")

    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--val_dir", type=str, default=None, help="Validation data directory (optional)")
    parser.add_argument("--output_dir", type=str, default="./encoder_projection_fast")

    # Model
    parser.add_argument("--codec_name", type=str, default="HeartMuLa/HeartCodec-oss-20260123")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--projection_type", type=str, default="linear", choices=["linear", "conv1d"],
                        help="Projection architecture: 'linear' or 'conv1d' (may capture temporal structure better)")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_duration", type=float, default=5.0, help="Max audio duration in seconds")
    parser.add_argument("--num_workers", type=int, default=4)

    # Loss weights
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    parser.add_argument("--include_phase", action="store_true", help="Include phase loss in STFT loss")
    parser.add_argument("--highfreq_weight", type=float, default=0.0,
                        help="Extra weight on 4-8 kHz shimmer zone (try 0.5-1.0 if reconstructions have shimmer)")

    # Stability
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Linear warmup steps")
    parser.add_argument("--grad_clip", type=float, default=0.5, help="Gradient clipping norm (default: 0.5)")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false", help="Disable AMP")

    # Validation
    parser.add_argument("--val_interval", type=int, default=5, help="Run validation every N epochs")
    parser.add_argument("--val_diffusion_steps", type=int, default=16, help="Diffusion steps during validation")
    parser.add_argument("--val_detokenize_steps", type=int, default=20, help="Detokenize steps during validation")
    parser.add_argument("--num_val_samples", type=int, default=5, help="Number of validation samples")
    parser.add_argument("--deshimmer", action="store_true",
                        help="Apply deshimmer post-processing during validation (saves A/B files for comparison)")

    # Checkpointing
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

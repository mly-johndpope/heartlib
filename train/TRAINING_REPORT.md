# HeartCodec Encoder Projection Training Report

## Executive Summary

The HeartCodec public release is **missing the encoder projection layer** (128→512 dimensions) required for audio tokenization. This report documents everything needed to train this missing component.

---

## 1. The Problem

### What's Missing

```
ENCODE PATH (Audio → Tokens):

audio [48kHz]
    ↓
ScalarModel.encode() → 128-dim latents ✓ (works)
    ↓
??? MISSING: 128→512 projection ???  ← THIS IS WHAT WE TRAIN
    ↓
project_in (512→32) → VQ quantize → tokens [8, frames]
```

The decode path works perfectly. The encode path fails because HeartMuLa didn't release the learned projection that maps ScalarModel's 128-dim output to the VQ's expected 512-dim input.

### Why It Matters

Without this projection, you cannot:
- Tokenize your own audio for fine-tuning HeartMuLa
- Create audio-to-audio pipelines
- Build real-time encoding applications

---

## 2. Architecture Overview

### HeartCodec Components

| Component | Role | Dimensions |
|-----------|------|------------|
| **ScalarModel** | Waveform encoder/decoder | audio ↔ 128-dim @ 25fps |
| **Projection** | Dimension bridge (MISSING) | 128 → 512 |
| **ResidualVQ** | Discrete tokenization | 512 → 8 codebooks × 8192 vocab |
| **FlowMatching** | Latent diffusion | VQ cond → smooth latents |

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLETE PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ENCODE:  Audio → ScalarModel → [PROJECTION] → VQ → Tokens  │
│                                      ↑                      │
│                              WE TRAIN THIS                  │
│                                                             │
│  DECODE:  Tokens → VQ → FlowMatching → ScalarModel → Audio  │
│                         (works fine)                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Training Requirements

### Hardware

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU VRAM | 8 GB | 16+ GB |
| GPU Type | RTX 3070 / V100 | A100 / RTX 4090 |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB | 50+ GB (for datasets) |

### Software Dependencies

```bash
# Core
torch==2.4.1
torchaudio==2.4.1
transformers==4.57.0
vector-quantize-pytorch==1.27.15

# Training
accelerate==1.12.0
einops==0.8.1

# Optional (validation)
scipy
soundfile
```

### Dataset Requirements

| Aspect | Requirement |
|--------|-------------|
| **Format** | MP3, WAV, FLAC, OGG, M4A |
| **Sample Rate** | 48 kHz (auto-resampled) |
| **Channels** | Mono (auto-converted from stereo) |
| **Duration** | 1-60 seconds per file |
| **Minimum Size** | 100+ files |
| **Recommended** | 400-800 hours diverse music |
| **Diversity** | Multiple genres, vocals, instruments |

---

## 4. Training Script: `train_encoder_projection_fast.py`

### Why "Fast"?

The fast version **bypasses FlowMatching during training**:
- FlowMatching requires ~20 ODE steps per sample (expensive)
- Training focuses on encoder→VQ→decoder path only
- FlowMatching is still used during validation/inference
- **Result: 10-50x faster training**

### Key Features

| Feature | Description |
|---------|-------------|
| Gradient clipping | 0.5 (stronger than typical 1.0) |
| LR schedule | Linear warmup → Cosine decay |
| NaN handling | Detects NaN in loss AND gradients, skips bad steps |
| Validation | Real detokenize every N epochs with .wav output |
| Checkpointing | Full state: model, optimizer, scheduler, global_step |
| AMP | Mixed precision with safe backward pass |
| Deshimmer | Optional A/B comparison for shimmer artifacts |

### CLI Arguments

```bash
python train/train_encoder_projection_fast.py \
  # Data
  --data_dir /path/to/audio \
  --val_dir /path/to/validation \
  --output_dir ./projection_output \

  # Model
  --codec_name HeartMuLa/HeartCodec-oss-20260123 \
  --hidden_dim 256 \
  --projection_type linear \    # or "conv1d"

  # Training
  --batch_size 8 \
  --epochs 100 \
  --lr 1e-3 \
  --max_duration 5.0 \

  # Loss weights
  --commitment_weight 0.25 \
  --include_phase \             # Add phase loss
  --highfreq_weight 0.5 \       # Extra weight on 4-8kHz shimmer zone

  # Stability
  --warmup_steps 2000 \
  --grad_clip 0.5 \

  # Validation
  --val_interval 5 \
  --val_detokenize_steps 20 \
  --deshimmer \                 # Save deshimmered A/B comparison

  # Resume
  --resume ./projection_output/checkpoint_epoch50.pt
```

### Loss Functions

1. **Multi-Scale STFT Loss** (primary)
   - Scales: n_fft = [512, 1024, 2048]
   - Log magnitude L1 + spectral convergence
   - Optional phase loss (weight 0.1)
   - Optional high-freq emphasis (4-8 kHz)

2. **Mel-Spectrogram Loss** (weight 0.5)
   - 80 mel bins @ 48kHz
   - Log-magnitude matching

3. **Commitment Loss** (weight 0.25)
   - Pulls encoder output toward VQ codebook
   - Essential for stable quantization

4. **L2 Regularization** (weight 0.001)
   - Prevents projection output explosion

---

## 5. Step-by-Step Training Guide

### Step 1: Prepare Dataset

```bash
# Create dataset directory
mkdir -p ~/heartcodec_training/audio
mkdir -p ~/heartcodec_training/validation

# Copy training audio (100+ files)
cp /path/to/music/*.mp3 ~/heartcodec_training/audio/

# Copy 5-8 validation files (different genres)
cp song1.wav song2.wav song3.wav ~/heartcodec_training/validation/
```

### Step 2: Install Dependencies

```bash
cd /Users/johndpope/Documents/GitHub/heartlib
pip install -e .

# Or manually:
pip install torch==2.4.1 torchaudio==2.4.1 transformers vector-quantize-pytorch einops
```

### Step 3: Run Training

```bash
python train/train_encoder_projection_fast.py \
  --data_dir ~/heartcodec_training/audio \
  --val_dir ~/heartcodec_training/validation \
  --output_dir ~/heartcodec_training/output \
  --epochs 100 \
  --batch_size 4 \
  --highfreq_weight 0.5 \
  --deshimmer \
  --val_interval 5
```

### Step 4: Monitor Progress

```bash
# Watch training
tail -f ~/heartcodec_training/output/training.log

# Check validation outputs
ls ~/heartcodec_training/output/validation/
# song1_original.wav
# song1_epoch0005.wav
# song1_epoch0005_deshimmered.wav
# ...
```

### Step 5: Evaluate Quality

Listen to validation outputs:
- Compare `*_original.wav` vs `*_epochXXXX.wav`
- If `*_deshimmered.wav` sounds much better → increase `--highfreq_weight`
- Good projection: raw reconstruction ≈ deshimmered version

### Step 6: Use Trained Projection

```python
import torch
from heartlib import HeartCodec

# Load codec
codec = HeartCodec.from_pretrained("HeartMuLa/HeartCodec-oss-20260123")

# Load trained projection
ckpt = torch.load("~/heartcodec_training/output/best_projection.pt")
projection = EncoderProjection(128, 512, 256)
projection.load_state_dict(ckpt["projection_state_dict"])

# Now you can encode!
audio = ...  # [batch, samples] @ 48kHz
tokens = encode_with_projection(audio, codec, projection)
```

---

## 6. Common Issues & Solutions

### Issue: NaN Loss

**Symptoms**: Training crashes with NaN, loss explodes

**Solutions**:
```bash
# Reduce learning rate
--lr 5e-4

# Stronger gradient clipping
--grad_clip 0.25

# Disable AMP if persistent
--no_amp
```

### Issue: High Shimmer Artifacts (5-7 kHz)

**Symptoms**: Reconstructions sound "crystalline" or "glassy"

**Solutions**:
```bash
# Add high-frequency loss emphasis
--highfreq_weight 1.0

# Add phase loss
--include_phase

# Use deshimmer for comparison
--deshimmer
```

### Issue: Poor Codebook Usage

**Symptoms**: `usage` metric stays low (<100)

**Solutions**:
```bash
# Increase commitment weight
--commitment_weight 0.5

# Try conv1d projection (better temporal structure)
--projection_type conv1d
```

### Issue: Out of Memory

**Solutions**:
```bash
# Reduce batch size
--batch_size 2

# Reduce audio duration
--max_duration 3.0

# Disable AMP (uses more compute but less peak memory)
--no_amp
```

---

## 7. Quality Benchmarks

### Expected Loss Curves

| Epoch | STFT Loss | Commitment | Codebook Usage |
|-------|-----------|------------|----------------|
| 1 | 5.0-8.0 | 0.5-1.0 | 50-200 |
| 10 | 2.0-4.0 | 0.2-0.5 | 200-500 |
| 50 | 1.0-2.0 | 0.1-0.3 | 500-2000 |
| 100 | 0.5-1.5 | 0.05-0.2 | 1000-4000 |

### Subjective Quality Levels

| Quality | STFT Loss | Description |
|---------|-----------|-------------|
| Poor | >3.0 | Obvious artifacts, buzzy, metallic |
| Fair | 2.0-3.0 | Recognizable but noticeable degradation |
| Good | 1.0-2.0 | Minor artifacts, mostly clean |
| Excellent | <1.0 | Near-transparent, hard to distinguish |

---

## 8. Alternative Training Scripts

| Script | Use Case | Pros | Cons |
|--------|----------|------|------|
| `train_encoder_projection_fast.py` | **Recommended** | Fast, stable, full features | - |
| `train_encoder_projection_v2.py` | Experimental | Gumbel-Softmax, code revival | Complex, slower |
| `train_encoder_projection_v2_lite.py` | Low VRAM | Memory efficient | Fewer features |
| `optimize_encode.py` | Single-file | Highest quality per-file | Very slow |

---

## 9. Post-Training: Tokenization

Once trained, use the projection to tokenize audio:

```bash
python train/tokenize_with_projection.py \
  --audio_path input.wav \
  --projection_path ~/heartcodec_training/output/best_projection.pt \
  --output_path tokens.npy
```

Or in Python:

```python
import torch
import torchaudio
from train.train_encoder_projection_fast import EncoderProjection, encode_with_vq_fast

# Load audio
audio, sr = torchaudio.load("input.wav")
if sr != 48000:
    audio = torchaudio.transforms.Resample(sr, 48000)(audio)
audio = audio.mean(0)  # mono

# Load models
codec = HeartCodec.from_pretrained("HeartMuLa/HeartCodec-oss-20260123")
projection = EncoderProjection(128, 512, 256)
projection.load_state_dict(torch.load("best_projection.pt")["projection_state_dict"])

# Encode
with torch.no_grad():
    latent_512, latent_512_out, _, _ = encode_with_vq_fast(audio.unsqueeze(0), codec, projection, "cuda")
    # Get tokens from VQ...
```

---

## 10. Resources

### Official Links
- HeartMuLa HuggingFace: https://huggingface.co/HeartMuLa
- HeartCodec Model: `HeartMuLa/HeartCodec-oss-20260123`
- HeartMuLa-3B: `HeartMuLa/HeartMuLa-oss-3B`

### Community Tools
- Deshimmer (shimmer removal): https://github.com/TheApeMachine/deshimmer
- SimpleTuner (LoRA training): Available via pip

### This Repository
- Training script: `train/train_encoder_projection_fast.py`
- Fork: https://github.com/mly-johndpope/heartlib

---

## Appendix A: Full Architecture Details

### ScalarModel (Conv1D Codec)

```
Encoder:
  Conv1D(1, 128) → PreProcessor →
  ResEncoderBlock(128, 256, stride=4) →
  ResEncoderBlock(256, 512, stride=4) →
  ResEncoderBlock(512, 512, stride=4) →
  ResEncoderBlock(512, 512, stride=5) →
  ResEncoderBlock(512, 128, stride=4) →
  Conv1D(128, 128)

  Total downsampling: 4×4×4×5×4 = 1280
  48000 Hz / 1280 ≈ 37.5 fps → interpolated to 25 fps

Decoder: (mirror of encoder with transposed convs)
```

### ResidualVQ Configuration

```python
ResidualVQ(
    num_codebooks=8,
    codebook_size=8192,
    codebook_dim=32,
    input_dim=512,  # After projection
    # Uses commitment loss, EMA updates
)
```

### FlowMatching (Diffusion)

```python
FlowMatching(
    estimator=LlamaTransformer(
        layers=24,
        heads=24,
        dim=1536,
    ),
    # ODE solver: Euler method
    # Default steps: 8 (training), 20 (inference)
    # Supports CFG with guidance_scale
)
```

---

## Appendix B: Troubleshooting Checklist

- [ ] GPU has enough VRAM (8GB+)
- [ ] Audio files are 48kHz or will be resampled
- [ ] Dataset has 100+ diverse files
- [ ] Validation directory has 5-8 reference clips
- [ ] Using latest `train_encoder_projection_fast.py`
- [ ] Checkpoint directory is writable
- [ ] Monitoring validation outputs every 5 epochs
- [ ] Comparing raw vs deshimmered if shimmer present

---

*Report generated: 2026-02-27*
*Training script version: be6ac5a*

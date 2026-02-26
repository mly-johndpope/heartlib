# HeartCodec Tokenization Guide

## The Problem

HeartCodec is a neural audio codec that uses:
1. **ScalarModel**: Conv1d encoder/decoder for waveform ↔ 128-dim latents
2. **FlowMatching**: Diffusion-based generative model for latent ↔ VQ conditioning
3. **ResidualVQ**: 8 codebooks, each with 8192 entries of 32-dim embeddings

The **decode path** (tokens → audio) is fully available:
```
tokens [8, frames]
  → VQ decode to 32-dim
  → project_out to 512-dim
  → FlowMatching diffusion → 256-dim latents
  → reshape to 128-dim (stereo split)
  → ScalarModel decode → audio
```

The **encode path** (audio → tokens) is **incomplete**:
```
audio
  → ScalarModel encode → 128-dim
  → ??? (missing 128→512 projection)
  → project_in to 32-dim
  → Residual VQ quantize → tokens
```

The missing 128→512 (or 256→512) projection layer is not included in the public release.

## Available Approaches

### 1. Optimization-Based Encoding (Recommended for Quality)

File: `optimize_encode.py`

Uses gradient descent to find tokens that decode to the target audio:
- Initialize soft VQ embeddings
- Decode through FlowMatching → ScalarModel
- Compute loss vs target audio
- Backpropagate and update embeddings
- Quantize to discrete tokens

**Pros**: Can find tokens that accurately represent the audio
**Cons**: Slow (~minutes per audio clip), requires GPU

```bash
python train/optimize_encode.py \
    --input /path/to/audio.wav \
    --output_dir /path/to/tokens \
    --num_steps 300 \
    --verify
```

### 2. Simple Tiled Encoding (Fast but Poor Quality)

File: `tokenize_audio_simple.py`

Uses ScalarModel encoder + tiling (128→512) for quick tokenization:
- Fast (seconds per audio clip)
- Poor reconstruction quality (correlation ~0)
- May be useful for data augmentation or quick experiments

```bash
python train/tokenize_audio_simple.py \
    --input_dir /path/to/audio_dir \
    --output_dir /path/to/tokens \
    --max_duration 60
```

### 3. Direct ScalarModel Encoding (Alternative)

Skip the VQ entirely and work with ScalarModel latents directly.
This would require modifying HeartMuLa to use continuous latents instead of discrete tokens.

## Decoding Tokens

File: `decode_tokens.py`

Decodes tokens back to audio using manual decode path (official `detokenize` has a bug):

```bash
python train/decode_tokens.py \
    --input /path/to/tokens.npy \
    --output_dir /path/to/audio \
    --num_steps 20
```

**Note**: Decoded audio is 2x the original duration due to FlowMatching's 2x upsampling.

## Token Format

- Shape: `[8, num_frames]` - 8 codebooks, variable frames
- Frame rate: ~25 fps (from ScalarModel encoder)
- Vocab size: 8192 per codebook
- Duration: `num_frames / 25` seconds (before decode 2x expansion)

For HeartMuLa training, tokens need to be formatted as:
- `[batch, frames, num_codebooks + 1]` where last position is text token
- Use special tokens for padding/masking as needed

## Recommended Approach: Train Encoder Projection

File: `train_encoder_projection.py`

Based on Grok's recommendations, train a learnable 128→512 projection network:

```bash
python train/train_encoder_projection.py \
    --data_dir /path/to/audio \
    --output_dir ./encoder_projection \
    --projection_type mlp \
    --hidden_dim 256 \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --init_pinv  # Initialize from pseudo-inverse of project_out
```

**Training Process:**
1. Audio → ScalarModel.encode → 128-dim latents
2. **Learned Projection** → 512-dim (trainable)
3. project_in → 32-dim → Residual VQ → tokens
4. Full decode path → reconstructed audio
5. Multi-scale STFT loss to match original

**Requirements:**
- GPU with 8GB+ VRAM
- Audio dataset (LibriSpeech, FMA, or custom)
- ~50 epochs, ~1 hour on GPU

## Recommendations

1. **Best quality**: Train encoder projection (train_encoder_projection.py)
2. **Fast but slow per-sample**: Optimization-based encoding (GPU required)
3. **Quick experiments**: Simple tiled encoding (poor quality)
4. **Contact HeartMuLa team**: Ask for pre-tokenized data or encoder weights

## Files

- `train_encoder_projection.py` - **NEW**: Train missing 128→512 projection
- `optimize_encode.py` - Optimization-based encoding (slow, high quality)
- `tokenize_audio_simple.py` - Simple tiled encoding (fast, poor quality)
- `decode_tokens.py` - Decode tokens to audio
- `tokenize_audio_proper.py` - Original attempt (uses tiled approach)

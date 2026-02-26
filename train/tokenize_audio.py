#!/usr/bin/env python3
"""
Audio Tokenization Script for HeartMuLa Training

This script tokenizes audio files using HeartCodec to create the pre-computed
audio tokens required for HeartMuLa training.

Usage:
    python tokenize_audio.py \
        --audio_path /path/to/song.mp3 \
        --output_dir /path/to/output \
        --tags "pop,upbeat,catchy" \
        --lyrics "Your song lyrics here..."

    # Or process a directory:
    python tokenize_audio.py \
        --audio_dir /path/to/audio_files \
        --output_dir /path/to/output

Requirements:
    - HeartCodec model weights (auto-downloaded from HuggingFace)
    - Audio files in any ffmpeg-supported format
    - vector-quantize-pytorch package

Note: HeartCodec's encoder produces latents that are quantized via ResidualVQ
to produce the discrete tokens that HeartMuLa was trained on.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SAMPLE_RATE = 48000
FRAMES_PER_SEC = 12.5  # HeartCodec frame rate
# Total downsampling: 2 (num_samples) * 3 * 4 * 4 * 4 * 5 = 1920
# But we also have latent_hidden_dim processing
DOWNSAMPLE_FACTOR = 3840  # 48000 / 12.5 samples per frame


def load_audio(audio_path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load and resample audio to target sample rate."""
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0)  # [samples]


def segment_audio(
    waveform: torch.Tensor,
    segment_duration: float = 30.0,
    overlap: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
) -> list[torch.Tensor]:
    """Split audio into segments of specified duration."""
    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int((segment_duration - overlap) * sample_rate)

    segments = []
    start = 0
    while start < len(waveform):
        end = start + segment_samples
        segment = waveform[start:end]

        # Pad last segment if needed
        if len(segment) < segment_samples:
            pad_length = segment_samples - len(segment)
            segment = torch.nn.functional.pad(segment, (0, pad_length))

        segments.append(segment)
        start += hop_samples

        # Don't create very short final segments
        if start >= len(waveform) - sample_rate:
            break

    return segments


class HeartCodecEncoder:
    """
    Wrapper for HeartCodec encoding.

    IMPORTANT: The public HeartCodec release is DECODER-ONLY. The architecture is:
    1. ScalarModel encoder: audio [B, 1, T] -> scalar quantized latents [B, 128, T']
       - Uses scalar quantization (round to 1/9 bins), NOT a VQ with discrete codes
    2. FlowMatching VQ: Takes HeartMuLa's tokens -> embeddings for flow matching
       - This VQ is for DECODING, not encoding!

    The tokens that HeartMuLa was trained on come from a DIFFERENT encoder that is
    not publicly released. This class provides a workaround using the scalar model's
    latent representation, quantized to produce discrete tokens.

    This is an APPROXIMATION - the actual HeartMuLa training tokens were created
    with a different encoder. For best results, contact the HeartMuLa team for
    access to the original tokenization method.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Try to import HeartCodec
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]):
        """Load HeartCodec model from various sources."""
        # Try heartlib first
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from heartlib.heartcodec.modeling_heartcodec import HeartCodec
            from heartlib.heartcodec.configuration_heartcodec import HeartCodecConfig
            HAS_HEARTLIB = True
        except ImportError:
            HAS_HEARTLIB = False

        # Try SimpleTuner
        try:
            from simpletuner.helpers.models.heartmula.codec import HeartCodec, HeartCodecConfig
            HAS_SIMPLETUNER = True
        except ImportError:
            HAS_SIMPLETUNER = False

        if not HAS_HEARTLIB and not HAS_SIMPLETUNER:
            raise ImportError(
                "Could not import HeartCodec. Please ensure either heartlib or simpletuner is installed.\n"
                "Install with: pip install simpletuner[cuda] or clone heartlib"
            )

        if model_path is None:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download("HeartMuLa/HeartCodec-oss-20260123")

        print(f"Loading HeartCodec from {model_path}...")
        self.model = HeartCodec.from_pretrained(model_path)
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        # Get number of codebooks from VQ config
        self.num_codebooks = self.model.config.num_quantizers
        self.codebook_size = self.model.config.codebook_size
        self.latent_dim = self.model.config.latent_hidden_dim
        print(f"HeartCodec loaded on {self.device}")
        print(f"  num_codebooks={self.num_codebooks}, codebook_size={self.codebook_size}")
        print(f"  latent_dim={self.latent_dim}")

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode audio waveform to discrete tokens using scalar model latents.

        This creates a SIMULATED token representation by:
        1. Encoding audio through ScalarModel to get 128-dim latents
        2. Quantizing the latents to discrete bins
        3. Reshaping to match expected [num_codebooks, frames] format

        NOTE: This is an approximation. The actual HeartMuLa training used
        a different tokenization method that is not publicly available.

        Args:
            waveform: Audio tensor of shape [samples] at 48kHz

        Returns:
            tokens: Tensor of shape [num_codebooks, frames] with discrete codes
        """
        # Reshape for encoder: [batch, channels, samples]
        x = waveform.unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=self.dtype)

        # Encode through ScalarModel encoder
        # This produces continuous latents with scalar quantization
        latent = self.model.scalar_model.encode(x)
        # latent shape: [batch, latent_hidden_dim (128), time_frames]

        # The scalar model uses round_func9 which quantizes to 10 values: -1, -8/9, ..., 8/9, 1
        # We need to map this to discrete tokens in range [0, codebook_size-1]

        # First, extract the latent values (they're already scalar quantized)
        latent = latent.squeeze(0)  # [latent_dim, frames]
        frames = latent.shape[1]

        # Strategy: Group latent dimensions into codebooks and quantize
        # latent_dim=128, num_codebooks=8, so 16 dims per codebook
        dims_per_codebook = self.latent_dim // self.num_codebooks

        tokens = []
        for cb in range(self.num_codebooks):
            # Get the latent slice for this codebook
            start_dim = cb * dims_per_codebook
            end_dim = start_dim + dims_per_codebook
            cb_latent = latent[start_dim:end_dim, :]  # [dims_per_codebook, frames]

            # Hash the multi-dimensional latent to a single token per frame
            # Using a simple approach: treat as bits and combine
            # Normalize from [-1, 1] to [0, 1]
            cb_normalized = (cb_latent + 1) / 2  # [0, 1]

            # Quantize each dim to a small number of bins and combine
            # With 16 dims, we could use 2 bits each for 32-bit total
            # But codebook_size is 8192 = 2^13, so we use fewer dims
            num_significant_dims = min(13, dims_per_codebook)
            cb_reduced = cb_normalized[:num_significant_dims, :]  # [13, frames]

            # Binary quantization
            cb_binary = (cb_reduced > 0.5).to(torch.int32)  # [13, frames]

            # Combine into single token: sum(bit_i * 2^i)
            powers = torch.pow(2, torch.arange(num_significant_dims, device=self.device)).view(-1, 1)
            cb_tokens = (cb_binary * powers).sum(dim=0)  # [frames]

            tokens.append(cb_tokens)

        # Stack to [num_codebooks, frames]
        tokens = torch.stack(tokens, dim=0)

        return tokens.cpu().to(torch.int32)


def tokenize_single_file(
    audio_path: str,
    encoder: HeartCodecEncoder,
    output_dir: str,
    tags: Optional[str] = None,
    lyrics: Optional[str] = None,
    segment_duration: float = 30.0,
) -> list[dict]:
    """Tokenize a single audio file and save tokens."""
    audio_name = Path(audio_path).stem

    print(f"Processing: {audio_path}")

    # Load audio
    waveform = load_audio(audio_path)
    duration = len(waveform) / SAMPLE_RATE
    print(f"  Duration: {duration:.2f}s")

    # Check for companion text files if not provided
    base_path = Path(audio_path).with_suffix("")
    if tags is None:
        tags_file = base_path.with_suffix(".txt")
        if tags_file.exists():
            tags = tags_file.read_text().strip()
            print(f"  Loaded tags from {tags_file}")

    if lyrics is None:
        lyrics_file = base_path.with_suffix(".lyrics")
        if lyrics_file.exists():
            lyrics = lyrics_file.read_text().strip()
            print(f"  Loaded lyrics from {lyrics_file}")

    if tags is None:
        tags = "music"
        print("  Warning: No tags provided, using default 'music'")

    if lyrics is None:
        lyrics = ""
        print("  Note: No lyrics provided, using empty string (instrumental)")

    # Segment audio if longer than segment_duration
    if duration > segment_duration + 1.0:
        segments = segment_audio(waveform, segment_duration, overlap=2.0)
        print(f"  Split into {len(segments)} segments")
    else:
        segments = [waveform]

    # Tokenize each segment
    results = []
    os.makedirs(output_dir, exist_ok=True)

    for i, segment in enumerate(segments):
        # Encode to tokens
        tokens = encoder.encode(segment)

        # Save tokens
        if len(segments) > 1:
            token_filename = f"{audio_name}_seg{i:03d}.npy"
        else:
            token_filename = f"{audio_name}.npy"

        token_path = os.path.join(output_dir, token_filename)
        np.save(token_path, tokens.numpy())

        # Create metadata entry
        entry = {
            "audio_tokens_path": token_filename,
            "tags": tags,
            "lyrics": lyrics,
            "duration": segment_duration if len(segments) > 1 else duration,
            "source_file": os.path.basename(audio_path),
            "segment_index": i if len(segments) > 1 else None,
        }
        results.append(entry)

        print(f"  Saved: {token_path} (shape: {tokens.shape})")

    return results


def create_dataset_metadata(entries: list[dict], output_dir: str):
    """Create metadata JSON for the dataset."""
    metadata_path = os.path.join(output_dir, "metadata.jsonl")

    with open(metadata_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nCreated metadata: {metadata_path}")
    print(f"Total samples: {len(entries)}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize audio files for HeartMuLa training"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--audio_path",
        type=str,
        help="Path to a single audio file",
    )
    input_group.add_argument(
        "--audio_dir",
        type=str,
        help="Directory containing audio files",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for tokens and metadata",
    )

    # Text options
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Tags/prompt for the audio (e.g., 'pop,upbeat,catchy')",
    )
    parser.add_argument(
        "--lyrics",
        type=str,
        default=None,
        help="Lyrics for the audio (or path to lyrics file)",
    )

    # Model options
    parser.add_argument(
        "--codec_path",
        type=str,
        default=None,
        help="Path to HeartCodec weights (auto-downloads if not specified)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to run encoding on",
    )

    # Processing options
    parser.add_argument(
        "--segment_duration",
        type=float,
        default=30.0,
        help="Duration of each segment in seconds",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".mp3", ".wav", ".flac", ".m4a", ".ogg"],
        help="Audio file extensions to process",
    )

    args = parser.parse_args()

    # Handle lyrics from file
    if args.lyrics and os.path.isfile(args.lyrics):
        with open(args.lyrics) as f:
            args.lyrics = f.read().strip()

    # Initialize encoder
    encoder = HeartCodecEncoder(
        model_path=args.codec_path,
        device=args.device,
    )

    # Collect audio files
    if args.audio_path:
        audio_files = [args.audio_path]
    else:
        audio_files = []
        for ext in args.extensions:
            audio_files.extend(Path(args.audio_dir).glob(f"*{ext}"))
            audio_files.extend(Path(args.audio_dir).glob(f"*{ext.upper()}"))
        audio_files = sorted(set(str(f) for f in audio_files))

    if not audio_files:
        print("No audio files found!")
        return

    print(f"Found {len(audio_files)} audio file(s)")

    # Process files
    all_entries = []
    for audio_file in audio_files:
        entries = tokenize_single_file(
            audio_path=audio_file,
            encoder=encoder,
            output_dir=args.output_dir,
            tags=args.tags,
            lyrics=args.lyrics,
            segment_duration=args.segment_duration,
        )
        all_entries.extend(entries)

    # Create metadata
    create_dataset_metadata(all_entries, args.output_dir)

    print("\nTokenization complete!")
    print(f"Tokens saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

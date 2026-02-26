#!/bin/bash
# Run HeartMuLa training on GPU machine
# This script should be run on the machine with the 3090 GPU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "HeartMuLa Training Setup on GPU Machine"
echo "=========================================="
echo "Repo root: $REPO_ROOT"
echo "Script dir: $SCRIPT_DIR"

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || echo "No NVIDIA GPU found"

# Check/install dependencies
echo ""
echo "Checking dependencies..."

if ! python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo "Installing PyTorch with CUDA..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

if ! python -c "import simpletuner" 2>/dev/null; then
    echo "Installing SimpleTuner..."
    pip install 'simpletuner[cuda]'
fi

if ! python -c "import vector_quantize_pytorch" 2>/dev/null; then
    echo "Installing vector-quantize-pytorch..."
    pip install vector-quantize-pytorch
fi

# Check if tokens already exist
DATASET_DIR="$REPO_ROOT/train/datasets/max_richter"

if [ -f "$DATASET_DIR/metadata.jsonl" ]; then
    echo ""
    echo "Tokens already exist at $DATASET_DIR"
    echo "Found $(wc -l < "$DATASET_DIR/metadata.jsonl") samples"
else
    # Tokenize the audio if needed
    AUDIO_FILE="$DATASET_DIR/on_the_nature_of_daylight.mp3"
    if [ -f "$AUDIO_FILE" ]; then
        echo ""
        echo "Tokenizing audio: $AUDIO_FILE"
        python "$SCRIPT_DIR/tokenize_audio.py" \
            --audio_path "$AUDIO_FILE" \
            --output_dir "$DATASET_DIR" \
            --device cuda
    else
        echo "ERROR: Audio file not found: $AUDIO_FILE"
        echo "Please add an audio file to tokenize."
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Dataset: $DATASET_DIR"
echo ""
echo "To start training, run from repo root ($REPO_ROOT):"
echo ""
echo "  simpletuner train --config train/overfit_max_richter/config.json"
echo ""
echo ""
echo "IMPORTANT: The tokenization uses an APPROXIMATION since the original"
echo "HeartMuLa training tokenizer is not publicly released. Results may differ"
echo "from the original HeartMuLa training quality."
echo ""

# Ask if user wants to start training
read -p "Start training now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting training..."
    cd "$REPO_ROOT"
    simpletuner train --config train/overfit_max_richter/config.json
fi

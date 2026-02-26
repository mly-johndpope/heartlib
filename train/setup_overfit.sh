#!/bin/bash
# HeartMuLa Overfitting Setup Script
# This script sets up everything needed to overfit HeartMuLa on a single song

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}HeartMuLa Overfitting Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${YELLOW}Usage: $0 <audio_file> [tags] [lyrics_file]${NC}"
    echo ""
    echo "Arguments:"
    echo "  audio_file    - Path to your audio file (mp3, wav, flac, etc.)"
    echo "  tags          - Optional comma-separated tags (default: 'pop,music')"
    echo "  lyrics_file   - Optional path to lyrics text file"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/song.mp3 'pop,catchy,upbeat,2024' /path/to/lyrics.txt"
    exit 1
fi

AUDIO_FILE="$1"
TAGS="${2:-pop,music}"
LYRICS_FILE="${3:-}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Verify audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo -e "${RED}Error: Audio file not found: $AUDIO_FILE${NC}"
    exit 1
fi

# Get audio filename without extension
AUDIO_NAME=$(basename "${AUDIO_FILE%.*}")

echo -e "\n${GREEN}Step 1: Setting up directories${NC}"
DATASET_DIR="$SCRIPT_DIR/datasets/popchart_tokens"
OUTPUT_DIR="$SCRIPT_DIR/output/overfit_$AUDIO_NAME"
CONFIG_DIR="$SCRIPT_DIR/overfit_$AUDIO_NAME"

mkdir -p "$DATASET_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CONFIG_DIR"

echo "  Dataset dir: $DATASET_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Config dir: $CONFIG_DIR"

echo -e "\n${GREEN}Step 2: Tokenizing audio${NC}"
LYRICS_ARG=""
if [ -n "$LYRICS_FILE" ] && [ -f "$LYRICS_FILE" ]; then
    LYRICS_ARG="--lyrics $LYRICS_FILE"
    echo "  Using lyrics from: $LYRICS_FILE"
fi

python "$SCRIPT_DIR/tokenize_audio.py" \
    --audio_path "$AUDIO_FILE" \
    --output_dir "$DATASET_DIR" \
    --tags "$TAGS" \
    $LYRICS_ARG

echo -e "\n${GREEN}Step 3: Creating companion text files${NC}"
# Create .txt file with tags
TOKEN_FILE=$(ls -1 "$DATASET_DIR"/*.npy 2>/dev/null | head -1)
if [ -n "$TOKEN_FILE" ]; then
    BASE_NAME=$(basename "${TOKEN_FILE%.npy}")
    echo "$TAGS" > "$DATASET_DIR/${BASE_NAME}.txt"
    echo "  Created: $DATASET_DIR/${BASE_NAME}.txt"

    # Create .lyrics file if lyrics were provided
    if [ -n "$LYRICS_FILE" ] && [ -f "$LYRICS_FILE" ]; then
        cp "$LYRICS_FILE" "$DATASET_DIR/${BASE_NAME}.lyrics"
        echo "  Created: $DATASET_DIR/${BASE_NAME}.lyrics"
    else
        touch "$DATASET_DIR/${BASE_NAME}.lyrics"
        echo "  Created: $DATASET_DIR/${BASE_NAME}.lyrics (empty for instrumental)"
    fi
fi

echo -e "\n${GREEN}Step 4: Creating training configuration${NC}"

# Read lyrics for validation if available
VALIDATION_LYRICS=""
if [ -n "$LYRICS_FILE" ] && [ -f "$LYRICS_FILE" ]; then
    VALIDATION_LYRICS=$(cat "$LYRICS_FILE" | tr '\n' ' ' | head -c 500)
fi

# Create config.json
cat > "$CONFIG_DIR/config.json" << EOF
{
  "model_family": "heartmula",
  "model_type": "lora",
  "model_flavour": "3b",
  "pretrained_model_name_or_path": "HeartMuLa/HeartMuLa-oss-3B",

  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",

  "data_backend_config": "$CONFIG_DIR/multidatabackend.json",

  "learning_rate": 1e-4,
  "lora_rank": 64,
  "lora_alpha": 64,
  "lora_dropout": 0.0,

  "optimizer": "adamw_bf16",
  "lr_scheduler": "constant",
  "lr_warmup_steps": 0,

  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "max_grad_norm": 1.0,

  "max_train_steps": 1000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 100,

  "output_dir": "$OUTPUT_DIR",

  "seed": 42,
  "report_to": "none",

  "validation_steps": 50,
  "validation_guidance": 3.0,
  "validation_seed": 42,
  "validation_num_inference_steps": 50,
  "validation_disable_unconditional": true,

  "validation_prompt": "$TAGS",
  "validation_lyrics": "$VALIDATION_LYRICS",

  "vae_cache_disable": true,

  "resolution": 0
}
EOF
echo "  Created: $CONFIG_DIR/config.json"

# Create multidatabackend.json
cat > "$CONFIG_DIR/multidatabackend.json" << EOF
[
  {
    "id": "overfit-data",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "$DATASET_DIR",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "disabled": false,
    "config": {
      "audio_caption_fields": ["tags"],
      "lyrics_column": "lyrics"
    }
  }
]
EOF
echo "  Created: $CONFIG_DIR/multidatabackend.json"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "To start training with SimpleTuner:"
echo ""
echo -e "  ${YELLOW}cd /path/to/SimpleTuner${NC}"
echo -e "  ${YELLOW}mkdir -p config/overfit_$AUDIO_NAME${NC}"
echo -e "  ${YELLOW}cp $CONFIG_DIR/*.json config/overfit_$AUDIO_NAME/${NC}"
echo -e "  ${YELLOW}simpletuner train env=overfit_$AUDIO_NAME${NC}"
echo ""
echo "Or directly:"
echo ""
echo -e "  ${YELLOW}simpletuner train --config $CONFIG_DIR/config.json${NC}"
echo ""
echo "Dataset files:"
ls -la "$DATASET_DIR"

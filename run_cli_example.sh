#!/bin/bash

# KV-Edit CLI Demo Execution Script
# This script provides examples of how to run the KV-Edit CLI demo with different parameters

# Activate conda environment
echo "🔧 Activating conda environment 'KV-Edit'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate KV-Edit

# Check if environment activation was successful
if [ "$CONDA_DEFAULT_ENV" != "KV-Edit" ]; then
    echo "❌ Error: Failed to activate conda environment 'KV-Edit'"
    echo "Please make sure the environment exists: conda env list"
    exit 1
fi

echo "✅ Conda environment 'KV-Edit' activated successfully"
echo ""

# Set default paths (modify these according to your setup)
INPUT_IMAGE="path/to/your/input_image.jpg"
MASK_IMAGE="path/to/your/mask_image.png"
SOURCE_PROMPT="A person standing in a garden"
TARGET_PROMPT="A robot standing in a garden"

# Check if required files exist
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "❌ Error: Input image not found at $INPUT_IMAGE"
    echo "Please modify the INPUT_IMAGE variable in this script to point to your image"
    exit 1
fi

if [ ! -f "$MASK_IMAGE" ]; then
    echo "❌ Error: Mask image not found at $MASK_IMAGE"
    echo "Please modify the MASK_IMAGE variable in this script to point to your mask"
    exit 1
fi

echo "🚀 Starting KV-Edit CLI Demo..."
echo "Input Image: $INPUT_IMAGE"
echo "Mask Image: $MASK_IMAGE"
echo "Source Prompt: $SOURCE_PROMPT"
echo "Target Prompt: $TARGET_PROMPT"
echo ""

# Basic example with default parameters
echo "📝 Running basic edit with default parameters..."
python cli_kv_edit.py \
    --input_image "$INPUT_IMAGE" \
    --mask_image "$MASK_IMAGE" \
    --source_prompt "$SOURCE_PROMPT" \
    --target_prompt "$TARGET_PROMPT"

echo "✅ Basic edit completed!" 
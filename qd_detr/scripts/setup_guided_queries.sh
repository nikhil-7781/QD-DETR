#!/bin/bash

# Setup script for guided query generation system
# This script helps you set up the environment and generate guided queries

echo "=================================================="
echo "QD-DETR Guided Query Generation Setup"
echo "=================================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with your Mistral API key:"
    echo "  MISTRAL_API_KEY=your_api_key_here"
    exit 1
fi

# Check if API key is set
if grep -q "your_mistral_api_key_here" .env; then
    echo "ERROR: Please set your actual Mistral API key in .env file"
    exit 1
fi

echo "✓ .env file found with API key"
echo ""

# Install required dependencies
echo "Installing required dependencies..."
pip install python-dotenv requests -q
echo "✓ Dependencies installed"
echo ""

# Check if data files exist
if [ ! -f "data/highlight_train_release.jsonl" ]; then
    echo "WARNING: Training data not found at data/highlight_train_release.jsonl"
    echo "Please ensure your dataset is in the correct location"
    exit 1
fi

echo "✓ Dataset files found"
echo ""

# Ask user if they want to generate guided queries now
read -p "Do you want to generate guided queries now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Step 1: Generating guided queries using Mistral LLM..."
    echo "This may take a while depending on dataset size and API rate limits..."
    python generate_guided_queries.py

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Guided queries generated successfully"
        echo ""

        read -p "Do you want to generate CLIP embeddings now? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo ""
            echo "Step 2: Generating CLIP embeddings for guided queries..."
            echo "This requires GPU for faster processing..."
            python generate_guided_clip_features.py

            if [ $? -eq 0 ]; then
                echo ""
                echo "✓ CLIP embeddings generated successfully"
                echo ""
                echo "=================================================="
                echo "Setup Complete!"
                echo "=================================================="
                echo ""
                echo "You can now train with guided queries using:"
                echo "  bash qd_detr/scripts/train.sh --use_guided_queries"
                echo ""
            else
                echo "ERROR: CLIP embedding generation failed"
                exit 1
            fi
        fi
    else
        echo "ERROR: Guided query generation failed"
        exit 1
    fi
fi

echo ""
echo "Setup script finished!"

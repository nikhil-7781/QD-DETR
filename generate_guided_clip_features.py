"""
Generate CLIP text features for guided queries.
This script creates CLIP embeddings for the guided queries to be used in training.
"""
import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Try to import the local CLIP implementation
    from run_on_video.clip import clip
except ImportError:
    # Fallback to pip-installed CLIP
    try:
        import clip
    except ImportError:
        raise ImportError(
            "CLIP is not installed. Please install it with:\n"
            "pip install git+https://github.com/openai/CLIP.git"
        )


def load_clip_model(device="cuda"):
    """Load CLIP model for text encoding."""
    print("Loading CLIP model...")
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    return model


def encode_text_batch(model, texts, device="cuda", batch_size=32):
    """
    Encode a batch of texts using CLIP.

    Args:
        model: CLIP model
        texts: List of text strings
        device: Device to run on
        batch_size: Batch size for encoding

    Returns:
        Dictionary with 'pooler_output' and 'last_hidden_state' features
    """
    all_pooler_outputs = []
    all_last_hidden_states = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)

            # Get text features
            text_features = model.encode_text(text_tokens)

            # Normalize features (as done in CLIP)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            all_pooler_outputs.append(text_features.cpu().numpy())

    pooler_output = np.concatenate(all_pooler_outputs, axis=0)

    # For compatibility, create last_hidden_state as expanded version
    # CLIP's text encoder produces a single pooled output, but we need to match
    # the expected format which might include sequence-level features
    last_hidden_state = np.expand_dims(pooler_output, axis=1)  # (N, 1, D)

    return pooler_output, last_hidden_state


def process_guided_queries(guided_data_path, output_dir, device="cuda"):
    """
    Process guided queries and generate CLIP features.

    Args:
        guided_data_path: Path to guided queries JSONL file
        output_dir: Directory to save CLIP features
        device: Device to run on
    """
    print(f"Processing {guided_data_path}...")

    # Load guided queries
    data = []
    with open(guided_data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Load CLIP model
    model = load_clip_model(device)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each query
    print(f"Generating CLIP features for {len(data)} queries...")
    for item in tqdm(data, desc="Encoding queries"):
        qid = item['qid']
        guided_query = item['guided_query']

        # Encode guided query
        pooler_output, last_hidden_state = encode_text_batch(
            model, [guided_query], device=device, batch_size=1
        )

        # Save as .npz file (matching the original format)
        output_path = output_dir / f"qid{qid}.npz"
        np.savez(
            output_path,
            pooler_output=pooler_output[0],  # Shape: (512,)
            last_hidden_state=last_hidden_state[0]  # Shape: (1, 512) or (L, 512)
        )

    print(f"Saved CLIP features to {output_dir}")


def main():
    """Process all dataset splits."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    guided_data_dir = Path("data/guided_queries")
    features_root = Path("../features")
    guided_features_dir = features_root / "clip_text_features_guided"

    datasets = [
        "highlight_train_guided.jsonl",
        "highlight_val_guided.jsonl",
        "highlight_test_guided.jsonl",
    ]

    for dataset_file in datasets:
        guided_data_path = guided_data_dir / dataset_file

        if not guided_data_path.exists():
            print(f"Skipping {dataset_file} - file not found")
            continue

        process_guided_queries(guided_data_path, guided_features_dir, device)

    print("\n" + "="*80)
    print("CLIP feature generation complete!")
    print(f"Features saved to: {guided_features_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

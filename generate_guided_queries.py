"""
Generate guided queries using Mistral LLM for video moment retrieval.
This script processes the dataset and creates visually detailed guided queries.
"""
import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
import requests
from pathlib import Path

# Load environment variables
load_dotenv()

MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
if not MISTRAL_API_KEY or MISTRAL_API_KEY == 'your_mistral_api_key_here':
    raise ValueError("Please set MISTRAL_API_KEY in .env file")

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def generate_guided_query(original_query, max_retries=3):
    """
    Generate a guided query with more visual details using Mistral LLM.

    Args:
        original_query: Original query text
        max_retries: Maximum number of retries on failure

    Returns:
        Guided query string with enhanced visual details
    """
    prompt = f"""You are an expert at creating detailed visual descriptions for video moment retrieval tasks.

Given a simple query about a video moment, expand it with more visual details that would help identify the specific moment in a video. Focus on:
- Visual appearance (colors, clothing, objects, actions)
- Spatial relationships (positions, movements)
- Temporal aspects (sequence of actions)

Keep the expanded query concise (1-2 sentences) but more descriptive than the original.

Original query: "{original_query}"

Expanded query with visual details:"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }

    data = {
        "model": "mistral-tiny",  # Free tier model
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(MISTRAL_API_URL, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            guided_query = result['choices'][0]['message']['content'].strip()

            # Remove any quotation marks that might be added by the model
            guided_query = guided_query.strip('"').strip("'")

            return guided_query

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to generate guided query for: {original_query}")
                return original_query  # Fallback to original query
        except (KeyError, IndexError) as e:
            print(f"Error parsing response: {e}")
            return original_query


def process_dataset(input_path, output_path):
    """
    Process a dataset file and generate guided queries for all entries.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file with guided queries
    """
    print(f"Processing {input_path}...")

    # Read input data
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Generate guided queries
    output_data = []
    for item in tqdm(data, desc=f"Generating guided queries"):
        original_query = item['query']
        guided_query = generate_guided_query(original_query)

        # Add guided query to the item
        item['original_query'] = original_query
        item['guided_query'] = guided_query

        output_data.append(item)

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    # Write output data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + '\n')

    print(f"Saved guided queries to {output_path}")
    return output_data


def main():
    """Process all dataset splits."""
    data_dir = Path("data")
    guided_data_dir = Path("data/guided_queries")

    datasets = [
        ("highlight_train_release.jsonl", "highlight_train_guided.jsonl"),
        ("highlight_val_release.jsonl", "highlight_val_guided.jsonl"),
        ("highlight_test_release.jsonl", "highlight_test_guided.jsonl"),
    ]

    all_results = {}

    for input_file, output_file in datasets:
        input_path = data_dir / input_file
        output_path = guided_data_dir / output_file

        if not input_path.exists():
            print(f"Skipping {input_file} - file not found")
            continue

        result = process_dataset(input_path, output_path)
        all_results[output_file] = result

    # Print sample results
    print("\n" + "="*80)
    print("SAMPLE ORIGINAL vs GUIDED QUERIES (10 random examples):")
    print("="*80)

    if "highlight_train_guided.jsonl" in all_results:
        import random
        samples = random.sample(all_results["highlight_train_guided.jsonl"],
                              min(10, len(all_results["highlight_train_guided.jsonl"])))

        for i, sample in enumerate(samples, 1):
            print(f"\n{i}. QID: {sample['qid']}")
            print(f"   Original: {sample['original_query']}")
            print(f"   Guided:   {sample['guided_query']}")

    print("\n" + "="*80)
    print("Guided queries generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()

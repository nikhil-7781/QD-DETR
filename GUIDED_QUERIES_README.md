# Guided Query Generation for Video Moment Retrieval

This guide explains how to use LLM-generated guided queries for enhanced video moment retrieval in QD-DETR.

## Overview

The system generates visually detailed "guided queries" from original dataset queries using Mistral LLM, then creates CLIP embeddings for these enhanced queries to improve video moment retrieval performance.

## Setup

### 1. Install Dependencies

```bash
pip install python-dotenv requests torch torchvision
```

If CLIP is not already installed:
```bash
pip install git+https://github.com/openai/CLIP.git
```

### 2. Configure Mistral API Key

1. Edit the `.env` file in the project root
2. Add your Mistral API key:
   ```
   MISTRAL_API_KEY=your_actual_api_key_here
   ```

You can get a free API key from [Mistral AI](https://console.mistral.ai/).

## Usage Workflow

### Step 1: Generate Guided Queries

Run the guided query generation script to create enhanced queries using Mistral LLM:

```bash
python generate_guided_queries.py
```

This will:
- Read original queries from `data/highlight_train_release.jsonl`, `data/highlight_val_release.jsonl`, `data/highlight_test_release.jsonl`
- Generate visually detailed guided queries using Mistral LLM
- Save results to `data/guided_queries/highlight_*_guided.jsonl`
- Display 10 random original vs guided query pairs

**Expected output structure:**
```json
{
  "qid": 9769,
  "query": "some military patriots takes us through their safety procedures.",
  "original_query": "some military patriots takes us through their safety procedures.",
  "guided_query": "Military personnel in uniform demonstrating and explaining safety procedures and protocols in a structured setting, with visible safety equipment and instructional materials.",
  "vid": "j7rJstUseKg_360.0_510.0",
  ...
}
```

### Step 2: Generate CLIP Embeddings for Guided Queries

Run the CLIP feature generation script:

```bash
python generate_guided_clip_features.py
```

This will:
- Read guided queries from `data/guided_queries/`
- Generate CLIP text embeddings for each guided query
- Save features to `../features/clip_text_features_guided/qid{qid}.npz`

Each `.npz` file contains:
- `pooler_output`: Shape (512,) - pooled CLIP text features
- `last_hidden_state`: Shape (1, 512) - sequence-level features

### Step 3: Train with Guided Queries

Modify your training script to use the `--use_guided_queries` flag:

```bash
bash qd_detr/scripts/train.sh --use_guided_queries --seed 2018
```

Or for audio-enhanced training:
```bash
bash qd_detr/scripts/train_audio.sh --use_guided_queries --seed 2018
```

The system will:
- Automatically load guided queries from `data/guided_queries/`
- Use CLIP features from `../features/clip_text_features_guided/`
- Print 10 random original-guided query pairs at the end of training

## File Structure

```
QD-DETR/
├── .env                                    # Mistral API key configuration
├── generate_guided_queries.py              # Script to generate guided queries
├── generate_guided_clip_features.py        # Script to generate CLIP embeddings
├── data/
│   ├── highlight_train_release.jsonl       # Original training data
│   ├── highlight_val_release.jsonl         # Original validation data
│   ├── highlight_test_release.jsonl        # Original test data
│   └── guided_queries/                     # Generated guided queries
│       ├── highlight_train_guided.jsonl
│       ├── highlight_val_guided.jsonl
│       └── highlight_test_guided.jsonl
└── ../features/
    ├── clip_text_features/                 # Original CLIP features
    └── clip_text_features_guided/          # Guided query CLIP features
        └── qid{qid}.npz
```

## Implementation Details

### Modified Files

1. **qd_detr/config.py**
   - Added `--use_guided_queries` flag

2. **qd_detr/train.py**
   - Updated dataset initialization to use guided queries when flag is set
   - Added `print_guided_query_samples()` function to display query pairs
   - Automatically redirects data/feature paths when using guided queries

3. **qd_detr/start_end_dataset.py**
   - Added `use_guided_queries` parameter to dataset class

4. **qd_detr/start_end_dataset_audio.py**
   - Added `use_guided_queries` parameter to audio dataset class

### Guided Query Generation Prompt

The system uses the following prompt template with Mistral LLM:

```
You are an expert at creating detailed visual descriptions for video moment retrieval tasks.

Given a simple query about a video moment, expand it with more visual details that would help
identify the specific moment in a video. Focus on:
- Visual appearance (colors, clothing, objects, actions)
- Spatial relationships (positions, movements)
- Temporal aspects (sequence of actions)

Keep the expanded query concise (1-2 sentences) but more descriptive than the original.

Original query: "{original_query}"

Expanded query with visual details:
```

### Error Handling

- **Rate Limiting**: The script includes exponential backoff and retries for API calls
- **Fallback**: If LLM generation fails, the original query is used as fallback
- **Validation**: Scripts check for API key presence before running

## Troubleshooting

### Issue: "MISTRAL_API_KEY not found"
**Solution**: Make sure you've set your API key in the `.env` file

### Issue: "Data file not found"
**Solution**: Ensure your dataset files are in the correct location (`data/` directory)

### Issue: CLIP feature generation is slow
**Solution**: The script uses GPU if available. Check `torch.cuda.is_available()`. You can also reduce batch size if running into memory issues.

### Issue: Training script doesn't find guided queries
**Solution**: Make sure you've run both `generate_guided_queries.py` and `generate_guided_clip_features.py` before training

## Performance Considerations

- **API Calls**: Generating guided queries for the full dataset may take time due to API rate limits
- **Storage**: Guided CLIP features require approximately the same storage as original features (~500 bytes per query)
- **Computation**: CLIP feature generation benefits significantly from GPU acceleration

## Example Output

At the end of training with `--use_guided_queries`, you'll see:

```
================================================================================
SAMPLE ORIGINAL vs GUIDED QUERIES (10 random examples):
================================================================================

1. QID: 9769
   Original: some military patriots takes us through their safety procedures.
   Guided:   Military personnel in uniform demonstrating and explaining safety procedures
             and protocols in a structured setting, with visible safety equipment.

2. QID: 10016
   Original: Man in baseball cap eats before doing his interview.
   Guided:   A man wearing a baseball cap is shown eating food, then transitions to speaking
             during an interview segment, seated in an indoor setting.

...

================================================================================
```

## Citation

If you use this guided query generation approach in your research, please cite the original QD-DETR paper and acknowledge the use of Mistral AI for query enhancement.

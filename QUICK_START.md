# Quick Start Guide - Guided Queries for QD-DETR

## Prerequisites Checklist

Before running, ensure you have:

- ✅ GPU access (CUDA-enabled)
- ✅ Python 3.7+ installed
- ✅ Dataset files in `data/` directory
- ✅ Mistral API key (get free tier from https://console.mistral.ai/)

## Step-by-Step Execution

### 1. Install Dependencies

```bash
pip install python-dotenv requests
pip install torch torchvision  # If not already installed
```

### 2. Configure API Key

Edit `.env` file and add your Mistral API key:
```bash
nano .env
# Change: MISTRAL_API_KEY=your_mistral_api_key_here
# To:     MISTRAL_API_KEY=<your_actual_key>
```

### 3. Generate Guided Queries (First Time Only)

```bash
python generate_guided_queries.py
```

**Expected output:**
- Creates `data/guided_queries/` directory
- Generates 3 files: `highlight_train_guided.jsonl`, `highlight_val_guided.jsonl`, `highlight_test_guided.jsonl`
- Displays 10 sample query pairs
- Takes ~10-30 minutes depending on dataset size and API rate limits

### 4. Generate CLIP Embeddings (First Time Only)

```bash
python generate_guided_clip_features.py
```

**Expected output:**
- Creates `../features/clip_text_features_guided/` directory
- Generates `.npz` files for each query (format: `qid{number}.npz`)
- Takes ~5-15 minutes with GPU

### 5. Train with Guided Queries

```bash
bash qd_detr/scripts/train.sh --use_guided_queries --seed 2018
```

**Or for audio-enhanced:**
```bash
bash qd_detr/scripts/train_audio.sh --use_guided_queries --seed 2018
```

**At the end of training, you'll see:**
```
================================================================================
SAMPLE ORIGINAL vs GUIDED QUERIES (10 random examples):
================================================================================

1. QID: 9769
   Original: some military patriots takes us through their safety procedures.
   Guided:   Military personnel in uniform demonstrating and explaining safety...

...
```

## Verification Commands

Check if everything is set up correctly:

```bash
# 1. Check .env file
cat .env | grep MISTRAL_API_KEY

# 2. Check if guided queries were generated
ls -lh data/guided_queries/

# 3. Check if CLIP features were generated
ls ../features/clip_text_features_guided/ | wc -l
# Should show number of queries in dataset

# 4. Test imports
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "from dotenv import load_dotenv; print('dotenv OK')"
python -c "import requests; print('requests OK')"
```

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'dotenv'
```bash
pip install python-dotenv
```

### Issue: ModuleNotFoundError: No module named 'clip'
```bash
pip install git+https://github.com/openai/CLIP.git
```

### Issue: CUDA out of memory during CLIP generation
Edit `generate_guided_clip_features.py` line 40 and change:
```python
batch_size=32  # Change to smaller value like 16 or 8
```

### Issue: Mistral API rate limit errors
The script has built-in retry logic. Just wait and it will continue. For very large datasets, consider running overnight.

### Issue: FileNotFoundError for guided queries
Make sure you completed Step 3 (generate_guided_queries.py) before Step 4.

## Files Created

After running all steps:

```
QD-DETR/
├── .env                                      # Your API key (DO NOT COMMIT)
├── data/guided_queries/
│   ├── highlight_train_guided.jsonl          # ~3.9 MB
│   ├── highlight_val_guided.jsonl            # ~800 KB
│   └── highlight_test_guided.jsonl           # ~200 KB
└── ../features/clip_text_features_guided/
    ├── qid9769.npz                           # ~500 bytes each
    ├── qid10016.npz
    └── ... (one per query)
```

## Notes

- **First time setup**: Steps 3-4 only need to be run once
- **Training**: Once guided queries and features are generated, you can train multiple times without regenerating
- **Cost**: Mistral free tier should be sufficient for QVHighlights dataset
- **Time**: Total first-time setup takes ~20-45 minutes

## Ready to Run?

Yes! The code is ready. Just follow steps 1-5 above.

Good luck with your experiments! 🚀

# QDECA Module Documentation

## Query Decomposition and Event-Centric Attention (QDECA)

This module enhances QD-DETR by decomposing text queries into semantic components (events, objects, temporal modifiers) and applying specialized cross-attention between video clips and each component.

---

## Files Modified

### 1. **qd_detr/qdeca.py** (NEW)
- `LearnedDecomposer`: Decomposes text queries using learnable attention queries
- `QDECA`: Main module that applies component-wise cross-attention and gating

### 2. **qd_detr/model.py** (MODIFIED)
- Added QDECA import (line 15)
- Added `use_qdeca`, `max_q_l`, `nheads` parameters to `QDDETR.__init__()` (line 30)
- Instantiated QDECA module in `__init__` (lines 93-95)
- Applied QDECA to video features in `forward()` (lines 121-122)
- Updated `build_model()` to pass QDECA parameters (lines 539-579)

### 3. **qd_detr/config.py** (MODIFIED)
- Added `--use_qdeca` flag (lines 152-153)

### 4. **qd_detr/__init__.py** (MODIFIED)
- Exported QDECA and LearnedDecomposer classes

---

## Usage

### Training with QDECA

Add the `--use_qdeca` flag to your training command:

```bash
# Standard training (video only) with QDECA
bash qd_detr/scripts/train.sh --seed 2018 --use_qdeca

# Training with video + audio and QDECA
bash qd_detr/scripts/train_audio.sh --seed 2018 --use_qdeca
```

### Training without QDECA (baseline)

Simply omit the flag:

```bash
bash qd_detr/scripts/train.sh --seed 2018
```

---

## How QDECA Works

### Architecture Flow

```
Input Text Query: "person picks up ball then throws it"
                          ↓
            [LearnedDecomposer]
                    ↓
    ┌──────────────┼──────────────┐
    ↓              ↓               ↓
Event Tokens   Object Tokens  Temporal Tokens
["picks","throws"] ["person","ball"]  ["then"]
    ↓              ↓               ↓
    [Cross-Attention with Video Clips]
    ↓              ↓               ↓
attn_e         attn_o          attn_t
(B,L_vid,D)   (B,L_vid,D)    (B,L_vid,D)
    └──────────────┼──────────────┘
                   ↓
        [Clip-Adaptive Gating]
                   ↓
           Enhanced Video Features
              (B, L_vid, D)
```

### Key Components

1. **LearnedDecomposer**
   - Uses learnable query vectors to extract event, object, and temporal components
   - No external NLP parser required
   - End-to-end trainable

2. **Component-Wise Cross-Attention**
   - Video clips attend to each semantic component separately
   - Preserves token-level granularity (no pooling)
   - Uses standard multi-head attention

3. **Clip-Adaptive Gating**
   - Each video clip learns to weight event/object/temporal branches
   - Gating weights: `g = softmax(MLP(video_clip))` ∈ R³
   - Final output: `g[0]·attn_e + g[1]·attn_o + g[2]·attn_t`

---

## Testing

### Run Integration Tests

```bash
python3 test_qdeca_integration.py
```

This verifies:
- ✓ Shape preservation: `(B, L_vid, D)` → `(B, L_vid, D)`
- ✓ Gradient flow through all branches
- ✓ Batch independence
- ✓ Null temporal token fallback

---

## Expected Improvements

Based on the module design, QDECA should improve performance on:

1. **Multi-action queries**: "person walks then sits down"
   - Event branch can distinguish between multiple actions

2. **Complex descriptions**: "red ball on the table near the window"
   - Object branch focuses on entities and attributes

3. **Temporal queries**: "at the beginning", "after the speech", "towards the end"
   - Temporal branch biases attention to relevant time regions

4. **Long queries**: More robust to noisy or verbose descriptions
   - Gating can down-weight irrelevant components

---

## Evaluation & Analysis

### Extract Gating Weights for Analysis

```python
from qd_detr.model import build_model

model, _ = build_model(args)
model.eval()

# During inference
with torch.no_grad():
    output = model(src_txt, src_txt_mask, src_vid, src_vid_mask)

    # Get gating weights
    if model.use_qdeca:
        gate_weights = model.qdeca.get_gate_weights(src_vid)
        # gate_weights: (B, L_vid, 3) - [event, object, temporal]

        print(f"Avg Event Weight: {gate_weights[..., 0].mean():.3f}")
        print(f"Avg Object Weight: {gate_weights[..., 1].mean():.3f}")
        print(f"Avg Temporal Weight: {gate_weights[..., 2].mean():.3f}")
```

### Ablation Studies

To evaluate component importance:

```bash
# Full QDECA
bash qd_detr/scripts/train.sh --use_qdeca --exp_id qdeca_full

# Baseline (no QDECA)
bash qd_detr/scripts/train.sh --exp_id baseline
```

Compare metrics (mAP, R@1) across IoU thresholds on QVHighlights.

---

## Implementation Details

### Module Parameters

- **d_model**: 256 (matches transformer hidden dimension)
- **num_heads**: 8 (matches transformer attention heads)
- **max_txt_len**: 32 (from `args.max_q_l`)

### Computational Overhead

- **Parameters**: ~3 attention modules + gating MLP ≈ <1% of total model
- **FLOPs**: `O(L_vid × L_txt × D)` - same order as baseline T2V encoder
- **Memory**: Minimal additional GPU memory required

### Initialization

- Learnable query vectors: Orthogonal initialization for diversity
- Gating MLP: Small weights (std=0.01) to avoid initial imbalance
- Attention modules: Standard PyTorch initialization

---

## Troubleshooting

### Issue: Shape mismatch errors

**Solution**: Ensure `max_q_l` and `nheads` are passed correctly in `build_model()`.

### Issue: NaN loss or unstable training

**Solution**:
- Check gating MLP initialization (should be small)
- Verify gradient clipping is enabled (`--grad_clip 0.1`)
- Try reducing learning rate for QDECA parameters

### Issue: No improvement over baseline

**Possible causes**:
- Dataset has simple queries (QDECA helps more on complex queries)
- Need longer training for QDECA to learn effective decomposition
- Try analyzing gating weights to verify branches are being used

---

## Citation

If you use QDECA in your research, please cite:

```bibtex
@inproceedings{moon2023query,
  title={Query-dependent video representation for moment retrieval and highlight detection},
  author={Moon, WonJun and Hyun, Sangeek and Park, SangUk and Park, Dongchan and Heo, Jae-Pil},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23023--23033},
  year={2023}
}
```

---

## Contact

For questions about QDECA implementation, please open an issue on the GitHub repository.

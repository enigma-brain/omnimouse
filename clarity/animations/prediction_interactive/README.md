# Vizeval blog tensors

Extracted from the vizeval prediction run for a single video (session `29234-6-9`,
video_id 147, 2s sliding windows at stride 3 @ 30 Hz for neural / stride 2 @ 20 Hz
for behavior). Each `.npy` is the **second** 1 s of the window (i.e. the
reconstructed half): 30 samples for neural, 20 for behavior.

This video has **71 chunks** (`chunk_idx` 0..70), i.e. 71 sliding windows per
strategy per modality.

## Layout

```
predictions/{strategy}/chunk_{NNNN}_response.npy   # shape (35, 30) float32
predictions/{strategy}/chunk_{NNNN}_behavior.npy   # shape (5, 20)  float32
ground_truth/chunk_{NNNN}_response.npy             # shape (35, 30) float32
ground_truth/chunk_{NNNN}_behavior.npy             # shape (5, 20)  float32
meta.csv                                           # strategy, modality, chunk_idx, path
neuron_index.csv                                   # idx (1..35), neuron_id
```

- Row `i` of a response array corresponds to `neuron_index.csv` row with `idx == i+1`.
- Ground truth is stored once per chunk (shared across strategies) and uses
  `strategy == "ground_truth"` in `meta.csv`.
- Not every strategy produces every modality:
  - Strategies with `max_n_reconstructed_neurons = 0` (e.g. `all_response_only`,
    `all_response_video`) have no response prediction — no response `.npy` for them.
  - `_behavior*` strategies use behavior as **input** (`behavior_encoded=true,
    behavior_reconstructed=false`) — no behavior `.npy` for them.
  - Strategies with `max_n_reconstructed_neurons = 3072` (e.g. `population_*`,
    `video_only`) reconstruct the **last 3072 neuron IDs** (5213..8284) — always
    the same set, every chunk. All 35 target neurons fall inside this range, so
    no NaNs in practice. The `NaN` fill is kept as a safety net in case future
    targets fall outside.

## Loading

```python
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("/path/to/blog")
meta = pd.read_csv(root / "meta.csv")
neurons = pd.read_csv(root / "neuron_index.csv")  # idx, neuron_id

def load(strategy: str, modality: str, chunk_idx: int) -> np.ndarray:
    row = meta[(meta.strategy == strategy) & (meta.modality == modality) & (meta.chunk_idx == chunk_idx)]
    if row.empty:
        raise KeyError(f"No tensor for ({strategy}, {modality}, chunk {chunk_idx})")
    return np.load(root / row.iloc[0].path)

# Prediction + matching ground truth for one chunk
pred = load("causal_response_25", "response", chunk_idx=0)   # (35, 30)
gt   = load("ground_truth",       "response", chunk_idx=0)   # (35, 30)

# Behavior (only for strategies with behavior_reconstructed=true)
beh_pred = load("causal_response_25", "behavior", chunk_idx=0)  # (5, 20)
beh_gt   = load("ground_truth",       "behavior", chunk_idx=0)  # (5, 20)

# Look up neuron id for the 3rd row of any response array
neuron_id_of_row_3 = int(neurons[neurons.idx == 3].neuron_id.iloc[0])
```

## Strategies (107 total)

Variants below use a common suffix set: `{"", "_behavior", "_video", "_behavior_video"}`.
`_behavior`/`_video`/`_behavior_video` means that modality is *encoded as input*;
`_behavior` also implies behavior is **not** reconstructed (so those strategies
have no behavior `.npy`). Plain (no suffix) predicts both response and behavior.

### Non-combo (31)

**Context-only baselines (3)** — no response reconstruction, behavior only:
- `all_response_only`, `all_response_video`, `video_only`

**Population-of-N visible neurons (12)** — for `N ∈ {64, 256, 1024}`:
- `population_{N}_response{suffix}` for each of the 4 suffixes

**Causal response-prefix (16)** — for `N ∈ {10, 15, 20, 25}`:
- `causal_response_{N}{suffix}` for each of the 4 suffixes

### Combo: population + causal (76)

Combines a population mask (P visible neurons) with a causal prefix (first N samples).
Same 4 suffix variants.

- `population_{P}_causal_response_{N}{suffix}`
- `inv_population_{P}_causal_response_{N}{suffix}` — inverse mask (complementary visible/reconstructed sets)

Presence per `(P, N)` — each cell represents up to 4 suffix variants:

| P     | regular N                                         | inv_ N                                            |
|-------|---------------------------------------------------|---------------------------------------------------|
| 64    | 10 ✓, 15 ✓, 20 ✓, 25 ✓  (all 4 suffixes each)     | 10 ✓, 15 ✓, 20 ✓, 25 ✓  (all 4 suffixes each)     |
| 256   | 10 ✓, 15 ✓, 20 ✓, 25 ✓  (all 4 suffixes each)     | 10 ✓, 15 ✓, 20 ✓, 25 ✓  (all 4 suffixes each)     |
| 1024  | 20 (only plain + `_behavior`), 25 ✓ (all 4)       | 20 (only plain + `_behavior`), 25 ✓ (all 4)       |

**Missing cells** (never produced — the resume run ended before these):
- `[inv_]population_1024_causal_response_{10, 15}` — all 4 suffix variants each (16 total)
- `[inv_]population_1024_causal_response_20_video`, `[inv_]population_1024_causal_response_20_behavior_video` (4 total)

### Modality availability

- **Response prediction** present for every strategy **except** `all_response_only` and `all_response_video`.
- **Behavior prediction** present for every strategy **except** those ending in `_behavior` or `_behavior_video`.

## Assumptions / conventions

- Neural sample rate 30 Hz, behavior sample rate 20 Hz.
- The 2s context window is split into 1s "prefix" + 1s "suffix"; the stored arrays
  are the suffix (positions 30–59 for neural, 20–39 for behavior).
- Behavior channel order (rows of the (5, 20) arrays):
  0 = Pupil Size, 1 = Pupil Size Delta, 2 = Pupil Position X,
  3 = Pupil Position Y, 4 = Treadmill Speed.

# Local data directory

Market data are not distributed with this artifact. Downloaded archives and all
derived arrays remain local and are ignored by Git.

The snapshot loader expects each processed dataset at
`data/processed/snapshot_l2/<dataset-id>/` with the following files:

```text
states.npy
states_raw.npy
actions.npy
timestamps.npy
day_id.npy
segment_id.npy
source_row_idx.npy
split.npy
normalization.json
metadata.json
manifest.yaml
DATASET_CARD.md
```

The Public-ITCH builder in `scripts/prepare_public_itch_multisymbol.py` creates
this layout and can update `data/manifests/datasets.yaml`.

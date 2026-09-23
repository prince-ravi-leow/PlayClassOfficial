# Dataset

Pipeline outputs from the three-step dataset build (see
`pipeline/build_dataset.py`, `pipeline/extract_features.py`, and the embedding
extraction scripts).

## Files

| File                                          | Description                                                       |
| --------------------------------------------- | ----------------------------------------------------------------- |
| `tracks.parquet`                              | Postprocessed tracks with protocol bird IDs and window column.    |
| `labels.parquet`                              | Ethogram annotations aligned to tracking windows.                 |
| `features_windowed.parquet`                   | Morphokinematic descriptors summarised over 5-second windows.     |
| `embeddings_{backbone}_{size}[_{variant}].pt` | Embeddings per (video, bird, window), keyed as a dict of tensors. |

## Embedding variants

| File                                     | Backbone   |
| ---------------------------------------- | ---------- |
| `embeddings_dinov3_vitb_temporal.pt`     | DINOv3     |
| `embeddings_dinov3_vitl_temporal.pt`     | DINOv3     |
| `embeddings_vjepa2_vitl_temporal.pt`     | V-JEPA 2   |
| `embeddings_vjepa21_vitb_temporal.pt`    | V-JEPA 2.1 |
| `embeddings_vjepa21_vitl_temporal.pt`    | V-JEPA 2.1 |
| `embeddings_videoprism_vitb_temporal.pt` | VideoPrism |
| `embeddings_videoprism_vitl_temporal.pt` | VideoPrism |

# Embeddings Extraction

| Script                                      | Env          | Description                                  |
| ------------------------------------------- | ------------ | -------------------------------------------- |
| `pipeline/extract_embeddings_dinov3.py`     | `default`    | DINOv3 CLS-token embeddings (image backbone) |
| `pipeline/extract_embeddings_vjepa2.py`     | `default`    | V-JEPA 2 / 2.1 video embeddings              |
| `pipeline/extract_embeddings_videoprism.py` | `videoprism` | VideoPrism video embeddings (JAX)            |

All three scripts read `tracks.parquet` from the dataset dir, load video frames,
and save a `.pt` dict keyed by `(video_id, bird_id, window)`.

---

## DINOv3 (image backbone)

No setup needed — model weights download automatically from HuggingFace on first
run.

```sh
# Default: ViT-L, bbox crop
pixi run extract_dinov3 \
    --video-dir data/videos/day_28 data/videos/day_29

# ViT-B backbone
pixi run extract_dinov3 \
    --video-dir data/videos/day_28 data/videos/day_29 \
    --model-name facebook/dinov3-vitb16-pretrain-lvd1689m

# Custom resolution (DINOv3 was trained at 256; supports up to 768)
pixi run extract_dinov3 \
    --video-dir data/videos/day_28 data/videos/day_29 --resolution 256
```

Output: `embeddings_dinov3_vitl.pt` (or `vitb`, `r256`, etc. — name
auto-generated from args).

---

## V-JEPA 2 / 2.1 (video backbone)

### V-JEPA 2 (HuggingFace) — no setup needed

```sh
pixi run extract_vjepa2 \
    --video-dir data/videos --device cuda:0 --temporal
```

### V-JEPA 2.1 (torch.hub) — one-time setup required

V-JEPA 2.1 checkpoints are not on HuggingFace; they're downloaded separately and
loaded via `torch.hub`. The setup script also patches a namespace collision
between the hub repo's `src/` directory and this project's own `src/` package.

```sh
# Download ViT-B + ViT-L checkpoints and patch hub cache (run once)
bash scripts/setup_vjepa2.1.sh
```

Available models:

| Model name                  | Size                         |
| --------------------------- | ---------------------------- |
| `vjepa2_1_vit_base_384`     | ViT-B (distilled from ViT-G) |
| `vjepa2_1_vit_large_384`    | ViT-L (default)              |
| `vjepa2_1_vit_giant_384`    | ViT-G                        |
| `vjepa2_1_vit_gigantic_384` | ViT-G2                       |

Then extract:

```sh
pixi run extract_vjepa2 \
    --video-dir data/videos/day_28 data/videos/day_29 \
    --device cuda:0 --temporal \
    --model-name vjepa2_1_vit_large_384
```

Output: `embeddings_vjepa21_vitl_temporal.pt` (name auto-generated from args).

---

## VideoPrism (JAX)

Runs in its own `videoprism` pixi env (JAX + TensorFlow, solved separately
from the PyTorch envs). TF is imported only to be blocked from grabbing the
GPU — the actual inference runs in JAX.

No extra setup: model weights download automatically on first run via the
`videoprism` package.

```sh
# Temporal embeddings (default: ViT-B)
pixi run -e videoprism extract_videoprism \
    --video-dir data/videos/day_28 data/videos/day_29 --device 0 --temporal

# Raw patch tokens for a trainable pooler (~88 GB output — large!)
pixi run -e videoprism extract_videoprism \
    --video-dir data/videos/day_28 data/videos/day_29 --device 0 --raw
```

Output: `embeddings_videoprism_temporal.pt` or `embeddings_videoprism_raw.pt`.

---

## Crop modes

`--crop-mode` controls which part of the frame is fed to the backbone. Not every
script accepts every mode:

| Mode        | Description                                                                                 | DINOv3 | V-JEPA 2 | VideoPrism |
| ----------- | ------------------------------------------------------------------------------------------- | :----: | :------: | :--------: |
| `bbox`      | Per-frame bounding box crop (default)                                                       |   ✓    |    ✓     |     ✓      |
| `plain256`  | Fixed 256×256 square around the bird's bbox centroid, per frame                             |   ✓    |    ✓     |     ✓      |
| `plain384`  | Same, 384×384                                                                               |        |    ✓     |     ✓      |
| `union512`  | Fixed 512×512 square around the centre of the bird's bboxes, same for the whole window      |   ✓    |    ✓     |     ✓      |
| `union384`  | Same, 384×384                                                                               |        |    ✓     |     ✓      |
| `union`     | Union of the bird's bboxes in the window, resized to `--frame-size`                         |        |          |     ✓      |
| `darken512` | `union512` with everything outside the bird's bbox dimmed to 40% _(untested)_               |   ✓    |    ✓     |     ✓      |
| `roi512`    | `union512`; VideoPrism then pools only the patch tokens inside the bird's bbox _(untested)_ |   ✓    |    ✓     |     ✓      |

`roi512` only differs from `union512` in VideoPrism; DINOv3 and V-JEPA 2 ignore
the ROI patch indices.

Any mode other than `bbox` is appended to the output filename, e.g.
`embeddings_vjepa21_vitl_union512_temporal.pt`.

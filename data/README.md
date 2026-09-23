# Data

Research data for automated play-behaviour classification in young chickens,
covering 30 overhead recordings from Oscarsson et al. (2026). Generated results
are excluded from Git; publication on Zenodo is pending. Git retains the data
documentation and manually edited postprocessing corrections.

## Directory structure

- `videos/`: 30 raw 15-minute recordings, days 28 and 29 of rearing
- `labels/`: human-expert ethogram annotations (Excel format)
- `dataset/`: pipeline outputs — cleaned tracks, morphokinematic features, and
  model embeddings
- `postprocessing/`: per-video human-audit JSONs for resolving identity switches
  and removing erroneous tracks
- `results/`: outputs from tracking, classification, and data analysis

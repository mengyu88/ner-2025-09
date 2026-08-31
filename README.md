# Server project backup — 2026-08-31

This repository is a recovery copy of the project sources, configurations,
scripts, and small-to-medium datasets from `/root/shared-nvme/projects`.

It also includes the paper working materials from `SPSR/` and `文献/`: rendered
figures (PNG/PDF/SVG), their Python plotting scripts, JSON source data, and the
project-level CSV/JSON/XLSX metrics, logs, and run records used to support the
paper's experiment visualisations.

## Included projects

- `CNN_Nested_NER`
- `DiffusionNER`
- `Flat-Lattice-Transformer`
- `LEBERT`
- `SPSR-Net`
- `W2NER`
- `locate-and-label`

The 233 MB LEBERT `.npz` dataset is stored with Git LFS. Restore it with
`git lfs install && git lfs pull` after cloning.

## Intentionally excluded from this GitHub backup

Generated caches, Python bytecode, and model checkpoints were not copied.
Metrics, logs, run records, figures, and other non-model output files are
included. The excluded heavyweight directories contain individual model files
up to 521 MB:

- `SPSR-Net/_saved_models` (~19 GB)
- `SPSR-Net/pretrained_models` (~421 MB)
- `CNN_Nested_NER/outputs` (~396 MB)
- `DiffusionNER/outputs` (~510 MB)
- `LEBERT/outputs` (~531 MB)
- `W2NER/outputs` (~413 MB)
- `locate-and-label/outputs` (~415 MB)

They require Git LFS storage capacity or separate object/cloud storage before
the server can be retired. On 2026-08-31, the 19 GB
`SPSR-Net/_saved_models` history was intentionally moved to the server's
recovery trash and is not part of this backup.

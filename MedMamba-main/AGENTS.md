# Repository Guidelines

## Project Structure & Module Organization
Reusable configuration presets live in `configs/` with dataset-specific overrides under `configs/datasets/`. Training loops, dataloaders, and utilities sit inside `src/trainer/`, `src/data/`, and `src/utils/`, while neural architectures reside in `models/`. Root scripts (`train.py`, `test.py`, `example_usage.py`) bootstrap experiments, and visualization helpers are kept inside `assets/`, `grad_cam/`, and `ConfusionMatrix/`. Place new datasets outside the repo and reference them via absolute paths inside configs.

## Build, Test, and Development Commands
Run `pip install -r requirements.txt` inside a CUDA 12+ virtual environment to match the pinned PyTorch/Triton stack. Launch training with `python train.py --dataset custom --train_root <path> --val_root <path> --model small --epochs 100`; set `USE_CONFIG` in `train.py` for predefined recipes. Evaluate checkpoints via `python test.py`, aligning `USE_CONFIG` and `SAVE_RESULTS_PATH` for reproducible metrics. Use `pytest -q` for focused regression tests next to the code they cover.

## Coding Style & Naming Conventions
Python code follows 4-space indentation, descriptive docstrings, and `typing` hints when practical. Prefer `snake_case` for functions and variables, `PascalCase` for model classes, and keep config attributes aligned with `config.<area>.<field>` naming (e.g., `config.model.model_size`). Isolate I/O logic under `src/utils/` or `src/data/`, and add new dataset presets as `configs/datasets/<dataset>.py` or `.yaml`.

## Testing Guidelines
Unit tests live beside their modules or under `tests/`. Name test files `test_<feature>.py` and ensure new behaviors have assertions covering success and failure paths. Always run `python test.py` after modifying models, dataloaders, or configs, since it exercises the end-to-end evaluation pipeline found in `src/utils/metrics.py`.

## Commit & Pull Request Guidelines
Write concise, action-oriented commits such as `add new data type .nii.gz` or `加入早停机制`; include body text only when explaining rationale or metrics. Reference issues with `[#123]`. Pull requests should describe affected datasets/configs, list exact commands for reviewers, attach metrics or confusion matrices, and call out external assets needed to reproduce results.

## Security & Configuration Tips
Keep PHI, credentials, and raw datasets out of the repo. Define GPU, AMP, and path requirements inside configs (`config.training.device`, `config.training.amp`). Document any non-default environment variables or secrets in the PR description so others can replay the workflow safely.

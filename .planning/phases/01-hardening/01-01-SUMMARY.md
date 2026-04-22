---
phase: 01-hardening
plan: 01
subsystem: training-pipeline
tags: [environment, collation, glm-ocr]
requires: []
provides: [hardened-training-env, corrected-collator]
affects: [Dockerfile, requirements.txt, src/train.py]
tech-stack: [torch, transformers, flash-attn]
key-files: [Dockerfile, requirements.txt, src/train.py]
decisions:
  - Update environment to Torch 2.10.0 and Transformers 5.5.4 for GLM-OCR compatibility.
  - Implement torch.cat for multimodal tensors in ManualMultimodalDataCollator.
  - Map pixel_values to images in data collator.
metrics:
  duration: 15m
  completed_date: 2024-11-20
---

# Phase 01 Plan 01: Hardening Summary

## One-liner
Hardened the GLM-OCR training pipeline by aligning environment versions and correcting the multimodal data collation strategy to use concatenation instead of stacking.

## Accomplishments
- **Environment Alignment**: Updated `requirements.txt` and `Dockerfile` to use `torch==2.10.0`, `transformers==5.5.4`, and `flash-attn==2.7.0`.
- **Collator Rewrite**: Re-implemented `ManualMultimodalDataCollator` in `src/train.py` to:
    - Use `torch.cat` with `dim=0` for `pixel_values` and `image_grid_thw`, supporting multiple images per sample.
    - Added padding for `rope_deltas`.
    - Mapped `pixel_values` to `images` to match the model's forward pass signature.
- **Verification**: Validated the collator logic via a standalone mock test and performed a syntax check on the training script.

## Deviations from Plan
None - plan executed exactly as written.

## Self-Check: PASSED
- [x] Environment files updated to RESEARCH.md specifications.
- [x] ManualMultimodalDataCollator rewritten with concatenation and rope_deltas padding.
- [x] Commits made atomically for each task.
- [x] Smoke test logic verified.

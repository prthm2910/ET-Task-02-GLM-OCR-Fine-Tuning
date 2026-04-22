# Project: AWS-GLM-OCR-Fine-Tuning

## Objective
Harden and fine-tune the GLM-OCR model training pipeline on Amazon SageMaker, ensuring compatibility with Transformers 5.x and Torch 2.10.0+.

## Tech Stack
- **Model**: `zai-org/GLM-OCR`
- **Framework**: PyTorch 2.10.0+
- **Library**: Transformers 5.5.4+, PEFT 0.14.0+
- **Infrastructure**: AWS SageMaker (Docker-based)
- **Data**: HuggingFace Datasets

## Key Constraints
- Use CONCATENATION for multimodal tensors (`pixel_values`, `image_grid_thw`) on `dim=0`.
- Use PADDING for sequence tensors (`input_ids`, `labels`, `mm_token_type_ids`, `attention_mask`, `rope_deltas`).
- Ensure `rope_deltas` are passed to the model.
- Mapping `pixel_values` to `images` if required by the model signature.

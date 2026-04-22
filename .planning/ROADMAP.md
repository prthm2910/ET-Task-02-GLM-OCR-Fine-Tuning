# Roadmap: GLM-OCR Fine-Tuning

## Phase 0: Research (Completed)
Audit current implementation and identify version mismatches and architecture patterns.
- [x] Audit `src/train.py`
- [x] Identify required tensor formats
- [x] Determine compatible library versions

## Phase 1: Pipeline Hardening
Implement the findings from Research to ensure a zero-error deployment.
**Requirements:** [ENV-01, CODE-01, CODE-02, CODE-03]
**Plans:**
- [x] 01-01-PLAN.md — Environment Hardening (requirements & Docker)
- [ ] 01-02-PLAN.md — Collator & Forward Pass Rewrite
- [ ] 01-03-PLAN.md — Verification & Smoke Test

## Phase 2: Training Execution
Run the full training job on SageMaker.
**Requirements:** [TRAIN-01]
- [ ] Submit SageMaker training job
- [ ] Monitor logs and resource utilization
- [ ] Save and verify model artifacts

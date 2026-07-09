# Vendored slice of zhiyuanyou/DeQA-Score, adapted from transformers==4.36.1 to
# work under our installed transformers==4.57.1. Compatibility patches mirror
# the ones in pyiqa.archs.q_align (which already targets a recent transformers).
# Inference-only: training-only code paths (Fidelity loss, score-distribution
# learning, LoRA / 4-bit / 8-bit) have been dropped.

from .scorer import Scorer

__all__ = ["Scorer"]

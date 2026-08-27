"""MF2.1 training components built on frozen ETP-R1 features."""

from .data import RevealFeatureDataset, collate_reveal_examples
from .losses import RevealOptionLoss, RevealOptionLossConfig
from .model import RevealOptionHeads, RevealOptionOutput, select_topk_options

__all__ = [
    "RevealOptionHeads",
    "RevealOptionLoss",
    "RevealOptionLossConfig",
    "RevealOptionOutput",
    "RevealFeatureDataset",
    "collate_reveal_examples",
    "select_topk_options",
]

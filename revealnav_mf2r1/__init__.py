"""Versioned MF2 revision-1 objective for structured U/A/D alignment."""

from .losses import StructuredUADLoss, factorized_uad_probabilities

__all__ = ["StructuredUADLoss", "factorized_uad_probabilities"]

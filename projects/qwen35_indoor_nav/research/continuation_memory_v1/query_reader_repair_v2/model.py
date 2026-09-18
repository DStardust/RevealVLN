"""Retain early continuation evidence in the training-only query reader.

Identical parameters and causal memory/action path to the first pilot. Each
reader call receives one unpadded query, so every recurrent output is valid.
"""
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('v6_original_memory', Path(__file__).resolve().parent.parent / 'pilot/model.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


class MemoryPolicy(original.MemoryPolicy):
    def reader(self, memory, query_tokens):
        outputs, _ = self.query_gru(self.query_embedding(query_tokens))
        query_summary = outputs.mean(dim=1)
        return self.result_head(torch.cat([memory.flatten(1), query_summary], dim=-1)).squeeze(-1)

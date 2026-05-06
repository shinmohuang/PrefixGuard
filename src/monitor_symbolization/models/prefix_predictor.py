from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PrefixPredictorOutput:
    hidden_states: torch.Tensor
    next_event_logits: torch.Tensor


class SymbolicPrefixPredictor(nn.Module):
    def __init__(self, symbol_embedding_dim: int, hidden_dim: int, num_symbols: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=symbol_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.next_event_head = nn.Linear(hidden_dim, num_symbols)

    def forward(self, symbol_embeddings: torch.Tensor) -> PrefixPredictorOutput:
        hidden_states, _ = self.gru(symbol_embeddings)
        next_event_logits = self.next_event_head(hidden_states)
        return PrefixPredictorOutput(
            hidden_states=hidden_states,
            next_event_logits=next_event_logits,
        )

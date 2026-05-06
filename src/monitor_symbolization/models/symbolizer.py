from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SymbolizerOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    hard_ids: torch.Tensor
    symbol_embeddings: torch.Tensor


class GumbelEventSymbolizer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_symbols: int,
        symbol_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.num_symbols = num_symbols
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_symbols),
        )
        self.symbol_embeddings = nn.Parameter(
            torch.randn(num_symbols, symbol_embedding_dim) * 0.02
        )

    def logits(self, segment_embeddings: torch.Tensor) -> torch.Tensor:
        if segment_embeddings.layout in {torch.sparse_coo, torch.sparse_csr}:
            first_layer = self.projection[0]
            activation = self.projection[1]
            second_layer = self.projection[2]
            hidden = torch.sparse.mm(segment_embeddings, first_layer.weight.t())
            hidden = hidden + first_layer.bias
            hidden = activation(hidden)
            return second_layer(hidden)
        return self.projection(segment_embeddings)

    def deterministic_output(
        self,
        segment_embeddings: torch.Tensor,
        temperature: float,
    ) -> SymbolizerOutput:
        logits = self.logits(segment_embeddings)
        tau = max(float(temperature), 1e-4)
        probs = torch.softmax(logits / tau, dim=-1)
        symbol_embeddings = probs @ self.symbol_embeddings
        hard_ids = logits.argmax(dim=-1)
        return SymbolizerOutput(
            logits=logits,
            probs=probs,
            hard_ids=hard_ids,
            symbol_embeddings=symbol_embeddings,
        )

    def forward(
        self,
        segment_embeddings: torch.Tensor,
        temperature: float,
        hard: bool = False,
    ) -> SymbolizerOutput:
        logits = self.logits(segment_embeddings)
        probs = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        symbol_embeddings = probs @ self.symbol_embeddings
        hard_ids = probs.argmax(dim=-1)
        return SymbolizerOutput(
            logits=logits,
            probs=probs,
            hard_ids=hard_ids,
            symbol_embeddings=symbol_embeddings,
        )

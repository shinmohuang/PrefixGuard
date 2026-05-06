from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DifferentiableAutomatonOutput:
    state_probs: torch.Tensor
    risk_logits: torch.Tensor
    risk_scores: torch.Tensor
    transition_matrices: torch.Tensor


class DifferentiableFiniteStateSurrogate(nn.Module):
    method_name = "differentiable_automaton"

    def __init__(
        self,
        num_symbols: int,
        num_states: int,
    ) -> None:
        super().__init__()
        self.num_symbols = num_symbols
        self.num_states = num_states
        self.initial_state_logits = nn.Parameter(torch.zeros(num_states))
        self.transition_logits = nn.Parameter(torch.zeros(num_symbols, num_states, num_states))
        self.risk_head = nn.Linear(num_states, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.initial_state_logits.fill_(-4.0)
            self.initial_state_logits[0] = 4.0
            self.transition_logits.zero_()
        nn.init.xavier_uniform_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)

    def transition_matrices(self, temperature: float) -> torch.Tensor:
        tau = max(float(temperature), 1e-4)
        return F.softmax(self.transition_logits / tau, dim=-1)

    def initial_state_distribution(self) -> torch.Tensor:
        return F.softmax(self.initial_state_logits, dim=-1)

    def forward(
        self,
        symbol_probs: torch.Tensor,
        transition_temperature: float,
        padding_mask: torch.Tensor | None = None,
    ) -> DifferentiableAutomatonOutput:
        if symbol_probs.ndim == 2:
            if padding_mask is not None:
                raise ValueError("padding_mask is only supported for batched [B, T, K] inputs")
            return self._forward_single_sequence(
                symbol_probs,
                transition_temperature=transition_temperature,
            )
        if symbol_probs.ndim != 3:
            raise ValueError(
                f"Expected symbol_probs with shape [T, K] or [B, T, K], got {tuple(symbol_probs.shape)}"
            )
        if symbol_probs.size(-1) != self.num_symbols:
            raise ValueError(
                f"Expected last dimension {self.num_symbols}, got {symbol_probs.size(-1)}"
            )
        if padding_mask is not None and tuple(padding_mask.shape) != tuple(symbol_probs.shape[:2]):
            raise ValueError(
                "padding_mask must have shape [B, T] matching the first two dimensions of symbol_probs"
            )

        transitions = self.transition_matrices(transition_temperature)
        batch_size, sequence_length, _ = symbol_probs.shape
        state = self.initial_state_distribution().to(symbol_probs).expand(batch_size, -1)
        state_history: list[torch.Tensor] = []
        for timestep in range(sequence_length):
            mixed_transition = torch.einsum("bk,kij->bij", symbol_probs[:, timestep], transitions)
            next_state = torch.bmm(state.unsqueeze(1), mixed_transition).squeeze(1)
            next_state = next_state / next_state.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            if padding_mask is not None:
                mask_t = padding_mask[:, timestep].unsqueeze(-1)
                state = torch.where(mask_t, next_state, state)
            else:
                state = next_state
            state_history.append(state)

        if state_history:
            state_probs = torch.stack(state_history, dim=1)
        else:
            state_probs = symbol_probs.new_zeros((batch_size, 0, self.num_states))
        risk_logits = self.risk_head(state_probs).squeeze(-1)
        risk_scores = torch.sigmoid(risk_logits)
        return DifferentiableAutomatonOutput(
            state_probs=state_probs,
            risk_logits=risk_logits,
            risk_scores=risk_scores,
            transition_matrices=transitions,
        )

    def _forward_single_sequence(
        self,
        symbol_probs: torch.Tensor,
        transition_temperature: float,
    ) -> DifferentiableAutomatonOutput:
        if symbol_probs.size(-1) != self.num_symbols:
            raise ValueError(
                f"Expected last dimension {self.num_symbols}, got {symbol_probs.size(-1)}"
            )

        transitions = self.transition_matrices(transition_temperature)
        state = self.initial_state_distribution().to(symbol_probs).unsqueeze(0)
        state_history: list[torch.Tensor] = []
        for timestep in range(symbol_probs.size(0)):
            mixed_transition = torch.einsum("k,kij->ij", symbol_probs[timestep], transitions)
            state = torch.matmul(state, mixed_transition)
            state = state / state.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            state_history.append(state.squeeze(0))

        if state_history:
            state_probs = torch.stack(state_history, dim=0)
        else:
            state_probs = symbol_probs.new_zeros((0, self.num_states))
        risk_logits = self.risk_head(state_probs).squeeze(-1)
        risk_scores = torch.sigmoid(risk_logits)
        return DifferentiableAutomatonOutput(
            state_probs=state_probs,
            risk_logits=risk_logits,
            risk_scores=risk_scores,
            transition_matrices=transitions,
        )


class FlatPrefixRiskHead(nn.Module):
    method_name = "uncoupled_flat_prefix_head"

    def __init__(
        self,
        num_symbols: int,
        num_states: int,
    ) -> None:
        super().__init__()
        self.num_symbols = num_symbols
        self.num_states = num_states
        self.prefix_projection = nn.Linear(num_symbols, num_states)
        self.risk_head = nn.Linear(num_states, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.prefix_projection.weight)
        nn.init.zeros_(self.prefix_projection.bias)
        nn.init.xavier_uniform_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)

    def _prefix_average(
        self,
        symbol_probs: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if symbol_probs.ndim == 2:
            cumulative = symbol_probs.cumsum(dim=0)
            counts = torch.arange(
                1,
                symbol_probs.size(0) + 1,
                device=symbol_probs.device,
                dtype=symbol_probs.dtype,
            ).unsqueeze(-1)
            return cumulative / counts

        if padding_mask is None:
            cumulative = symbol_probs.cumsum(dim=1)
            counts = torch.arange(
                1,
                symbol_probs.size(1) + 1,
                device=symbol_probs.device,
                dtype=symbol_probs.dtype,
            ).view(1, -1, 1)
            return cumulative / counts

        valid = padding_mask.to(dtype=symbol_probs.dtype).unsqueeze(-1)
        cumulative = (symbol_probs * valid).cumsum(dim=1)
        counts = valid.cumsum(dim=1).clamp_min(1.0)
        return cumulative / counts

    def forward(
        self,
        symbol_probs: torch.Tensor,
        transition_temperature: float,
        padding_mask: torch.Tensor | None = None,
    ) -> DifferentiableAutomatonOutput:
        del transition_temperature
        if symbol_probs.ndim not in {2, 3}:
            raise ValueError(
                f"Expected symbol_probs with shape [T, K] or [B, T, K], got {tuple(symbol_probs.shape)}"
            )
        if symbol_probs.size(-1) != self.num_symbols:
            raise ValueError(
                f"Expected last dimension {self.num_symbols}, got {symbol_probs.size(-1)}"
            )
        if padding_mask is not None and symbol_probs.ndim != 3:
            raise ValueError("padding_mask is only supported for batched [B, T, K] inputs")
        if padding_mask is not None and tuple(padding_mask.shape) != tuple(symbol_probs.shape[:2]):
            raise ValueError(
                "padding_mask must have shape [B, T] matching the first two dimensions of symbol_probs"
            )

        prefix_features = self._prefix_average(symbol_probs, padding_mask=padding_mask)
        state_probs = F.gelu(self.prefix_projection(prefix_features))
        risk_logits = self.risk_head(state_probs).squeeze(-1)
        risk_scores = torch.sigmoid(risk_logits)
        return DifferentiableAutomatonOutput(
            state_probs=state_probs,
            risk_logits=risk_logits,
            risk_scores=risk_scores,
            transition_matrices=symbol_probs.new_zeros((0,)),
        )


def _sinusoidal_position_encoding(
    sequence_length: int,
    hidden_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sequence_length <= 0:
        return torch.empty((0, hidden_dim), device=device, dtype=dtype)
    positions = torch.arange(sequence_length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2, device=device, dtype=dtype)
        * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / hidden_dim)
    )
    encoding = torch.zeros((sequence_length, hidden_dim), device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
    return encoding


class GruPrefixRiskHead(nn.Module):
    method_name = "direct_gru_prefix_head"

    def __init__(
        self,
        num_symbols: int,
        num_states: int,
    ) -> None:
        super().__init__()
        self.num_symbols = num_symbols
        self.num_states = num_states
        self.input_projection = nn.Linear(num_symbols, num_states)
        self.gru = nn.GRU(
            input_size=num_states,
            hidden_size=num_states,
            batch_first=True,
        )
        self.risk_head = nn.Linear(num_states, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)
        nn.init.xavier_uniform_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)

    def forward(
        self,
        symbol_probs: torch.Tensor,
        transition_temperature: float,
        padding_mask: torch.Tensor | None = None,
    ) -> DifferentiableAutomatonOutput:
        del transition_temperature
        if symbol_probs.ndim not in {2, 3}:
            raise ValueError(
                f"Expected symbol_probs with shape [T, K] or [B, T, K], got {tuple(symbol_probs.shape)}"
            )
        if symbol_probs.size(-1) != self.num_symbols:
            raise ValueError(
                f"Expected last dimension {self.num_symbols}, got {symbol_probs.size(-1)}"
            )
        if padding_mask is not None and symbol_probs.ndim != 3:
            raise ValueError("padding_mask is only supported for batched [B, T, K] inputs")
        if padding_mask is not None and tuple(padding_mask.shape) != tuple(symbol_probs.shape[:2]):
            raise ValueError(
                "padding_mask must have shape [B, T] matching the first two dimensions of symbol_probs"
            )

        single_sequence = symbol_probs.ndim == 2
        sequence_batch = symbol_probs.unsqueeze(0) if single_sequence else symbol_probs
        projected_inputs = F.gelu(self.input_projection(sequence_batch))
        hidden_states, _ = self.gru(projected_inputs)
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask.unsqueeze(-1).to(hidden_states.dtype)
        risk_logits = self.risk_head(hidden_states).squeeze(-1)
        risk_scores = torch.sigmoid(risk_logits)

        if single_sequence:
            hidden_states = hidden_states.squeeze(0)
            risk_logits = risk_logits.squeeze(0)
            risk_scores = risk_scores.squeeze(0)
        return DifferentiableAutomatonOutput(
            state_probs=hidden_states,
            risk_logits=risk_logits,
            risk_scores=risk_scores,
            transition_matrices=symbol_probs.new_zeros((0,)),
        )


class TransformerPrefixRiskHead(nn.Module):
    method_name = "direct_transformer_prefix_head"

    def __init__(
        self,
        num_symbols: int,
        num_states: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_symbols = num_symbols
        self.num_states = num_states
        attention_heads = next(
            candidate
            for candidate in (8, 4, 2, 1)
            if num_states % candidate == 0
        )
        self.input_projection = nn.Linear(num_symbols, num_states)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_states,
            nhead=attention_heads,
            dim_feedforward=max(4 * num_states, 64),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.risk_head = nn.Linear(num_states, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)

    def forward(
        self,
        symbol_probs: torch.Tensor,
        transition_temperature: float,
        padding_mask: torch.Tensor | None = None,
    ) -> DifferentiableAutomatonOutput:
        del transition_temperature
        if symbol_probs.ndim not in {2, 3}:
            raise ValueError(
                f"Expected symbol_probs with shape [T, K] or [B, T, K], got {tuple(symbol_probs.shape)}"
            )
        if symbol_probs.size(-1) != self.num_symbols:
            raise ValueError(
                f"Expected last dimension {self.num_symbols}, got {symbol_probs.size(-1)}"
            )
        if padding_mask is not None and symbol_probs.ndim != 3:
            raise ValueError("padding_mask is only supported for batched [B, T, K] inputs")
        if padding_mask is not None and tuple(padding_mask.shape) != tuple(symbol_probs.shape[:2]):
            raise ValueError(
                "padding_mask must have shape [B, T] matching the first two dimensions of symbol_probs"
            )

        single_sequence = symbol_probs.ndim == 2
        sequence_batch = symbol_probs.unsqueeze(0) if single_sequence else symbol_probs
        batch_size, sequence_length, _ = sequence_batch.shape
        projected_inputs = F.gelu(self.input_projection(sequence_batch))
        positional_encoding = _sinusoidal_position_encoding(
            sequence_length,
            self.num_states,
            device=projected_inputs.device,
            dtype=projected_inputs.dtype,
        )
        projected_inputs = projected_inputs + positional_encoding.unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(
                (sequence_length, sequence_length),
                device=projected_inputs.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        encoded_states = self.encoder(
            projected_inputs,
            mask=causal_mask,
            src_key_padding_mask=None if padding_mask is None else ~padding_mask,
        )
        if padding_mask is not None:
            encoded_states = encoded_states * padding_mask.unsqueeze(-1).to(encoded_states.dtype)
        risk_logits = self.risk_head(encoded_states).squeeze(-1)
        risk_scores = torch.sigmoid(risk_logits)

        if single_sequence:
            encoded_states = encoded_states.squeeze(0)
            risk_logits = risk_logits.squeeze(0)
            risk_scores = risk_scores.squeeze(0)
        return DifferentiableAutomatonOutput(
            state_probs=encoded_states,
            risk_logits=risk_logits,
            risk_scores=risk_scores,
            transition_matrices=symbol_probs.new_zeros((0,)),
        )

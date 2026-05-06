"""
This module provides the DFAQueryInterface for querying a trained DFA monitor
for counterfactual risk analysis.

The workflow involves two steps:
1. Bake: Materialize a DFA, state risk table, and encoder state into a
   self-contained query bundle file (.query_bundle.pt) using `bake_query_bundle`.
2. Query: Load the bundle using `DFAQueryInterface.load` and perform
   queries like `state_of`, `query_action`, or `rank_actions`.

Example:
    # 1. Bake (usually done once after training)
    bake_query_bundle(
        checkpoint_path=Path("best_checkpoint.pt"),
        dataset_path=Path("data.jsonl"),
        output_path=Path("bundle.query_bundle.pt")
    )

    # 2. Query
    interface = DFAQueryInterface.load(Path("bundle.query_bundle.pt"))
    prefix = [StepRecord(...), StepRecord(...)]
    result = interface.state_of(prefix)
    print(f"Current state risk: {result.risk}")

    candidates = [StepRecord(...), StepRecord(...)]
    ranked = interface.rank_actions(prefix, candidates)
    for r in ranked:
        print(f"Action {r.index} risk: {r.next_risk}")
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch

from monitor_symbolization.data.io import load_trajectories, resolve_fit_cal_splits
from monitor_symbolization.data.schema import StepRecord
from monitor_symbolization.data.serialization import (
    build_step_payload,
    build_step_view,
    resolve_step_view_dataset_name,
)
from monitor_symbolization.models.encoders import TfidfSegmentEncoder
from monitor_symbolization.models.symbolizer import GumbelEventSymbolizer
from monitor_symbolization.monitor.backends import DfaBackendName
from monitor_symbolization.monitor.evaluation import (
    _fit_state_risk,
    _prefix_sequences,
    induce_dfa_from_symbol_sequences,
    symbolize_trajectories,
)
from monitor_symbolization.monitor.hardening import (
    DEFAULT_HARDENING_STRATEGY,
    HardeningStrategyName,
)

logger = logging.getLogger(__name__)

class DFAQueryBundle(TypedDict):
    transitions: dict[int, dict[int, int]]
    alphabet: tuple[int, ...]
    start_state: int
    sink_state: int | None
    accepting_states: set[int]
    rejecting_states: set[int]
    state_risk: dict[int, float]
    state_support: dict[int, int]
    global_failure_rate: float
    num_symbols: int
    encoder_artifact_state: dict[str, Any]
    symbolizer_state_dict: dict[str, Any]
    config_slice: dict[str, Any]
    provenance: dict[str, Any]

@dataclass(frozen=True)
class StateQueryResult:
    state_id: int
    risk: float | None
    is_sink: bool
    support: int

@dataclass(frozen=True)
class ActionQueryResult:
    next_state_id: int
    next_risk: float | None
    next_is_sink: bool
    next_support: int

@dataclass(frozen=True)
class RankedActionResult:
    index: int
    step: StepRecord
    next_state_id: int
    next_risk: float | None
    next_is_sink: bool
    next_support: int

def save_query_bundle(bundle: DFAQueryBundle, path: Path) -> None:
    """Save a query bundle to a file using torch.save."""
    torch.save(bundle, path)

def load_query_bundle(path: Path) -> DFAQueryBundle:
    """Load a query bundle from a file using torch.load."""
    bundle = torch.load(path, weights_only=False)

    required_keys = {
        "transitions", "alphabet", "start_state", "sink_state",
        "accepting_states", "rejecting_states", "state_risk",
        "state_support", "global_failure_rate", "num_symbols",
        "encoder_artifact_state", "symbolizer_state_dict",
        "config_slice", "provenance"
    }

    missing = required_keys - set(bundle.keys())
    if missing:
        raise ValueError(f"Query bundle at {path} is missing required keys: {missing}")

    return bundle

def bake_query_bundle(
    checkpoint_path: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    dfa_backend: DfaBackendName | None = None,
    horizon: int | None = None,
    fit_split: str = "train",
    cal_split: str = "train",
    derive_train_fit_cal: bool = False,
    train_fit_ratio: float = 0.8,
    train_cal_ratio: float = 0.2,
    protocol_split_seed: int = 42,
    trusted_state_min_count: int = 1,
    state_risk_smoothing_alpha: float = 1.0,
    calibration_bins: int = 10,
    device: str = "cpu",
    hardening_strategy: HardeningStrategyName = DEFAULT_HARDENING_STRATEGY,
    hardening_threshold: float = 0.5,
) -> None:
    """
    Materialize a DFA, state risk table, and encoder state into a self-contained query bundle.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = checkpoint["config"]

    warning_model_type = config.get("warning_model_type")
    if warning_model_type in {"uncoupled-gru", "uncoupled-transformer"}:
        raise ValueError(
            f"DFA extraction is not applicable for warning model type: {warning_model_type}"
        )

    # 2.2 Restore encoder
    encoder_artifact_state = checkpoint.get("encoder_artifact_state")
    if encoder_artifact_state is None:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is missing 'encoder_artifact_state'. "
            "Please re-run evaluation with a newer version of the code to generate this artifact."
        )

    encoder_type = config.get("encoder_type", "tfidf")
    if encoder_type in {"transformer", "hybrid"}:
        raise NotImplementedError(f"Query bundle baking is not yet implemented for encoder type: {encoder_type}")

    encoder = TfidfSegmentEncoder()
    encoder.load_artifact_state(encoder_artifact_state)

    # 2.3 Restore symbolizer
    symbolizer_state_dict = checkpoint["symbolizer_state_dict"]
    num_symbols = config["num_symbols"]
    hidden_dim = config["hidden_dim"]
    symbol_embedding_dim = config["symbol_embedding_dim"]

    symbolizer = GumbelEventSymbolizer(
        input_dim=encoder.output_dim,
        hidden_dim=hidden_dim,
        num_symbols=num_symbols,
        symbol_embedding_dim=symbol_embedding_dim,
    )
    symbolizer.load_state_dict(symbolizer_state_dict)
    symbolizer.to(device)
    symbolizer.eval()

    # 2.4 Resolve splits and symbolize
    trajectories = load_trajectories(dataset_path)
    fit_trajs, cal_trajs, _ = resolve_fit_cal_splits(
        trajectories,
        fit_split=fit_split,
        cal_split=cal_split,
        derive_train_fit_cal=derive_train_fit_cal,
        train_fit_ratio=train_fit_ratio,
        train_cal_ratio=train_cal_ratio,
        protocol_split_seed=protocol_split_seed,
    )

    representation_mode = config.get("representation_mode", "legacy")
    max_observation_lines = config.get("max_observation_lines", 8)
    tau_sym_end = config.get("tau_sym_end", 1.0)

    fit_symbols = symbolize_trajectories(
        fit_trajs,
        encoder=encoder,
        symbolizer=symbolizer,
        device=torch.device(device),
        deterministic=True,
        symbol_temperature=tau_sym_end,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
    )

    # Induce DFA
    if dfa_backend is None:
        dfa_backend = config.get("dfa_backend", "aalpy")

    dfa, _, _ = induce_dfa_from_symbol_sequences(
        train_trajectories=fit_trajs,
        train_symbols=fit_symbols,
        num_symbols=num_symbols,
        dfa_backend=dfa_backend,
    )

    # 2.5 Fit state risk and identify sink
    cal_symbols = symbolize_trajectories(
        cal_trajs,
        encoder=encoder,
        symbolizer=symbolizer,
        device=torch.device(device),
        deterministic=True,
        symbol_temperature=tau_sym_end,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
    )

    if horizon is None:
        horizon = config.get("horizon", 0)

    cal_prefixes, cal_labels, _ = _prefix_sequences(
        cal_trajs,
        cal_symbols,
        horizon=horizon,
    )

    state_risk, state_support, global_failure_rate = _fit_state_risk(
        dfa,
        cal_prefixes,
        cal_labels,
        smoothing_alpha=state_risk_smoothing_alpha,
    )

    # Identify sink state
    sink_state = None
    for state_id, state_transitions in dfa.transitions.items():
        if state_support.get(state_id, 0) == 0:
            if all(target == state_id for target in state_transitions.values()):
                sink_state = state_id
                break

    # 2.7 Assemble DFAQueryBundle
    config_slice = {
        "hidden_dim": hidden_dim,
        "num_symbols": num_symbols,
        "symbol_embedding_dim": symbol_embedding_dim,
        "encoder_type": encoder_type,
        "representation_mode": representation_mode,
        "max_observation_lines": max_observation_lines,
        "step_view_frontend": config.get("step_view_frontend", "inferred"),
        "tau2_refinement_profile": config.get("tau2_refinement_profile"),
        "skillsbench_process_profile": config.get("skillsbench_process_profile"),
        "tau_sym_end": tau_sym_end,
    }

    provenance = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "timestamp": datetime.datetime.now().isoformat(),
        "dfa_backend": dfa_backend,
        "has_dfa": True,
    }

    bundle: DFAQueryBundle = {
        "transitions": dfa.transitions,
        "alphabet": dfa.alphabet,
        "start_state": dfa.start_state,
        "sink_state": sink_state,
        "accepting_states": dfa.accepting_states,
        "rejecting_states": dfa.rejecting_states,
        "state_risk": state_risk,
        "state_support": state_support,
        "global_failure_rate": global_failure_rate,
        "num_symbols": num_symbols,
        "encoder_artifact_state": encoder_artifact_state,
        "symbolizer_state_dict": symbolizer_state_dict,
        "config_slice": config_slice,
        "provenance": provenance,
    }

    save_query_bundle(bundle, output_path)

class DFAQueryInterface:
    def __init__(
        self,
        bundle: DFAQueryBundle,
        encoder: TfidfSegmentEncoder,
        symbolizer: GumbelEventSymbolizer,
        device: str = "cpu",
        ood_fallback_risk: float | None = None,
    ):
        self.bundle = bundle
        self.encoder = encoder
        self.symbolizer = symbolizer
        self.device = device
        self.ood_fallback_risk = (
            ood_fallback_risk
            if ood_fallback_risk is not None
            else bundle["global_failure_rate"]
        )

    @classmethod
    def load(
        cls,
        bundle_path: Path,
        *,
        device: str = "cpu",
        ood_fallback_risk: float | None = None,
    ) -> DFAQueryInterface:
        bundle = load_query_bundle(bundle_path)

        config_slice = bundle["config_slice"]

        encoder = TfidfSegmentEncoder()
        encoder.load_artifact_state(bundle["encoder_artifact_state"])

        symbolizer = GumbelEventSymbolizer(
            input_dim=encoder.output_dim,
            hidden_dim=config_slice["hidden_dim"],
            num_symbols=config_slice["num_symbols"],
            symbol_embedding_dim=config_slice["symbol_embedding_dim"],
        )
        symbolizer.load_state_dict(bundle["symbolizer_state_dict"])
        symbolizer.to(device)
        symbolizer.eval()

        return cls(
            bundle=bundle,
            encoder=encoder,
            symbolizer=symbolizer,
            device=device,
            ood_fallback_risk=ood_fallback_risk,
        )

    def _encode_step(self, step: StepRecord) -> int:
        config = self.bundle["config_slice"]

        payload = build_step_payload(
            step,
            representation_mode=config.get("representation_mode", "legacy"),
            max_observation_lines=config.get("max_observation_lines", 8),
            dataset_name=config.get("step_view_frontend", "inferred"),
            tau2_refinement_profile=config.get("tau2_refinement_profile"),
            skillsbench_process_profile=config.get("skillsbench_process_profile"),
        )

        with torch.no_grad():
            embedding = self.encoder.encode([payload], device=torch.device(self.device))
            outputs = self.symbolizer.deterministic_output(
                embedding.embeddings, temperature=config.get("tau_sym_end", 1.0)
            )
            return outputs.hard_ids[0].item()

    def _replay_prefix(self, prefix: list[StepRecord]) -> int:
        state = self.bundle["start_state"]
        transitions = self.bundle["transitions"]
        for step in prefix:
            symbol = self._encode_step(step)
            state = transitions[state][symbol]
        return state

    def state_of(self, prefix: list[StepRecord]) -> StateQueryResult:
        state_id = self._replay_prefix(prefix)
        risk = self.bundle["state_risk"].get(state_id, self.ood_fallback_risk)
        support = self.bundle["state_support"].get(state_id, 0)
        is_sink = state_id == self.bundle["sink_state"]
        return StateQueryResult(
            state_id=state_id,
            risk=risk,
            is_sink=is_sink,
            support=support,
        )

    def query_action(
        self, prefix: list[StepRecord], candidate_step: StepRecord
    ) -> ActionQueryResult:
        q_t = self._replay_prefix(prefix)
        sigma_prime = self._encode_step(candidate_step)
        next_state_id = self.bundle["transitions"][q_t][sigma_prime]

        next_risk = self.bundle["state_risk"].get(next_state_id, self.ood_fallback_risk)
        next_support = self.bundle["state_support"].get(next_state_id, 0)
        next_is_sink = next_state_id == self.bundle["sink_state"]

        return ActionQueryResult(
            next_state_id=next_state_id,
            next_risk=next_risk,
            next_is_sink=next_is_sink,
            next_support=next_support,
        )

    def rank_actions(
        self, prefix: list[StepRecord], candidate_steps: list[StepRecord]
    ) -> list[RankedActionResult]:
        q_t = self._replay_prefix(prefix)
        results = []
        for i, step in enumerate(candidate_steps):
            sigma_prime = self._encode_step(step)
            next_state_id = self.bundle["transitions"][q_t][sigma_prime]

            next_risk = self.bundle["state_risk"].get(next_state_id, self.ood_fallback_risk)
            next_support = self.bundle["state_support"].get(next_state_id, 0)
            next_is_sink = next_state_id == self.bundle["sink_state"]

            results.append(
                RankedActionResult(
                    index=i,
                    step=step,
                    next_state_id=next_state_id,
                    next_risk=next_risk,
                    next_is_sink=next_is_sink,
                    next_support=next_support,
                )
            )

        # Sort by next_risk ascending, then original index
        results.sort(key=lambda x: (x.next_risk if x.next_risk is not None else 1.1, x.index))
        return results

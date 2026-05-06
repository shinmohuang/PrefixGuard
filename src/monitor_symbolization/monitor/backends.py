from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from aalpy.learning_algs import run_EDSM, run_RPNI

from monitor_symbolization.monitor.rpni import DFA, fit_blue_fringe_rpni


DfaBackendName = Literal["legacy", "aalpy", "aalpy-edsm", "aalpy-rpni"]


class DfaLearnerBackend(Protocol):
    name: DfaBackendName

    def fit(
        self,
        positive_traces: list[list[int]],
        negative_traces: list[list[int]],
        alphabet_size: int,
    ) -> DFA:
        raise NotImplementedError


def adapt_traces_for_aalpy(
    positive_traces: list[list[int]],
    negative_traces: list[list[int]],
) -> list[tuple[tuple[int, ...], bool]]:
    return [
        *((tuple(trace), True) for trace in positive_traces),
        *((tuple(trace), False) for trace in negative_traces),
    ]


def _aalpy_dfa_to_internal(learned_dfa, alphabet_size: int) -> DFA:
    alphabet = tuple(range(alphabet_size))
    states_by_external_id = {state.state_id: state for state in learned_dfa.states}
    ordered_external_ids = sorted(states_by_external_id)
    state_id_map = {external_id: index for index, external_id in enumerate(ordered_external_ids)}

    transitions: dict[int, dict[int, int]] = {}
    accepting_states: set[int] = set()
    rejecting_states: set[int] = set()
    sink_state = len(state_id_map)

    for external_id in ordered_external_ids:
        state = states_by_external_id[external_id]
        internal_state_id = state_id_map[external_id]
        if state.is_accepting:
            accepting_states.add(internal_state_id)
        else:
            rejecting_states.add(internal_state_id)

        transitions[internal_state_id] = {}
        for symbol in alphabet:
            target = state.transitions.get(symbol)
            if target is None:
                transitions[internal_state_id][symbol] = sink_state
                continue
            transitions[internal_state_id][symbol] = state_id_map[target.state_id]

    transitions[sink_state] = {symbol: sink_state for symbol in alphabet}
    rejecting_states.add(sink_state)
    return DFA(
        start_state=state_id_map[learned_dfa.initial_state.state_id],
        transitions=transitions,
        accepting_states=accepting_states,
        rejecting_states=rejecting_states,
        alphabet=alphabet,
    )


@dataclass(frozen=True)
class LegacyDfaBackend:
    name: DfaBackendName = "legacy"

    def fit(
        self,
        positive_traces: list[list[int]],
        negative_traces: list[list[int]],
        alphabet_size: int,
    ) -> DFA:
        return fit_blue_fringe_rpni(
            positive_traces=positive_traces,
            negative_traces=negative_traces,
            alphabet_size=alphabet_size,
        )


@dataclass(frozen=True)
class AalpyDfaBackend:
    name: DfaBackendName = "aalpy"
    algorithm: Literal["edsm", "rpni"] = "edsm"

    def fit(
        self,
        positive_traces: list[list[int]],
        negative_traces: list[list[int]],
        alphabet_size: int,
    ) -> DFA:
        data = adapt_traces_for_aalpy(
            positive_traces=positive_traces,
            negative_traces=negative_traces,
        )
        if self.algorithm == "edsm":
            learned_dfa = run_EDSM(
                data=data,
                automaton_type="dfa",
                print_info=False,
            )
        else:
            learned_dfa = run_RPNI(
                data=data,
                automaton_type="dfa",
                algorithm="gsm",
                print_info=False,
            )
        if learned_dfa is None:
            raise ValueError(
                f"AALpy {self.algorithm.upper()} could not induce a deterministic DFA from the provided traces"
            )
        return _aalpy_dfa_to_internal(learned_dfa=learned_dfa, alphabet_size=alphabet_size)


_BACKENDS: dict[DfaBackendName, DfaLearnerBackend] = {
    "legacy": LegacyDfaBackend(),
    "aalpy": AalpyDfaBackend(name="aalpy", algorithm="edsm"),
    "aalpy-edsm": AalpyDfaBackend(name="aalpy-edsm", algorithm="edsm"),
    "aalpy-rpni": AalpyDfaBackend(name="aalpy-rpni", algorithm="rpni"),
}


def get_dfa_backend(name: DfaBackendName) -> DfaLearnerBackend:
    try:
        return _BACKENDS[name]
    except KeyError as error:
        raise ValueError(f"Unsupported DFA backend: {name}") from error


def available_dfa_backends() -> tuple[DfaBackendName, ...]:
    return tuple(_BACKENDS)


def fit_dfa_with_backend(
    positive_traces: list[list[int]],
    negative_traces: list[list[int]],
    alphabet_size: int,
    backend: DfaBackendName = "legacy",
) -> DFA:
    return get_dfa_backend(backend).fit(
        positive_traces=positive_traces,
        negative_traces=negative_traces,
        alphabet_size=alphabet_size,
    )

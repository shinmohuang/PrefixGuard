from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

from monitor_symbolization.monitor.pta import PrefixTreeAcceptor, PTAState, build_pta


@dataclass
class DFA:
    start_state: int
    transitions: dict[int, dict[int, int]]
    accepting_states: set[int]
    rejecting_states: set[int]
    alphabet: tuple[int, ...]

    def transition(self, state: int, symbol: int) -> int:
        return self.transitions[state][symbol]

    @property
    def state_count(self) -> int:
        return len(self.transitions)


def _incoming_edges(states: dict[int, PTAState]) -> dict[int, list[tuple[int, int]]]:
    incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for source_state, state in states.items():
        for symbol, target_state in state.transitions.items():
            incoming[target_state].append((source_state, symbol))
    return incoming


def _merge_terminal_labels(left: str, right: str) -> str | None:
    if left == "UNKNOWN":
        return right
    if right == "UNKNOWN":
        return left
    if left == right:
        return left
    return None


def _fold_states(states: dict[int, PTAState], target: int, source: int) -> bool:
    if target == source:
        return True
    merged_label = _merge_terminal_labels(
        states[target].terminal_label,
        states[source].terminal_label,
    )
    if merged_label is None:
        return False
    states[target].terminal_label = merged_label

    for symbol, source_child in list(states[source].transitions.items()):
        if symbol in states[target].transitions:
            target_child = states[target].transitions[symbol]
            if not _fold_states(states, target_child, source_child):
                return False
        else:
            states[target].transitions[symbol] = source_child

    incoming = _incoming_edges(states)
    for parent, symbol in incoming[source]:
        states[parent].transitions[symbol] = target
    del states[source]
    return True


def _attempt_merge(
    pta: PrefixTreeAcceptor,
    red_state: int,
    blue_state: int,
) -> PrefixTreeAcceptor | None:
    candidate = PrefixTreeAcceptor(
        start_state=pta.start_state,
        states=deepcopy(pta.states),
    )
    if not _fold_states(candidate.states, red_state, blue_state):
        return None
    return candidate


def _children_of_red(pta: PrefixTreeAcceptor, red_states: set[int]) -> set[int]:
    blue_states = set()
    for red_state in red_states:
        blue_states.update(pta.states[red_state].transitions.values())
    return blue_states - red_states


def _edsm_score(before: PrefixTreeAcceptor, after: PrefixTreeAcceptor) -> int:
    return len(before.states) - len(after.states)


def _pta_to_dfa(pta: PrefixTreeAcceptor, alphabet: tuple[int, ...]) -> DFA:
    transitions: dict[int, dict[int, int]] = {}
    accepting_states: set[int] = set()
    rejecting_states: set[int] = set()

    for state_id, state in pta.states.items():
        transitions[state_id] = dict(state.transitions)
        if state.terminal_label == "ACCEPT":
            accepting_states.add(state_id)
        elif state.terminal_label == "REJECT":
            rejecting_states.add(state_id)

    sink_state = max(transitions) + 1 if transitions else 0
    transitions[sink_state] = {symbol: sink_state for symbol in alphabet}
    rejecting_states.add(sink_state)
    for state_id in list(transitions):
        if state_id == sink_state:
            continue
        for symbol in alphabet:
            transitions[state_id].setdefault(symbol, sink_state)
    return DFA(
        start_state=pta.start_state,
        transitions=transitions,
        accepting_states=accepting_states,
        rejecting_states=rejecting_states,
        alphabet=alphabet,
    )


def fit_blue_fringe_rpni(
    positive_traces: list[list[int]],
    negative_traces: list[list[int]],
    alphabet_size: int,
) -> DFA:
    alphabet = tuple(range(alphabet_size))
    pta = build_pta(positive_traces=positive_traces, negative_traces=negative_traces)
    red_states = {pta.start_state}

    while True:
        blue_states = _children_of_red(pta, red_states)
        if not blue_states:
            break

        selected_blue = min(blue_states)
        best_candidate = None
        best_score = -1
        for red_state in sorted(red_states):
            candidate = _attempt_merge(pta, red_state, selected_blue)
            if candidate is None:
                continue
            score = _edsm_score(pta, candidate)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            red_states.add(selected_blue)
        else:
            pta = best_candidate
            red_states = {state for state in red_states if state in pta.states}
    return _pta_to_dfa(pta, alphabet=alphabet)

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PTAState:
    state_id: int
    terminal_label: str = "UNKNOWN"
    transitions: dict[int, int] = field(default_factory=dict)


@dataclass
class PrefixTreeAcceptor:
    start_state: int
    states: dict[int, PTAState]


def build_pta(
    positive_traces: list[list[int]],
    negative_traces: list[list[int]],
) -> PrefixTreeAcceptor:
    states = {0: PTAState(state_id=0)}
    next_state_id = 1

    def add_trace(trace: list[int], label: str) -> None:
        nonlocal next_state_id
        current_state = 0
        for symbol in trace:
            transitions = states[current_state].transitions
            if symbol not in transitions:
                transitions[symbol] = next_state_id
                states[next_state_id] = PTAState(state_id=next_state_id)
                next_state_id += 1
            current_state = transitions[symbol]
        current_label = states[current_state].terminal_label
        if current_label not in ("UNKNOWN", label):
            raise ValueError(
                f"Inconsistent trace labels for prefix state {current_state}: {current_label} vs {label}"
            )
        states[current_state].terminal_label = label

    for trace in positive_traces:
        add_trace(trace, "ACCEPT")
    for trace in negative_traces:
        add_trace(trace, "REJECT")

    return PrefixTreeAcceptor(start_state=0, states=states)

from __future__ import annotations

import json
from pathlib import Path

import typer

from monitor_symbolization.data.io import load_trajectories
from monitor_symbolization.data.prefixes import summarize_prefix_dataset

app = typer.Typer(no_args_is_help=True)


@app.callback()
def callback() -> None:
    """Monitor-aware symbolization experiment utilities."""


@app.command("inspect")
def inspect_dataset(path: Path) -> None:
    trajectories = load_trajectories(path)
    summary = summarize_prefix_dataset(trajectories)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    app()

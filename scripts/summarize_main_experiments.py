from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/main_experiments.json")
DEFAULT_OUTPUT_JSON = Path("outputs/analysis/main_experiments_summary.json")
DEFAULT_OUTPUT_CSV = Path("outputs/analysis/main_experiments_summary.csv")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict[str, Any], dotted_key: str) -> float | None:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    return float(current)


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize reproduced main experiment metrics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def summarize(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in config["runs"]:
        values: dict[str, list[float]] = {
            "soft_auprc": [],
            "soft_auroc": [],
            "soft_calibration_error": [],
            "discrete_auprc": [],
            "discrete_available": [],
        }
        seed_rows = []
        for seed in run["seeds"]:
            train_dir = Path(run["output_template"].format(seed=seed))
            metrics_path = Path(f"{train_dir}_test_locked") / "best_metrics.json"
            if not metrics_path.exists():
                seed_rows.append(
                    {
                        "seed": seed,
                        "metrics_path": str(metrics_path),
                        "status": "missing",
                    }
                )
                continue
            payload = _load_json(metrics_path)
            seed_payload = {
                "seed": seed,
                "metrics_path": str(metrics_path),
                "status": "ok",
            }
            for key in values:
                value = _metric(payload, f"summary.{key}")
                if value is None:
                    value = _metric(payload, key.replace("soft_", "soft_metrics.").replace("discrete_", "discrete_metrics."))
                if value is not None:
                    values[key].append(value)
                    seed_payload[key] = value
            seed_rows.append(seed_payload)
        row = {
            "id": run["id"],
            "family": run["family"],
            "method": run["method"],
            "num_expected_seeds": len(run["seeds"]),
            "num_completed_seeds": sum(1 for seed_row in seed_rows if seed_row["status"] == "ok"),
            "seeds": seed_rows,
        }
        for key, metric_values in values.items():
            row[f"{key}_mean"] = _mean(metric_values)
            row[f"{key}_std"] = _std(metric_values)
        rows.append(row)
    return {"runs": rows}


def _write_csv(summary: dict[str, Any], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "family",
        "method",
        "num_completed_seeds",
        "num_expected_seeds",
        "soft_auprc_mean",
        "soft_auprc_std",
        "soft_auroc_mean",
        "soft_auroc_std",
        "soft_calibration_error_mean",
        "soft_calibration_error_std",
        "discrete_auprc_mean",
        "discrete_auprc_std",
        "discrete_available_mean",
        "discrete_available_std",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["runs"]:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    args = parse_args()
    config = _load_json(args.config)
    summary = summarize(config)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(summary, args.output_csv)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

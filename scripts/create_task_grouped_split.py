from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic task-group protocol artifacts."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--protocol-mode",
        choices=[
            "outer-train-internal-fit-cal",
            "top-level-fit-cal-val-test",
            "outer-train-val-test",
            "trajectory-stratified",
        ],
        default="outer-train-internal-fit-cal",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--train-fit-ratio", type=float, default=0.8)
    parser.add_argument("--train-cal-ratio", type=float, default=0.2)
    parser.add_argument("--fit-ratio", type=float, default=0.7)
    parser.add_argument("--cal-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def _bucket(task_id: str, seed: int) -> float:
    payload = f"{seed}:{task_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") / 2**64


def _assign_top_level_split(
    task_id: str,
    *,
    seed: int,
    fit_ratio: float,
    cal_ratio: float,
    val_ratio: float,
) -> str:
    bucket = _bucket(task_id, seed=seed)
    if bucket < fit_ratio:
        return "fit"
    if bucket < fit_ratio + cal_ratio:
        return "cal"
    if bucket < fit_ratio + cal_ratio + val_ratio:
        return "val"
    return "test"


def _assign_train_internal_split(
    task_id: str,
    *,
    seed: int,
    fit_ratio: float,
) -> str:
    return "fit" if _bucket(task_id, seed=seed) < fit_ratio else "cal"


def _assign_outer_split(
    task_id: str,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> str:
    bucket = _bucket(task_id, seed=seed)
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def _write_summary(summary: dict, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _copy_with_train_internal_split(args: argparse.Namespace) -> dict:
    total = args.train_fit_ratio + args.train_cal_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"train_fit_ratio + train_cal_ratio must sum to 1.0, got {total}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    task_to_inner_split: dict[str, str] = {}
    outer_split_counts: dict[str, int] = {}
    inner_task_counts = {"fit": 0, "cal": 0}
    inner_trajectory_counts = {"fit": 0, "cal": 0}

    with args.input.open("r", encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            record = json.loads(line)
            outer_split = str(record.get("split", "train"))
            outer_split_counts[outer_split] = outer_split_counts.get(outer_split, 0) + 1
            if outer_split == args.train_split:
                task_id = str(record["task_id"])
                inner_split = task_to_inner_split.get(task_id)
                if inner_split is None:
                    inner_split = _assign_train_internal_split(
                        task_id,
                        seed=args.seed,
                        fit_ratio=args.train_fit_ratio,
                    )
                    task_to_inner_split[task_id] = inner_split
                    inner_task_counts[inner_split] += 1
                metadata = dict(record.get("metadata", {}))
                metadata["protocol_train_split"] = inner_split
                record["metadata"] = metadata
                inner_trajectory_counts[inner_split] += 1
            dst.write(json.dumps(record, ensure_ascii=True) + "\n")

    if not inner_trajectory_counts["fit"] or not inner_trajectory_counts["cal"]:
        raise ValueError(
            "Train-internal fit/cal derivation produced an empty partition; "
            f"fit={inner_trajectory_counts['fit']}, cal={inner_trajectory_counts['cal']}"
        )

    return {
        "protocol_mode": args.protocol_mode,
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "train_split": args.train_split,
        "ratios": {
            "train_fit": args.train_fit_ratio,
            "train_cal": args.train_cal_ratio,
        },
        "outer_split_counts": outer_split_counts,
        "train_internal_task_counts": inner_task_counts,
        "train_internal_trajectory_counts": inner_trajectory_counts,
        "num_unique_train_tasks": len(task_to_inner_split),
    }


def _write_top_level_protocol_split(args: argparse.Namespace) -> dict:
    total = args.fit_ratio + args.cal_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    task_to_split: dict[str, str] = {}
    split_counts = {"fit": 0, "cal": 0, "val": 0, "test": 0}
    task_counts = {"fit": 0, "cal": 0, "val": 0, "test": 0}

    with args.input.open("r", encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            record = json.loads(line)
            task_id = str(record["task_id"])
            split = task_to_split.get(task_id)
            if split is None:
                split = _assign_top_level_split(
                    task_id,
                    seed=args.seed,
                    fit_ratio=args.fit_ratio,
                    cal_ratio=args.cal_ratio,
                    val_ratio=args.val_ratio,
                )
                task_to_split[task_id] = split
                task_counts[split] += 1
            record["split"] = split
            dst.write(json.dumps(record, ensure_ascii=True) + "\n")
            split_counts[split] += 1

    return {
        "protocol_mode": args.protocol_mode,
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "ratios": {
            "fit": args.fit_ratio,
            "cal": args.cal_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "task_counts": task_counts,
        "trajectory_counts": split_counts,
        "num_unique_tasks": len(task_to_split),
    }


def _write_trajectory_stratified_split(args: argparse.Namespace) -> dict:
    """Assign fit/cal/val/test by shuffling within each task group.

    Every task_id appears in all four splits.  Within a task the trajectories
    are shuffled with `args.seed` and then assigned in proportion to
    fit/cal/val/test ratios.  Guarantees at least 1 trajectory per split per
    task (requires each task to have >= 4 trajectories).
    """
    total = args.fit_ratio + args.cal_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    # --- first pass: bucket records by task_id (keeps memory proportional) ---
    task_records: dict[str, list[dict]] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8") as src:
        for line in src:
            record = json.loads(line)
            task_id = str(record["task_id"])
            if task_id not in task_records:
                task_records[task_id] = []
            task_records[task_id].append(record)

    # --- assign splits within each task ---
    rng = random.Random(args.seed)
    split_counts = {"fit": 0, "cal": 0, "val": 0, "test": 0}
    tasks_per_split: dict[str, set[str]] = {s: set() for s in split_counts}
    all_assigned: list[dict] = []

    for task_id in sorted(task_records.keys()):  # deterministic iteration order
        records = task_records[task_id]
        rng.shuffle(records)
        n = len(records)
        if n < 4:
            raise ValueError(
                f"Task {task_id!r} has only {n} trajectories; "
                "trajectory-stratified requires >= 4 per task."
            )

        # Compute per-split counts using rounding; give remainder to fit.
        n_cal = max(1, round(n * args.cal_ratio))
        n_val = max(1, round(n * args.val_ratio))
        n_test = max(1, round(n * args.test_ratio))
        n_fit = n - n_cal - n_val - n_test
        if n_fit < 1:
            raise ValueError(
                f"Task {task_id!r}: n={n} too small to assign >=1 to every split."
            )

        labels = ["fit"] * n_fit + ["cal"] * n_cal + ["val"] * n_val + ["test"] * n_test
        assert len(labels) == n

        for record, split in zip(records, labels):
            record["split"] = split
            split_counts[split] += 1
            tasks_per_split[split].add(task_id)
            all_assigned.append(record)

    # Shuffle output so tasks are interleaved (avoids task-block bias in SGD).
    rng.shuffle(all_assigned)

    with args.output.open("w", encoding="utf-8") as dst:
        for record in all_assigned:
            dst.write(json.dumps(record, ensure_ascii=True) + "\n")

    # Verify every task appears in every split.
    all_task_ids = set(task_records.keys())
    for split, seen in tasks_per_split.items():
        missing = all_task_ids - seen
        if missing:
            raise RuntimeError(
                f"Split {split!r} is missing tasks: {sorted(missing)}"
            )

    return {
        "protocol_mode": args.protocol_mode,
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "ratios": {
            "fit": args.fit_ratio,
            "cal": args.cal_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "task_counts": {s: len(tasks_per_split[s]) for s in split_counts},
        "trajectory_counts": split_counts,
        "num_unique_tasks": len(task_records),
    }


def _write_outer_train_val_test_split(args: argparse.Namespace) -> dict:
    train_ratio = 1.0 - args.val_ratio - args.test_ratio
    if train_ratio <= 0.0:
        raise ValueError(
            f"val_ratio + test_ratio must be less than 1.0, got {args.val_ratio + args.test_ratio}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    task_to_split: dict[str, str] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}
    task_counts = {"train": 0, "val": 0, "test": 0}

    with args.input.open("r", encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            record = json.loads(line)
            task_id = str(record["task_id"])
            split = task_to_split.get(task_id)
            if split is None:
                split = _assign_outer_split(
                    task_id,
                    seed=args.seed,
                    train_ratio=train_ratio,
                    val_ratio=args.val_ratio,
                )
                task_to_split[task_id] = split
                task_counts[split] += 1
            record["split"] = split
            dst.write(json.dumps(record, ensure_ascii=True) + "\n")
            split_counts[split] += 1

    return {
        "protocol_mode": args.protocol_mode,
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "ratios": {
            "train": train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "task_counts": task_counts,
        "trajectory_counts": split_counts,
        "num_unique_tasks": len(task_to_split),
    }


def main() -> None:
    args = parse_args()
    if args.protocol_mode == "outer-train-internal-fit-cal":
        summary = _copy_with_train_internal_split(args)
    elif args.protocol_mode == "outer-train-val-test":
        summary = _write_outer_train_val_test_split(args)
    elif args.protocol_mode == "trajectory-stratified":
        summary = _write_trajectory_stratified_split(args)
    else:
        summary = _write_top_level_protocol_split(args)
    _write_summary(summary, args.summary_output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

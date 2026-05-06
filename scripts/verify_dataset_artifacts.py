import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExpectedArtifact:
    family: str
    path: Path
    rows: int
    sha256: str


def expected_artifacts() -> dict[str, ExpectedArtifact]:
    return {
        "webarena": ExpectedArtifact(
            family="webarena",
            path=Path("data/interim/webarena/source_raw/execution_union_v1_v2_source_raw_labeled_split.jsonl"),
            rows=4430,
            sha256="756ac7d4e9b5797e69bea90e5ffd27ca85ebd1752b5a74f566eb050e4dcf3819",
        ),
        "tau2": ExpectedArtifact(
            family="tau2",
            path=Path("data/interim/tau2_bench/source_raw/results_final_source_raw_outer_train_val_test.jsonl"),
            rows=10832,
            sha256="002c1a34d290b03c916b354c7b839a8ffefc4c90838f647c711e0689a97c3002",
        ),
        "terminalbench": ExpectedArtifact(
            family="terminalbench",
            path=Path(
                "data/interim/terminalbench/source_raw/"
                "terminalbench_trajectories_source_raw_traj_split_manifest.jsonl"
            ),
            rows=34397,
            sha256="bdb787c4daff4f76719d509da59865170940e98f027c0fa450c0a6f00aab0058",
        ),
        "skillsbench": ExpectedArtifact(
            family="skillsbench",
            path=Path("data/interim/skillsbench/source_raw/full_repo_main_traces_source_raw_split_manifest.jsonl"),
            rows=10951,
            sha256="a08b8dccded2fb65f0bb822fad0c24c360fdc0bc60358f3367a66be1cfdd5547",
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify canonical reconstructed dataset artifacts.")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(expected_artifacts()),
        default=sorted(expected_artifacts()),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable verification payload.",
    )
    return parser.parse_args()


def verify(root: Path, families: list[str]) -> dict:
    expected = expected_artifacts()
    results = {}
    ok = True
    for family in families:
        artifact = expected[family]
        path = root / artifact.path
        exists = path.exists()
        row_count = _count_rows(path) if exists else None
        checksum = _sha256(path) if exists else None
        family_ok = (
            exists
            and row_count == artifact.rows
            and checksum == artifact.sha256
        )
        ok = ok and family_ok
        results[family] = {
            "path": str(artifact.path),
            "exists": exists,
            "expected_rows": artifact.rows,
            "rows": row_count,
            "expected_sha256": artifact.sha256,
            "sha256": checksum,
            "ok": family_ok,
        }
    return {"ok": ok, "artifacts": results}


def main() -> None:
    args = parse_args()
    payload = verify(args.root, args.families)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for family, result in payload["artifacts"].items():
            status = "OK" if result["ok"] else "FAIL"
            print(f"{status} {family} {result['path']}")
        if not payload["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

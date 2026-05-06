from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any


REDACTED_PATH = "[redacted_path]"
REDACTED_EVAL_PROTOCOL = "[redacted_instruction]"
REDACTED_EVAL_FILE = "[redacted_file]"
_STEP_FIELDS = ("context", "action_text", "result_text")

_DROP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("explicit_verifier_key", re.compile(r"(?i)\bresult\.verifier_result\b")),
    ("explicit_verifier_key", re.compile(r"(?i)\bsource_result\b")),
    ("explicit_verifier_key", re.compile(r"(?i)\braw_reward\b")),
    ("explicit_verifier_key", re.compile(r"(?i)\blabel_origin\s*=\s*result\.verifier_result\.rewards\.reward\b")),
    ("explicit_verifier_key", re.compile(r"(?i)\breward\s*=\s*(?:0(?:\.0+)?|1(?:\.0+)?)\b")),
    ("explicit_verifier_key", re.compile(r"(?i)test-stdout\.txt")),
    ("explicit_verifier_key", re.compile(r"(?i)reward\.txt")),
)

_REWRITE_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "evaluator_protocol",
        re.compile(r"(?im)\b(?:the\s+)?(?:verifier|evaluator)\s+will\b[^\n]*"),
        " " + REDACTED_EVAL_PROTOCOL,
    ),
    (
        "verifier_path",
        re.compile(r"(?i)/logs/verifier(?:/[^\s\"'`]+)?"),
        REDACTED_PATH,
    ),
    (
        "verifier_path",
        re.compile(r"(?i)/root/verifier(?:-skills)?(?:/[^\s\"'`]+)?"),
        REDACTED_PATH,
    ),
    (
        "verifier_path",
        re.compile(r"(?i)\bverifier-skills\b"),
        REDACTED_PATH,
    ),
    (
        "evaluator_test",
        re.compile(r"(?i)/tests/test_outputs\.py"),
        REDACTED_EVAL_FILE,
    ),
)

_RESIDUAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("verifier_protocol_residual", re.compile(r"(?i)\b(?:the\s+)?verifier\s+will\b")),
    ("evaluator_protocol_residual", re.compile(r"(?i)\b(?:the\s+)?evaluator\s+will\b")),
    ("verifier_path_residual", re.compile(r"(?i)/logs/verifier(?:/|$)")),
    ("verifier_path_residual", re.compile(r"(?i)/root/verifier(?:-skills)?(?:/|$)")),
    ("evaluator_test_residual", re.compile(r"(?i)/tests/test_outputs\.py")),
)


def classify_clean_monitor_issues(text: str | None) -> dict[str, Any]:
    source = text or ""
    matched_categories: set[str] = set()
    needs_drop = False
    needs_rewrite = False

    for category, pattern in _DROP_PATTERNS:
        if pattern.search(source):
            matched_categories.add(category)
            needs_drop = True

    for category, pattern, _ in _REWRITE_RULES:
        if pattern.search(source):
            matched_categories.add(category)
            needs_rewrite = True

    return {
        "matched_categories": tuple(sorted(matched_categories)),
        "needs_drop": needs_drop,
        "needs_rewrite": needs_rewrite,
    }


def clean_skillsbench_monitor_text(text: str | None) -> str:
    cleaned, _ = clean_skillsbench_monitor_text_with_stats(text)
    return cleaned


def clean_skillsbench_monitor_text_with_stats(text: str | None) -> tuple[str, Counter[str]]:
    source = text or ""
    cleaned = source
    replacements: Counter[str] = Counter()
    for category, pattern, replacement in _REWRITE_RULES:
        cleaned, count = pattern.subn(replacement, cleaned)
        if count:
            replacements[category] += count
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), replacements


def clean_skillsbench_monitor_trajectory(record: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    drop_categories: set[str] = set()
    rewrite_categories: set[str] = set()
    replacement_counts: Counter[str] = Counter()
    residual_counts: Counter[str] = Counter()
    affected_fields = 0

    for step in record.get("steps", []):
        for field in _STEP_FIELDS:
            issues = classify_clean_monitor_issues(step.get(field))
            if issues["needs_drop"]:
                drop_categories.update(issues["matched_categories"])
            if issues["needs_rewrite"]:
                rewrite_categories.update(issues["matched_categories"])

    if drop_categories:
        return None, {
            "dropped": True,
            "drop_categories": tuple(sorted(drop_categories)),
            "rewrite_categories": tuple(sorted(rewrite_categories)),
            "replacement_counts": dict(replacement_counts),
            "affected_fields": affected_fields,
            "residual_counts": dict(residual_counts),
        }

    cleaned_record = copy.deepcopy(record)
    for step in cleaned_record.get("steps", []):
        for field in _STEP_FIELDS:
            source = step.get(field)
            cleaned, counts = clean_skillsbench_monitor_text_with_stats(source)
            if counts:
                step[field] = cleaned
                replacement_counts.update(counts)
                affected_fields += 1

    for step in cleaned_record.get("steps", []):
        for field in _STEP_FIELDS:
            text = step.get(field) or ""
            for category, pattern in _RESIDUAL_PATTERNS:
                if pattern.search(text):
                    residual_counts[category] += 1

    metadata = cleaned_record.setdefault("metadata", {})
    metadata["monitor_clean_profile"] = "skillsbench-clean-monitor-v1"
    metadata["monitor_clean_affected_fields"] = affected_fields
    metadata["monitor_clean_rewrite_categories"] = sorted(rewrite_categories)

    return cleaned_record, {
        "dropped": False,
        "drop_categories": tuple(),
        "rewrite_categories": tuple(sorted(rewrite_categories)),
        "replacement_counts": dict(replacement_counts),
        "affected_fields": affected_fields,
        "residual_counts": dict(residual_counts),
    }

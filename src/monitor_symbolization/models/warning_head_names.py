from __future__ import annotations

WARNING_MODEL_CANONICAL_CHOICES: tuple[str, ...] = (
    "soft-fsm",
    "symbol-flat",
    "symbol-gru",
    "symbol-transformer",
)

WARNING_MODEL_LEGACY_ALIASES: dict[str, str] = {
    "coupled": "soft-fsm",
    "uncoupled-flat": "symbol-flat",
    "uncoupled-gru": "symbol-gru",
    "uncoupled-transformer": "symbol-transformer",
}

WARNING_MODEL_ALIASES: dict[str, str] = {
    **{name: name for name in WARNING_MODEL_CANONICAL_CHOICES},
    **WARNING_MODEL_LEGACY_ALIASES,
}

WARNING_MODEL_CHOICES: tuple[str, ...] = (
    *WARNING_MODEL_CANONICAL_CHOICES,
    *WARNING_MODEL_LEGACY_ALIASES.keys(),
)


def normalize_warning_model_type(warning_model_type: str) -> str:
    try:
        return WARNING_MODEL_ALIASES[warning_model_type]
    except KeyError as error:
        supported = ", ".join(WARNING_MODEL_CHOICES)
        raise ValueError(
            f"Unsupported warning_model_type: {warning_model_type!r}. "
            f"Supported values: {supported}"
        ) from error

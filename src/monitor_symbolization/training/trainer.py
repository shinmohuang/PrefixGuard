from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from monitor_symbolization.data.schema import TrajectoryRecord
from monitor_symbolization.data.scrambled_index import trajectory_step_count
from monitor_symbolization.data.serialization import (
    summarize_representation_stats,
)
from monitor_symbolization.models.encoders import (
    BaseSegmentEncoder,
    DEFAULT_TRANSFORMER_MAX_LENGTH,
    DEFAULT_TRANSFORMER_MODEL,
    HybridStepEncoder,
    TfidfSegmentEncoder,
    TransformerSegmentEncoder,
)
from monitor_symbolization.models.prefix_predictor import SymbolicPrefixPredictor
from monitor_symbolization.models.symbolizer import GumbelEventSymbolizer
from monitor_symbolization.models.warning_head_names import normalize_warning_model_type
from monitor_symbolization.monitor.backends import DfaBackendName
from monitor_symbolization.monitor.evaluation import evaluate_symbolic_monitor
from monitor_symbolization.monitor.hardening import (
    DEFAULT_HARDENING_STRATEGY,
    HardeningStrategyName,
)
from monitor_symbolization.runtime_cache import (
    RuntimeCache,
    encode_trajectories,
    iter_trajectory_batches,
    precompute_trajectory_embeddings,
    split_encoded_trajectories,
)
from monitor_symbolization.training.losses import (
    compactness_loss,
    soft_target_cross_entropy,
    supervised_contrastive_loss,
)

_COLLAPSED_SYMBOL_FRONTEND_ERROR = (
    "No consistent positive/negative full traces remain after ambiguity filtering; "
    "the symbolic front-end is too collapsed for exact DFA induction."
)


@dataclass
class TrainingConfig:
    encoder_type: str = "tfidf"
    representation_mode: str = "reduced-dense"
    step_view_frontend: str = "inferred"
    tau2_refinement_profile: str | None = None
    skillsbench_process_profile: str | None = None
    step_view_text_mode: str = "full"
    transformer_stepview_view: str = "lexical"
    encoder_name: str = DEFAULT_TRANSFORMER_MODEL
    fine_tune_encoder: bool = False
    encoder_max_length: int = DEFAULT_TRANSFORMER_MAX_LENGTH
    encoder_batch_size: int = 1
    encoder_max_features: int = 4096
    tfidf_sparse_runtime: bool = False
    tfidf_metadata_sidechannel: str | None = None
    tfidf_metadata_sidechannel_scale: float = 1.0
    max_observation_lines: int = 8
    hidden_dim: int = 128
    symbol_embedding_dim: int = 64
    num_symbols: int = 16
    epochs: int = 24
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 1.0
    contrastive_temperature: float = 0.1
    lambda_pred: float = 1.0
    lambda_fut: float = 1.0
    lambda_compact: float = 0.1
    compact_marginal_weight: float = 1.0
    horizon: int = 3
    seed: int = 13
    eval_every_epochs: int = 1
    device: str = "cuda"
    dfa_backend: DfaBackendName = "aalpy"
    hardening_strategy: HardeningStrategyName = DEFAULT_HARDENING_STRATEGY
    hardening_threshold: float | None = None
    trusted_state_min_count: int = 10
    state_risk_smoothing_alpha: float = 5.0
    calibration_bins: int = 10
    fit_split: str = "train"
    cal_split: str = "train"
    val_split: str = "val"
    derive_train_fit_cal: bool = True
    train_fit_ratio: float = 0.8
    train_cal_ratio: float = 0.2
    protocol_split_seed: int = 1
    legacy_reproduction: bool = False


@dataclass
class TrainingResult:
    best_epoch: int
    best_metrics: dict
    output_dir: str
    checkpoint_path: str


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict:
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(rng_state: dict | None) -> bool:
    if rng_state is None:
        return False

    def _coerce_rng_tensor(value: object) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
        else:
            tensor = torch.as_tensor(value)
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if tensor.dtype != torch.uint8:
            tensor = tensor.to(dtype=torch.uint8)
        return tensor.contiguous()

    try:
        random.setstate(rng_state["python_random_state"])
        np.random.set_state(rng_state["numpy_random_state"])
        torch.set_rng_state(_coerce_rng_tensor(rng_state["torch_random_state"]))
    except (KeyError, TypeError, ValueError):
        return False
    cuda_state = rng_state.get("torch_cuda_random_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [_coerce_rng_tensor(state) for state in cuda_state]
        )
    return True


def _validate_resume_config(
    checkpoint_config: dict,
    current_config: dict,
    *,
    allowed_overrides: set[str],
) -> None:
    mismatches: list[str] = []
    for key, checkpoint_value in checkpoint_config.items():
        if key in allowed_overrides or key not in current_config:
            continue
        current_value = current_config[key]
        values_match = current_value == checkpoint_value
        if key == "warning_model_type":
            try:
                values_match = (
                    normalize_warning_model_type(str(current_value))
                    == normalize_warning_model_type(str(checkpoint_value))
                )
            except ValueError:
                values_match = False
        if not values_match:
            mismatches.append(
                f"{key}: checkpoint={checkpoint_value!r}, current={current_value!r}"
            )
    if mismatches:
        mismatch_preview = "\n".join(mismatches[:10])
        if len(mismatches) > 10:
            mismatch_preview += f"\n... and {len(mismatches) - 10} more"
        raise ValueError(
            "Resume checkpoint config mismatch; only 'epochs' and 'device' may differ.\n"
            f"{mismatch_preview}"
        )


def _restore_encoder_from_checkpoint(
    *,
    encoder: BaseSegmentEncoder,
    checkpoint: dict,
    encoder_type: str,
    fine_tune_encoder: bool,
) -> None:
    artifact_state = checkpoint.get("encoder_artifact_state")
    if encoder_type == "tfidf" and artifact_state is not None:
        encoder.load_artifact_state(artifact_state)

    encoder_state_dict = checkpoint.get("encoder_state_dict")
    if encoder_type == "hybrid":
        if encoder_state_dict is None:
            raise ValueError("Hybrid checkpoint is missing encoder_state_dict")
        encoder.load_state_dict(encoder_state_dict)
    elif encoder_type == "transformer" and fine_tune_encoder:
        if encoder_state_dict is None:
            raise ValueError("Fine-tuned transformer checkpoint is missing encoder_state_dict")
        encoder.load_state_dict(encoder_state_dict)


def _build_encoder(
    config: TrainingConfig,
    train_trajectories: list[TrajectoryRecord],
) -> BaseSegmentEncoder:
    if config.encoder_type == "transformer":
        return TransformerSegmentEncoder(
            model_name=config.encoder_name,
            fine_tune=config.fine_tune_encoder,
            batch_size=config.encoder_batch_size,
            max_length=config.encoder_max_length,
            step_view_text_mode=config.transformer_stepview_view,
        )
    if config.encoder_type == "hybrid":
        return HybridStepEncoder(
            model_name=config.encoder_name,
            lexical_max_features=config.encoder_max_features,
            step_view_text_mode=config.step_view_text_mode,
            lexical_metadata_sidechannel_mode=config.tfidf_metadata_sidechannel,
            lexical_metadata_sidechannel_scale=config.tfidf_metadata_sidechannel_scale,
            transformer_step_view_mode=config.transformer_stepview_view,
            fine_tune=config.fine_tune_encoder,
            batch_size=config.encoder_batch_size,
            max_length=config.encoder_max_length,
        )
    if config.encoder_type == "tfidf":
        return TfidfSegmentEncoder(
            max_features=config.encoder_max_features,
            step_view_text_mode=config.step_view_text_mode,
            sparse_output=config.tfidf_sparse_runtime,
            metadata_sidechannel_mode=config.tfidf_metadata_sidechannel,
            metadata_sidechannel_scale=config.tfidf_metadata_sidechannel_scale,
        )
    raise ValueError(f"Unsupported encoder_type: {config.encoder_type}")


def _fit_encoder_texts(
    encoder: BaseSegmentEncoder,
    train_trajectories: list[TrajectoryRecord],
    config: TrainingConfig,
    runtime_cache: RuntimeCache | None = None,
) -> None:
    if not isinstance(encoder, (TfidfSegmentEncoder, HybridStepEncoder)):
        return
    payloads = _build_training_payloads(
        train_trajectories,
        representation_mode=config.representation_mode,
        max_observation_lines=config.max_observation_lines,
        step_view_frontend=config.step_view_frontend,
        tau2_refinement_profile=config.tau2_refinement_profile,
        skillsbench_process_profile=config.skillsbench_process_profile,
        runtime_cache=runtime_cache,
    )
    encoder.fit(payloads)


def _build_training_payloads(
    trajectories: list[TrajectoryRecord],
    representation_mode: str,
    max_observation_lines: int,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    runtime_cache: RuntimeCache | None = None,
) -> list[object]:
    if runtime_cache is None:
        runtime_cache = RuntimeCache()
    return [
        payload
        for trajectory in trajectories
        for payload in runtime_cache.get_payloads(
            trajectory,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
    ]


def _trajectory_batches(
    trajectories: list[TrajectoryRecord],
    batch_size: int,
):
    class _Batches:
        def __len__(self) -> int:
            return (len(trajectories) + batch_size - 1) // batch_size

        def __iter__(self):
            for start in range(0, len(trajectories), batch_size):
                yield trajectories[start : start + batch_size]

    return _Batches()


def _length_bucketed_trajectory_batches(
    trajectories: list[TrajectoryRecord],
    batch_size: int,
):
    ordered = [
        trajectory
        for _, trajectory in sorted(
            enumerate(trajectories),
            key=lambda item: (trajectory_step_count(item[1]), item[0]),
        )
    ]
    return _trajectory_batches(ordered, batch_size)


def _encode_batch(
    encoder: BaseSegmentEncoder,
    batch: list[TrajectoryRecord],
    device: torch.device,
    representation_mode: str,
    max_observation_lines: int,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    runtime_cache: RuntimeCache | None = None,
) -> list[torch.Tensor]:
    cache = runtime_cache or RuntimeCache()
    per_trajectory_embeddings: list[torch.Tensor] = []
    for trajectory_batch in iter_trajectory_batches(
        encoder,
        batch,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        dataset_name=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
        runtime_cache=cache,
    ):
        encoded, lengths, _ = encode_trajectories(
            encoder,
            trajectory_batch,
            device=device,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            runtime_cache=cache,
        )
        per_trajectory_embeddings.extend(split_encoded_trajectories(encoded, lengths))
    return per_trajectory_embeddings


def _build_prefix_signature_keys(
    trajectory: TrajectoryRecord,
    horizon: int,
    runtime_cache: RuntimeCache | None = None,
) -> tuple[tuple[str, str, str], ...]:
    if runtime_cache is None:
        runtime_cache = RuntimeCache()
    return runtime_cache.get_future_signature_keys(trajectory, horizon=horizon)


def _precompute_frozen_encoder_embeddings(
    encoder: BaseSegmentEncoder,
    trajectories: list[TrajectoryRecord],
    *,
    device: torch.device,
    representation_mode: str,
    max_observation_lines: int,
    step_view_frontend: str,
    tau2_refinement_profile: str | None,
    skillsbench_process_profile: str | None,
    runtime_cache: RuntimeCache,
    progress_label: str,
) -> None:
    if not trajectories or not encoder.supports_runtime_embedding_cache():
        return
    if isinstance(encoder, TfidfSegmentEncoder):
        return
    precompute_trajectory_embeddings(
        encoder,
        trajectories,
        device=device,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        dataset_name=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
        runtime_cache=runtime_cache,
        progress_label=progress_label,
    )


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _is_collapsed_symbol_frontend_error(error: ValueError) -> bool:
    return str(error) == _COLLAPSED_SYMBOL_FRONTEND_ERROR


def _exact_checkpoint_selection_key(metrics: dict) -> tuple[float, float, float]:
    trusted_auprc = float(metrics.get("trusted_state_auprc", metrics["auprc"]))
    trusted_auroc = float(metrics.get("trusted_state_auroc", metrics["auroc"]))
    state_count = float(metrics["dfa_state_count"])
    return (trusted_auprc, trusted_auroc, -state_count)


def _legacy_exact_checkpoint_score(metrics: dict) -> float:
    return float(metrics["auroc"]) - 0.01 * float(metrics["dfa_state_count"])


def _attach_encoder_artifacts(checkpoint: dict, encoder: BaseSegmentEncoder) -> None:
    artifact_state = encoder.export_artifact_state()
    if artifact_state is not None:
        checkpoint["encoder_artifact_state"] = artifact_state


def train_symbolizer(
    train_trajectories: list[TrajectoryRecord],
    val_trajectories: list[TrajectoryRecord],
    output_dir: str | Path,
    config: TrainingConfig,
    cal_trajectories: list[TrajectoryRecord] | None = None,
    resume_from: str | Path | None = None,
) -> TrainingResult:
    _set_seed(config.seed)
    device = torch.device(config.device)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_cache = RuntimeCache()

    encoder = _build_encoder(config, train_trajectories)
    _fit_encoder_texts(
        encoder,
        train_trajectories,
        config,
        runtime_cache=runtime_cache,
    )
    _save_json(
        output_root / "representation_stats.json",
        {
            "fit": summarize_representation_stats(
                train_trajectories,
                representation_mode=config.representation_mode,
                max_observation_lines=config.max_observation_lines,
                dataset_name=config.step_view_frontend,
                tau2_refinement_profile=config.tau2_refinement_profile,
                skillsbench_process_profile=config.skillsbench_process_profile,
            ),
            "cal": summarize_representation_stats(
                cal_trajectories or train_trajectories,
                representation_mode=config.representation_mode,
                max_observation_lines=config.max_observation_lines,
                dataset_name=config.step_view_frontend,
                tau2_refinement_profile=config.tau2_refinement_profile,
                skillsbench_process_profile=config.skillsbench_process_profile,
            ),
            "val": summarize_representation_stats(
                val_trajectories,
                representation_mode=config.representation_mode,
                max_observation_lines=config.max_observation_lines,
                dataset_name=config.step_view_frontend,
                tau2_refinement_profile=config.tau2_refinement_profile,
                skillsbench_process_profile=config.skillsbench_process_profile,
            ),
        },
    )
    _precompute_frozen_encoder_embeddings(
        encoder,
        train_trajectories,
        device=device,
        representation_mode=config.representation_mode,
        max_observation_lines=config.max_observation_lines,
        step_view_frontend=config.step_view_frontend,
        tau2_refinement_profile=config.tau2_refinement_profile,
        skillsbench_process_profile=config.skillsbench_process_profile,
        runtime_cache=runtime_cache,
        progress_label="soft-monitor/precompute/train",
    )
    _precompute_frozen_encoder_embeddings(
        encoder,
        cal_trajectories or train_trajectories,
        device=device,
        representation_mode=config.representation_mode,
        max_observation_lines=config.max_observation_lines,
        step_view_frontend=config.step_view_frontend,
        tau2_refinement_profile=config.tau2_refinement_profile,
        skillsbench_process_profile=config.skillsbench_process_profile,
        runtime_cache=runtime_cache,
        progress_label="soft-monitor/precompute/cal",
    )
    _precompute_frozen_encoder_embeddings(
        encoder,
        val_trajectories,
        device=device,
        representation_mode=config.representation_mode,
        max_observation_lines=config.max_observation_lines,
        step_view_frontend=config.step_view_frontend,
        tau2_refinement_profile=config.tau2_refinement_profile,
        skillsbench_process_profile=config.skillsbench_process_profile,
        runtime_cache=runtime_cache,
        progress_label="soft-monitor/precompute/val",
    )

    input_dim = encoder.output_dim
    symbolizer = GumbelEventSymbolizer(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_symbols=config.num_symbols,
        symbol_embedding_dim=config.symbol_embedding_dim,
    ).to(device)
    prefix_predictor = SymbolicPrefixPredictor(
        symbol_embedding_dim=config.symbol_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_symbols=config.num_symbols,
    ).to(device)

    parameters: list[nn.Parameter] = list(symbolizer.parameters()) + list(
        prefix_predictor.parameters()
    )
    if config.encoder_type in {"transformer", "hybrid"} and (
        config.encoder_type == "hybrid" or config.fine_tune_encoder
    ):
        parameters.extend(parameter for parameter in encoder.parameters() if parameter.requires_grad)

    optimizer = AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_epoch = -1
    best_metrics: dict = {}
    best_selection_key = (-math.inf, -math.inf, -math.inf)
    best_score = -math.inf
    best_checkpoint_path = output_root / "best_checkpoint.pt"
    last_checkpoint_path = output_root / "last_checkpoint.pt"
    history: list[dict] = []
    start_epoch = 1

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        checkpoint_config = checkpoint.get("config")
        if not isinstance(checkpoint_config, dict):
            raise ValueError("Resume checkpoint is missing its training config")
        _validate_resume_config(
            checkpoint_config,
            asdict(config),
            allowed_overrides={"epochs", "device"},
        )
        completed_epoch = checkpoint.get("epoch")
        if completed_epoch is None:
            raise ValueError(
                "Resume checkpoint is missing its completed epoch; "
                "this checkpoint predates resume support."
            )
        completed_epoch = int(completed_epoch)
        if completed_epoch >= config.epochs:
            raise ValueError(
                f"Resume checkpoint already reached epoch {completed_epoch}, "
                f"but requested total epochs is {config.epochs}."
            )
        _restore_encoder_from_checkpoint(
            encoder=encoder,
            checkpoint=checkpoint,
            encoder_type=config.encoder_type,
            fine_tune_encoder=config.fine_tune_encoder,
        )
        symbolizer.load_state_dict(checkpoint["symbolizer_state_dict"])
        prefix_predictor.load_state_dict(checkpoint["prefix_predictor_state_dict"])
        optimizer_state_dict = checkpoint.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        else:
            print(
                f"[resume] {resume_path} lacks optimizer_state_dict; "
                "optimizer will restart from its configured defaults.",
                flush=True,
            )
        if not _restore_rng_state(checkpoint.get("rng_state")):
            print(
                f"[resume] {resume_path} lacks RNG state; "
                "shuffle order will restart from the configured seed.",
                flush=True,
            )
        history = list(checkpoint.get("history", []))
        best_epoch = int(checkpoint.get("best_epoch", -1))
        best_metrics = checkpoint.get("best_metrics", {})
        if config.legacy_reproduction:
            if "best_score" in checkpoint:
                best_score = float(checkpoint["best_score"])
            elif best_epoch >= 0 and best_metrics:
                best_score = _legacy_exact_checkpoint_score(best_metrics)
        else:
            resume_selection_key = checkpoint.get("best_selection_key")
            if resume_selection_key is not None:
                best_selection_key = tuple(float(value) for value in resume_selection_key)
            elif best_epoch >= 0 and best_metrics:
                best_selection_key = _exact_checkpoint_selection_key(best_metrics)
        start_epoch = completed_epoch + 1
        print(
            f"[resume] loaded {resume_path} at epoch {completed_epoch}; "
            f"continuing through epoch {config.epochs}",
            flush=True,
        )

    for epoch in range(start_epoch, config.epochs + 1):
        symbolizer.train()
        prefix_predictor.train()
        if isinstance(encoder, nn.Module):
            encoder.train(config.fine_tune_encoder)

        epoch_losses = {"total": 0.0, "pred": 0.0, "fut": 0.0, "compact": 0.0}
        batches = _trajectory_batches(train_trajectories, config.batch_size)
        random.shuffle(batches)
        for batch in tqdm(batches, desc=f"epoch-{epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            encoded_sequences = _encode_batch(
                encoder,
                batch,
                device,
                representation_mode=config.representation_mode,
                max_observation_lines=config.max_observation_lines,
                runtime_cache=runtime_cache,
            )
            batch_pred_losses = []
            batch_compact_probs = []
            prefix_embeddings = []
            prefix_labels = []

            for trajectory, encoded_sequence in zip(batch, encoded_sequences):
                prefix_signature_keys = _build_prefix_signature_keys(
                    trajectory,
                    horizon=config.horizon,
                    runtime_cache=runtime_cache,
                )
                symbolizer_output = symbolizer(
                    encoded_sequence,
                    temperature=config.temperature,
                    hard=False,
                )
                predictor_output = prefix_predictor(
                    symbolizer_output.symbol_embeddings.unsqueeze(0)
                )
                next_logits = predictor_output.next_event_logits.squeeze(0)
                hidden_states = predictor_output.hidden_states.squeeze(0)
                if next_logits.size(0) > 1:
                    batch_pred_losses.append(
                        soft_target_cross_entropy(
                            logits=next_logits[:-1],
                            target_probs=symbolizer_output.probs[1:].detach(),
                        )
                    )
                batch_compact_probs.append(symbolizer_output.probs)
                prefix_embeddings.append(hidden_states)
                prefix_labels.extend(prefix_signature_keys)

            pred_loss = (
                torch.stack(batch_pred_losses).mean()
                if batch_pred_losses
                else torch.tensor(0.0, device=device)
            )
            all_probs = torch.cat(batch_compact_probs, dim=0)
            compact_loss_value = compactness_loss(
                all_probs,
                marginal_weight=config.compact_marginal_weight,
            )
            fut_loss = supervised_contrastive_loss(
                torch.cat(prefix_embeddings, dim=0),
                prefix_labels,
                temperature=config.contrastive_temperature,
            )
            total_loss = (
                config.lambda_pred * pred_loss
                + config.lambda_fut * fut_loss
                + config.lambda_compact * compact_loss_value
            )
            total_loss.backward()
            optimizer.step()

            epoch_losses["total"] += float(total_loss.item())
            epoch_losses["pred"] += float(pred_loss.item())
            epoch_losses["fut"] += float(fut_loss.item())
            epoch_losses["compact"] += float(compact_loss_value.item())

        averaged_epoch_losses = {
            key: value / max(len(batches), 1)
            for key, value in epoch_losses.items()
        }

        epoch_record = {"epoch": epoch, "losses": averaged_epoch_losses}
        improved_best = False
        if epoch % config.eval_every_epochs == 0:
            print(
                f"[epoch {epoch}] training finished; starting validation with dfa_backend={config.dfa_backend}",
                flush=True,
            )
            try:
                eval_metrics = evaluate_symbolic_monitor(
                    train_trajectories=train_trajectories,
                    val_trajectories=val_trajectories,
                    cal_trajectories=cal_trajectories,
                    encoder=encoder,
                    symbolizer=symbolizer,
                    horizon=config.horizon,
                    num_symbols=config.num_symbols,
                    device=device,
                    dfa_backend=config.dfa_backend,
                    representation_mode=config.representation_mode,
                    max_observation_lines=config.max_observation_lines,
                    step_view_frontend=config.step_view_frontend,
                    tau2_refinement_profile=config.tau2_refinement_profile,
                    skillsbench_process_profile=config.skillsbench_process_profile,
                    trusted_state_min_count=config.trusted_state_min_count,
                    state_risk_smoothing_alpha=config.state_risk_smoothing_alpha,
                    calibration_bins=config.calibration_bins,
                    legacy_reproduction=config.legacy_reproduction,
                    runtime_cache=runtime_cache,
                    hardening_strategy=config.hardening_strategy,
                    hardening_threshold=config.hardening_threshold,
                )
            except ValueError as error:
                if not _is_collapsed_symbol_frontend_error(error):
                    raise
                epoch_record["validation_failure"] = str(error)
                print(
                    f"[epoch {epoch}] validation skipped: {error}",
                    flush=True,
                )
            else:
                epoch_record["validation"] = eval_metrics
                if config.legacy_reproduction:
                    score = _legacy_exact_checkpoint_score(eval_metrics)
                    epoch_record["selection_score"] = float(score)
                    print(
                        f"[epoch {epoch}] validation finished; "
                        f"legacy_score={score:.4f}, "
                        f"AUROC={eval_metrics['auroc']:.4f}, "
                        f"states={eval_metrics['dfa_state_count']}",
                        flush=True,
                    )
                    if score > best_score:
                        best_score = score
                        best_epoch = epoch
                        best_metrics = eval_metrics
                        improved_best = True
                else:
                    selection_key = _exact_checkpoint_selection_key(eval_metrics)
                    epoch_record["selection_key"] = {
                        "trusted_state_auprc": selection_key[0],
                        "trusted_state_auroc": selection_key[1],
                        "neg_dfa_state_count": selection_key[2],
                    }
                    print(
                        f"[epoch {epoch}] validation finished; "
                        f"trusted_AUPRC={selection_key[0]:.4f}, "
                        f"trusted_AUROC={selection_key[1]:.4f}, "
                        f"states={eval_metrics['dfa_state_count']}",
                        flush=True,
                    )
                    if selection_key > best_selection_key:
                        best_selection_key = selection_key
                        best_epoch = epoch
                        best_metrics = eval_metrics
                        improved_best = True

        history.append(epoch_record)
        checkpoint = {
            "config": asdict(config),
            "epoch": epoch,
            "symbolizer_state_dict": symbolizer.state_dict(),
            "prefix_predictor_state_dict": prefix_predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "best_selection_key": list(best_selection_key),
            "best_score": float(best_score),
            "rng_state": _capture_rng_state(),
        }
        _attach_encoder_artifacts(checkpoint, encoder)
        if config.encoder_type == "hybrid":
            checkpoint["encoder_state_dict"] = encoder.state_dict()
        elif config.encoder_type == "transformer" and config.fine_tune_encoder:
            checkpoint["encoder_state_dict"] = encoder.state_dict()
        torch.save(checkpoint, last_checkpoint_path)
        if improved_best:
            torch.save(checkpoint, best_checkpoint_path)

    _save_json(output_root / "train_history.json", {"epochs": history})
    _save_json(output_root / "train_config.json", asdict(config))
    if best_epoch < 0:
        raise RuntimeError("Training finished without producing any validation checkpoint")
    _save_json(output_root / "best_metrics.json", best_metrics)

    return TrainingResult(
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        output_dir=str(output_root),
        checkpoint_path=str(best_checkpoint_path),
    )

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from monitor_symbolization.data.schema import TrajectoryRecord
from monitor_symbolization.models.differentiable_automaton import (
    DifferentiableFiniteStateSurrogate,
    FlatPrefixRiskHead,
    GruPrefixRiskHead,
    TransformerPrefixRiskHead,
)
from monitor_symbolization.models.symbolizer import GumbelEventSymbolizer
from monitor_symbolization.models.warning_head_names import normalize_warning_model_type
from monitor_symbolization.monitor.backends import DfaBackendName
from monitor_symbolization.monitor.evaluation import (
    evaluate_paired_differentiable_monitor,
    evaluate_soft_differentiable_monitor,
)
from monitor_symbolization.monitor.hardening import (
    DEFAULT_HARDENING_STRATEGY,
    HardeningStrategyName,
)
from monitor_symbolization.runtime_cache import RuntimeCache, precompute_trajectory_embeddings
from monitor_symbolization.training.losses import compactness_loss
from monitor_symbolization.training.trainer import (
    _build_encoder,
    _capture_rng_state,
    _attach_encoder_artifacts,
    _exact_checkpoint_selection_key,
    _encode_batch,
    _fit_encoder_texts,
    _is_collapsed_symbol_frontend_error,
    _length_bucketed_trajectory_batches,
    _restore_encoder_from_checkpoint,
    _restore_rng_state,
    _save_json,
    _set_seed,
    _trajectory_batches,
    _validate_resume_config,
)


@dataclass
class DifferentiableTrainingConfig:
    warning_model_type: str = "soft-fsm"
    selection_metric: str = "discrete-trusted"
    encoder_type: str = "tfidf"
    representation_mode: str = "reduced-dense"
    step_view_frontend: str = "inferred"
    tau2_refinement_profile: str | None = None
    skillsbench_process_profile: str | None = None
    step_view_text_mode: str = "full"
    transformer_stepview_view: str = "lexical"
    encoder_name: str = "nomic-ai/nomic-embed-text-v1.5"
    fine_tune_encoder: bool = False
    encoder_max_length: int = 2048
    encoder_batch_size: int = 1
    encoder_max_features: int = 4096
    tfidf_sparse_runtime: bool = False
    tfidf_metadata_sidechannel: str | None = None
    tfidf_metadata_sidechannel_scale: float = 1.0
    max_observation_lines: int = 8
    hidden_dim: int = 128
    symbol_embedding_dim: int = 64
    num_symbols: int = 16
    q_max: int = 8
    epochs: int = 24
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    tau_sym_start: float = 1.0
    tau_sym_end: float = 0.25
    tau_trans_start: float = 1.0
    tau_trans_end: float = 0.25
    anneal_scheme: str = "linear"
    lambda_prefix: float = 1.0
    lambda_balance: float = 0.1
    balance_marginal_weight: float = 1.0
    batched_differentiable_runtime: bool = False
    length_bucketed_batches: bool = False
    horizon: int = 3
    seed: int = 13
    eval_every_epochs: int = 1
    device: str = "cuda"
    dfa_backend: DfaBackendName = "aalpy-rpni"
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
    final_best_paired_validation: bool = False
    skip_final_paired_validation: bool = False


@dataclass
class DifferentiableTrainingResult:
    best_epoch: int
    best_metrics: dict
    output_dir: str
    checkpoint_path: str


def _direct_soft_checkpoint_selection_key(metrics: dict) -> tuple[float, float, float]:
    return (
        float(metrics["auprc"]),
        -float(metrics["calibration_error"]),
        float(metrics["alert_lead_time"]),
    )


def _encoder_fit_trajectories(
    trajectories: list[TrajectoryRecord],
) -> list[TrajectoryRecord]:
    return getattr(trajectories, "encoder_fit_trajectories", trajectories)


def _disable_runtime_precompute(trajectories: list[TrajectoryRecord]) -> bool:
    return bool(getattr(trajectories, "disable_runtime_precompute", False))


def _soft_only_validation_payload(
    *,
    soft_output: dict,
    automaton: (
        DifferentiableFiniteStateSurrogate
        | FlatPrefixRiskHead
        | GruPrefixRiskHead
        | TransformerPrefixRiskHead
    ),
    config: DifferentiableTrainingConfig,
) -> dict:
    soft_metrics = soft_output["metrics"]
    return {
        "method": getattr(automaton, "method_name", "differentiable_automaton"),
        "validation_mode": "soft-only-checkpoint-selection",
        "dfa_backend": config.dfa_backend,
        "hardening_strategy": str(config.hardening_strategy),
        "hardening_threshold": (
            None
            if config.hardening_threshold is None
            else float(config.hardening_threshold)
        ),
        "num_symbols": int(config.num_symbols),
        "legacy_reproduction": bool(config.legacy_reproduction),
        "soft_metrics": soft_metrics,
        "summary": {
            "method": getattr(automaton, "method_name", "differentiable_automaton"),
            "validation_mode": "soft-only-checkpoint-selection",
            "soft_auroc": float(soft_metrics["auroc"]),
            "soft_auprc": float(soft_metrics["auprc"]),
            "soft_detection_latency": float(soft_metrics["detection_latency"]),
            "soft_alert_lead_time": float(soft_metrics["alert_lead_time"]),
            "soft_threshold": float(soft_metrics["threshold"]),
            "soft_state_count": int(soft_metrics["soft_state_count"]),
            "soft_calibration_error": float(soft_metrics["calibration_error"]),
            "soft_brier_score": float(soft_metrics["brier_score"]),
            "num_symbols": int(config.num_symbols),
            "dfa_backend": str(config.dfa_backend),
        },
    }


def _annealed_value(
    start: float,
    end: float,
    epoch: int,
    total_epochs: int,
    scheme: str,
) -> float:
    if total_epochs <= 1:
        return float(end)
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    if scheme == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(end + (start - end) * factor)
    if scheme != "linear":
        raise ValueError(f"Unsupported anneal scheme: {scheme}")
    return float(start + progress * (end - start))


def _compute_batch_losses_trajectory_loop(
    *,
    batch: list[TrajectoryRecord],
    encoded_sequences: list[torch.Tensor],
    runtime_cache: RuntimeCache,
    config: DifferentiableTrainingConfig,
    symbolizer: GumbelEventSymbolizer,
    automaton: (
        DifferentiableFiniteStateSurrogate
        | FlatPrefixRiskHead
        | GruPrefixRiskHead
        | TransformerPrefixRiskHead
    ),
    tau_sym: float,
    tau_trans: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_losses = []
    batch_probs = []

    for trajectory, encoded_sequence in zip(batch, encoded_sequences):
        labels = torch.tensor(
            runtime_cache.get_future_failure_labels(
                trajectory,
                horizon=config.horizon,
            ),
            dtype=torch.float32,
            device=device,
        )
        label_mask = torch.tensor(
            runtime_cache.get_prefix_label_mask(trajectory),
            dtype=torch.bool,
            device=device,
        )
        symbolizer_output = symbolizer(
            encoded_sequence,
            temperature=tau_sym,
            hard=False,
        )
        automaton_output = automaton(
            symbolizer_output.probs,
            transition_temperature=tau_trans,
        )
        prefix_losses.append(
            F.binary_cross_entropy(
                automaton_output.risk_scores[label_mask],
                labels[label_mask],
            )
        )
        batch_probs.append(symbolizer_output.probs)

    prefix_loss = torch.stack(prefix_losses).mean()
    balance_loss = compactness_loss(
        torch.cat(batch_probs, dim=0),
        marginal_weight=config.balance_marginal_weight,
    )
    return prefix_loss, balance_loss


def _compute_batch_losses_batched(
    *,
    batch: list[TrajectoryRecord],
    encoded_sequences: list[torch.Tensor],
    runtime_cache: RuntimeCache,
    config: DifferentiableTrainingConfig,
    symbolizer: GumbelEventSymbolizer,
    automaton: (
        DifferentiableFiniteStateSurrogate
        | FlatPrefixRiskHead
        | GruPrefixRiskHead
        | TransformerPrefixRiskHead
    ),
    tau_sym: float,
    tau_trans: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if any(sequence.layout in {torch.sparse_coo, torch.sparse_csr} for sequence in encoded_sequences):
        raise ValueError(
            "batched_differentiable_runtime does not support sparse encoder outputs; "
            "disable tfidf_sparse_runtime or the batched runtime path."
        )

    lengths = [sequence.size(0) for sequence in encoded_sequences]
    padded_embeddings = pad_sequence(encoded_sequences, batch_first=True)
    padding_mask = (
        torch.arange(padded_embeddings.size(1), device=device).unsqueeze(0)
        < torch.tensor(lengths, device=device).unsqueeze(1)
    )
    padded_labels = pad_sequence(
        [
            torch.tensor(
                runtime_cache.get_future_failure_labels(
                    trajectory,
                    horizon=config.horizon,
                ),
                dtype=torch.float32,
                device=device,
            )
            for trajectory in batch
        ],
        batch_first=True,
    )
    padded_label_mask = pad_sequence(
        [
            torch.tensor(
                runtime_cache.get_prefix_label_mask(trajectory),
                dtype=torch.bool,
                device=device,
            )
            for trajectory in batch
        ],
        batch_first=True,
    )

    symbolizer_output = symbolizer(
        padded_embeddings,
        temperature=tau_sym,
        hard=False,
    )
    automaton_output = automaton(
        symbolizer_output.probs,
        transition_temperature=tau_trans,
        padding_mask=padding_mask,
    )
    valid_label_mask = padding_mask & padded_label_mask
    valid_scores = automaton_output.risk_scores[valid_label_mask]
    valid_labels = padded_labels[valid_label_mask]
    prefix_loss = F.binary_cross_entropy(valid_scores, valid_labels)
    balance_loss = compactness_loss(
        symbolizer_output.probs[padding_mask],
        marginal_weight=config.balance_marginal_weight,
    )
    return prefix_loss, balance_loss


def _build_warning_model(
    config: DifferentiableTrainingConfig,
) -> (
    DifferentiableFiniteStateSurrogate
    | FlatPrefixRiskHead
    | GruPrefixRiskHead
    | TransformerPrefixRiskHead
):
    warning_model_type = normalize_warning_model_type(config.warning_model_type)
    if warning_model_type == "soft-fsm":
        return DifferentiableFiniteStateSurrogate(
            num_symbols=config.num_symbols,
            num_states=config.q_max,
        )
    if warning_model_type == "symbol-flat":
        return FlatPrefixRiskHead(
            num_symbols=config.num_symbols,
            num_states=config.q_max,
        )
    if warning_model_type == "symbol-gru":
        return GruPrefixRiskHead(
            num_symbols=config.num_symbols,
            num_states=config.q_max,
        )
    if warning_model_type == "symbol-transformer":
        return TransformerPrefixRiskHead(
            num_symbols=config.num_symbols,
            num_states=config.q_max,
        )
    raise ValueError(f"Unsupported warning_model_type: {config.warning_model_type}")


def train_differentiable_automaton(
    train_trajectories: list[TrajectoryRecord],
    val_trajectories: list[TrajectoryRecord],
    output_dir: str | Path,
    config: DifferentiableTrainingConfig,
    cal_trajectories: list[TrajectoryRecord] | None = None,
    resume_from: str | Path | None = None,
) -> DifferentiableTrainingResult:
    config.warning_model_type = normalize_warning_model_type(config.warning_model_type)
    if config.final_best_paired_validation and (
        config.legacy_reproduction or config.selection_metric != "direct-soft"
    ):
        raise ValueError(
            "final_best_paired_validation requires direct-soft checkpoint selection; "
            "legacy or discrete-selected protocols need per-epoch discrete validation."
        )
    _set_seed(config.seed)
    device = torch.device(config.device)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_cache = RuntimeCache()

    encoder = _build_encoder(config, train_trajectories)
    _fit_encoder_texts(
        encoder,
        _encoder_fit_trajectories(train_trajectories),
        config,
        runtime_cache=runtime_cache,
    )

    # Pre-encode all frozen transformer embeddings once before training.
    if encoder.supports_runtime_embedding_cache() and not config.fine_tune_encoder:
        for label, split_trajectories in [
            ("precompute-train", train_trajectories),
            ("precompute-val", val_trajectories),
        ]:
            if _disable_runtime_precompute(split_trajectories):
                print(
                    f"[{label}] skipping runtime embedding precompute for lazy/transient trajectories",
                    flush=True,
                )
                continue
            precompute_trajectory_embeddings(
                encoder,
                split_trajectories,
                device=device,
                representation_mode=config.representation_mode,
                max_observation_lines=config.max_observation_lines,
                dataset_name=config.step_view_frontend,
                tau2_refinement_profile=config.tau2_refinement_profile,
                skillsbench_process_profile=config.skillsbench_process_profile,
                runtime_cache=runtime_cache,
                progress_label=label,
            )

    symbolizer = GumbelEventSymbolizer(
        input_dim=encoder.output_dim,
        hidden_dim=config.hidden_dim,
        num_symbols=config.num_symbols,
        symbol_embedding_dim=config.symbol_embedding_dim,
    ).to(device)
    automaton = _build_warning_model(config).to(device)

    parameters: list[nn.Parameter] = list(symbolizer.parameters()) + list(automaton.parameters())
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
            allowed_overrides={
                "epochs",
                "device",
                "final_best_paired_validation",
                "skip_final_paired_validation",
            },
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
        warning_model_state = checkpoint.get("warning_model_state_dict")
        if warning_model_state is None:
            warning_model_state = checkpoint["automaton_state_dict"]
        automaton.load_state_dict(warning_model_state)
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
                discrete_metrics = best_metrics.get("discrete_metrics", {})
                faithfulness = best_metrics.get("faithfulness", {})
                if discrete_metrics and faithfulness:
                    best_score = (
                        float(discrete_metrics["auroc"])
                        - 0.01 * float(discrete_metrics["dfa_state_count"])
                        - float(faithfulness["gap_auroc"])
                    )
        else:
            resume_selection_key = checkpoint.get("best_selection_key")
            if resume_selection_key is not None:
                best_selection_key = tuple(float(value) for value in resume_selection_key)
            elif best_epoch >= 0 and best_metrics:
                if config.selection_metric == "direct-soft":
                    best_selection_key = _direct_soft_checkpoint_selection_key(
                        best_metrics["soft_metrics"]
                    )
                else:
                    best_selection_key = _exact_checkpoint_selection_key(
                        best_metrics["discrete_metrics"]
                    )
        start_epoch = completed_epoch + 1
        print(
            f"[resume] loaded {resume_path} at epoch {completed_epoch}; "
            f"continuing through epoch {config.epochs}",
            flush=True,
        )

    for epoch in range(start_epoch, config.epochs + 1):
        tau_sym = _annealed_value(
            config.tau_sym_start,
            config.tau_sym_end,
            epoch,
            config.epochs,
            config.anneal_scheme,
        )
        tau_trans = _annealed_value(
            config.tau_trans_start,
            config.tau_trans_end,
            epoch,
            config.epochs,
            config.anneal_scheme,
        )

        symbolizer.train()
        automaton.train()
        if isinstance(encoder, nn.Module):
            encoder.train(config.fine_tune_encoder)

        epoch_losses = {"total": 0.0, "prefix": 0.0, "balance": 0.0}
        batch_iterator = (
            _length_bucketed_trajectory_batches
            if config.length_bucketed_batches
            else _trajectory_batches
        )
        batches = batch_iterator(train_trajectories, config.batch_size)
        for batch in tqdm(batches, desc=f"epoch-{epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            encoded_sequences = _encode_batch(
                encoder,
                batch,
                device,
                representation_mode=config.representation_mode,
                max_observation_lines=config.max_observation_lines,
                step_view_frontend=config.step_view_frontend,
                tau2_refinement_profile=config.tau2_refinement_profile,
                skillsbench_process_profile=config.skillsbench_process_profile,
                runtime_cache=runtime_cache,
            )
            if config.batched_differentiable_runtime:
                prefix_loss, balance_loss = _compute_batch_losses_batched(
                    batch=batch,
                    encoded_sequences=encoded_sequences,
                    runtime_cache=runtime_cache,
                    config=config,
                    symbolizer=symbolizer,
                    automaton=automaton,
                    tau_sym=tau_sym,
                    tau_trans=tau_trans,
                    device=device,
                )
            else:
                prefix_loss, balance_loss = _compute_batch_losses_trajectory_loop(
                    batch=batch,
                    encoded_sequences=encoded_sequences,
                    runtime_cache=runtime_cache,
                    config=config,
                    symbolizer=symbolizer,
                    automaton=automaton,
                    tau_sym=tau_sym,
                    tau_trans=tau_trans,
                    device=device,
                )
            total_loss = (
                config.lambda_prefix * prefix_loss
                + config.lambda_balance * balance_loss
            )
            total_loss.backward()
            optimizer.step()

            epoch_losses["total"] += float(total_loss.item())
            epoch_losses["prefix"] += float(prefix_loss.item())
            epoch_losses["balance"] += float(balance_loss.item())

        averaged_epoch_losses = {
            key: value / max(len(batches), 1)
            for key, value in epoch_losses.items()
        }
        epoch_record = {
            "epoch": epoch,
            "losses": averaged_epoch_losses,
            "tau_sym": tau_sym,
            "tau_trans": tau_trans,
        }
        improved_best = False

        if epoch % config.eval_every_epochs == 0:
            if config.final_best_paired_validation:
                final_validation_note = (
                    "final best checkpoint will keep soft-only validation"
                    if config.skip_final_paired_validation
                    else (
                        "final best checkpoint will run paired "
                        f"DFA/RPNI validation with dfa_backend={config.dfa_backend}"
                    )
                )
                print(
                    f"[epoch {epoch}] training finished; starting soft validation "
                    f"for checkpoint selection; {final_validation_note}",
                    flush=True,
                )
                soft_output = evaluate_soft_differentiable_monitor(
                    train_trajectories=train_trajectories,
                    eval_trajectories=val_trajectories,
                    encoder=encoder,
                    symbolizer=symbolizer,
                    automaton=automaton,
                    horizon=config.horizon,
                    device=device,
                    symbol_temperature=tau_sym,
                    transition_temperature=tau_trans,
                    representation_mode=config.representation_mode,
                    max_observation_lines=config.max_observation_lines,
                    step_view_frontend=config.step_view_frontend,
                    tau2_refinement_profile=config.tau2_refinement_profile,
                    skillsbench_process_profile=config.skillsbench_process_profile,
                    runtime_cache=runtime_cache,
                )
                eval_metrics = _soft_only_validation_payload(
                    soft_output=soft_output,
                    automaton=automaton,
                    config=config,
                )
                epoch_record["validation"] = eval_metrics
                soft_metrics = eval_metrics["soft_metrics"]
                selection_key = _direct_soft_checkpoint_selection_key(soft_metrics)
                epoch_record["selection_key"] = {
                    "soft_auprc": selection_key[0],
                    "neg_soft_ece": selection_key[1],
                    "soft_alert_lead_time": selection_key[2],
                }
                print(
                    f"[epoch {epoch}] soft validation finished; "
                    f"soft_AUPRC={selection_key[0]:.4f}, "
                    f"soft_ECE={soft_metrics['calibration_error']:.4f}, "
                    f"soft_lead_time={selection_key[2]:.4f}",
                    flush=True,
                )
                if selection_key > best_selection_key:
                    best_selection_key = selection_key
                    best_epoch = epoch
                    best_metrics = eval_metrics
                    improved_best = True
            else:
                print(
                    f"[epoch {epoch}] training finished; starting paired validation with "
                    f"dfa_backend={config.dfa_backend}",
                    flush=True,
                )
                try:
                    eval_metrics = evaluate_paired_differentiable_monitor(
                        train_trajectories=train_trajectories,
                        eval_trajectories=val_trajectories,
                        cal_trajectories=cal_trajectories,
                        encoder=encoder,
                        symbolizer=symbolizer,
                        automaton=automaton,
                        horizon=config.horizon,
                        num_symbols=config.num_symbols,
                        device=device,
                        symbol_temperature=tau_sym,
                        transition_temperature=tau_trans,
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
                        allow_discrete_failure=config.selection_metric == "direct-soft",
                    )
                except ValueError as error:
                    if not _is_collapsed_symbol_frontend_error(error):
                        raise
                    epoch_record["validation_failure"] = str(error)
                    print(f"[epoch {epoch}] validation skipped: {error}", flush=True)
                else:
                    epoch_record["validation"] = eval_metrics
                    discrete_metrics = eval_metrics["discrete_metrics"]
                    if config.legacy_reproduction:
                        faithfulness = eval_metrics["faithfulness"]
                        score = (
                            float(discrete_metrics["auroc"])
                            - 0.01 * float(discrete_metrics["dfa_state_count"])
                            - float(faithfulness["gap_auroc"])
                        )
                        epoch_record["selection_score"] = float(score)
                        print(
                            f"[epoch {epoch}] paired validation finished; "
                            f"legacy_score={score:.4f}",
                            flush=True,
                        )
                        if score > best_score:
                            best_score = score
                            best_epoch = epoch
                            best_metrics = eval_metrics
                            improved_best = True
                    else:
                        if config.selection_metric == "direct-soft":
                            soft_metrics = eval_metrics["soft_metrics"]
                            selection_key = _direct_soft_checkpoint_selection_key(
                                soft_metrics
                            )
                            epoch_record["selection_key"] = {
                                "soft_auprc": selection_key[0],
                                "neg_soft_ece": selection_key[1],
                                "soft_alert_lead_time": selection_key[2],
                            }
                            print(
                                f"[epoch {epoch}] paired validation finished; "
                                f"soft_AUPRC={selection_key[0]:.4f}, "
                                f"soft_ECE={soft_metrics['calibration_error']:.4f}, "
                                f"soft_lead_time={selection_key[2]:.4f}",
                                flush=True,
                            )
                        else:
                            selection_key = _exact_checkpoint_selection_key(discrete_metrics)
                            epoch_record["selection_key"] = {
                                "trusted_state_auprc": selection_key[0],
                                "trusted_state_auroc": selection_key[1],
                                "neg_dfa_state_count": selection_key[2],
                            }
                            print(
                                f"[epoch {epoch}] paired validation finished; "
                                f"trusted_AUPRC={selection_key[0]:.4f}, "
                                f"trusted_AUROC={selection_key[1]:.4f}, "
                                f"states={discrete_metrics['dfa_state_count']}",
                                flush=True,
                            )
                        if selection_key > best_selection_key:
                            best_selection_key = selection_key
                            best_epoch = epoch
                            best_metrics = eval_metrics
                            improved_best = True

        history.append(epoch_record)
        last_checkpoint = {
            "config": asdict(config),
            "epoch": epoch,
            "symbolizer_state_dict": symbolizer.state_dict(),
            "automaton_state_dict": automaton.state_dict(),
            "warning_model_state_dict": automaton.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tau_sym": tau_sym,
            "tau_trans": tau_trans,
            "history": history,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "best_selection_key": list(best_selection_key),
            "best_score": float(best_score),
            "rng_state": _capture_rng_state(),
        }
        _attach_encoder_artifacts(last_checkpoint, encoder)
        if config.encoder_type == "hybrid":
            last_checkpoint["encoder_state_dict"] = encoder.state_dict()
        elif config.encoder_type == "transformer" and config.fine_tune_encoder:
            last_checkpoint["encoder_state_dict"] = encoder.state_dict()
        torch.save(last_checkpoint, last_checkpoint_path)
        if improved_best:
            torch.save(last_checkpoint, best_checkpoint_path)

    _save_json(output_root / "train_history.json", {"epochs": history})
    _save_json(output_root / "train_config.json", asdict(config))
    if best_epoch < 0:
        best_epoch = config.epochs
        best_metrics = {
            "validation_failure": (
                "Training finished without producing a discrete validation checkpoint; "
                "use last_checkpoint.pt for restart or inspection."
            )
        }
    elif (
        config.final_best_paired_validation
        and config.skip_final_paired_validation
        and best_checkpoint_path.exists()
    ):
        print(
            f"[final-best] keeping soft-only best checkpoint from epoch {best_epoch}; "
            "skipping paired DFA/RPNI validation by configuration",
            flush=True,
        )
        best_metrics["validation_mode"] = "final-best-soft-only"
        best_metrics.setdefault("summary", {})["validation_mode"] = "final-best-soft-only"
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        best_checkpoint["best_metrics"] = best_metrics
        best_checkpoint["final_best_paired_validation"] = True
        best_checkpoint["skip_final_paired_validation"] = True
        torch.save(best_checkpoint, best_checkpoint_path)
    elif config.final_best_paired_validation and best_checkpoint_path.exists():
        print(
            f"[final-best] loading best checkpoint from epoch {best_epoch} for one "
            f"paired DFA/RPNI validation with dfa_backend={config.dfa_backend}",
            flush=True,
        )
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        _restore_encoder_from_checkpoint(
            encoder=encoder,
            checkpoint=best_checkpoint,
            encoder_type=config.encoder_type,
            fine_tune_encoder=config.fine_tune_encoder,
        )
        symbolizer.load_state_dict(best_checkpoint["symbolizer_state_dict"])
        best_warning_model_state = best_checkpoint.get("warning_model_state_dict")
        if best_warning_model_state is None:
            best_warning_model_state = best_checkpoint["automaton_state_dict"]
        automaton.load_state_dict(best_warning_model_state)
        final_cache = (
            RuntimeCache()
            if config.encoder_type in {"transformer", "hybrid"} and config.fine_tune_encoder
            else runtime_cache
        )
        best_metrics = evaluate_paired_differentiable_monitor(
            train_trajectories=train_trajectories,
            eval_trajectories=val_trajectories,
            cal_trajectories=cal_trajectories,
            encoder=encoder,
            symbolizer=symbolizer,
            automaton=automaton,
            horizon=config.horizon,
            num_symbols=config.num_symbols,
            device=device,
            symbol_temperature=float(best_checkpoint.get("tau_sym", config.tau_sym_end)),
            transition_temperature=float(best_checkpoint.get("tau_trans", config.tau_trans_end)),
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
            runtime_cache=final_cache,
            hardening_strategy=config.hardening_strategy,
            hardening_threshold=config.hardening_threshold,
            allow_discrete_failure=True,
        )
        best_metrics["validation_mode"] = "final-best-paired-validation"
        best_checkpoint["best_metrics"] = best_metrics
        best_checkpoint["final_best_paired_validation"] = True
        torch.save(best_checkpoint, best_checkpoint_path)
    _save_json(output_root / "best_metrics.json", best_metrics)

    return DifferentiableTrainingResult(
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        output_dir=str(output_root),
        checkpoint_path=str(best_checkpoint_path if best_checkpoint_path.exists() else last_checkpoint_path),
    )

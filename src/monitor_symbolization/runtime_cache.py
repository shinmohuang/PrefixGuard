from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from monitor_symbolization.data.prefixes import (
    future_failure_labels_for_trajectory,
    future_signature_keys_for_trajectory,
    prefix_label_mask_for_trajectory,
)
from monitor_symbolization.data.schema import StepView, TrajectoryRecord
from monitor_symbolization.data.scrambled_index import (
    ScrambledPrefixView,
    is_scrambled_prefix_view,
    trajectory_step_count,
)
from monitor_symbolization.data.serialization import (
    build_step_payload,
    resolve_skillsbench_process_profile,
    resolve_step_view_dataset_name,
    resolve_tau2_refinement_profile,
)
from monitor_symbolization.models.encoders import BaseSegmentEncoder

Payload = str | StepView
TrajectoryKey = tuple[str, str, int, bool]
FutureSignatureKey = tuple[str, str, str]
PayloadKey = tuple[TrajectoryKey, str, int, str, str, str]
DEFAULT_TRAJECTORY_STEP_BATCH_LIMIT = 50_000
DISABLE_RUNTIME_CACHE_METADATA_KEY = "disable_runtime_cache"


def _disable_runtime_cache(trajectory: TrajectoryRecord) -> bool:
    return bool(trajectory.metadata.get(DISABLE_RUNTIME_CACHE_METADATA_KEY, False))


def _iter_encoded_payload_batches_for_runtime_cache(
    encoder: BaseSegmentEncoder,
    payloads: list[Payload],
    *,
    device: torch.device,
    progress_label: str | None,
) -> list[torch.Tensor]:
    batch_cost_limit = encoder.runtime_cache_batch_cost_limit()
    batch_item_limit = encoder.runtime_cache_batch_item_limit()
    batch_costs = encoder.runtime_cache_batch_costs(payloads)
    if batch_cost_limit is None and batch_item_limit is None:
        return [
            encoder.encode(
                payloads,
                device=device,
                progress_label=progress_label,
            ).embeddings.detach().cpu()
        ]
    if batch_cost_limit is not None and batch_costs is None:
        return [
            encoder.encode(
                payloads,
                device=device,
                progress_label=progress_label,
            ).embeddings.detach().cpu()
        ]

    embeddings_batches: list[torch.Tensor] = []
    batch_start = 0
    batch_cost = 0
    effective_costs = (
        [max(int(raw_cost), 1) for raw_cost in batch_costs]
        if batch_costs is not None
        else [1] * len(payloads)
    )
    for index, cost in enumerate(effective_costs):
        batch_size = index - batch_start
        would_exceed_cost = (
            batch_cost_limit is not None and batch_start < index and batch_cost + cost > batch_cost_limit
        )
        would_exceed_items = (
            batch_item_limit is not None and batch_start < index and batch_size >= batch_item_limit
        )
        if would_exceed_cost or would_exceed_items:
            embeddings_batches.append(
                encoder.encode(
                    payloads[batch_start:index],
                    device=device,
                    progress_label=progress_label,
                ).embeddings.detach().cpu()
            )
            batch_start = index
            batch_cost = 0
        batch_cost += cost
        hit_cost_limit = batch_cost_limit is not None and batch_cost >= batch_cost_limit
        hit_item_limit = (
            batch_item_limit is not None and (index - batch_start + 1) >= batch_item_limit
        )
        if hit_cost_limit or hit_item_limit:
            embeddings_batches.append(
                encoder.encode(
                    payloads[batch_start : index + 1],
                    device=device,
                    progress_label=progress_label,
                ).embeddings.detach().cpu()
            )
            batch_start = index + 1
            batch_cost = 0

    if batch_start < len(payloads):
        embeddings_batches.append(
            encoder.encode(
                payloads[batch_start:],
                device=device,
                progress_label=progress_label,
            ).embeddings.detach().cpu()
        )
    if not embeddings_batches:
        output_dim = getattr(encoder, "output_dim", 0)
        return [torch.empty((0, output_dim), dtype=torch.float32)]
    return embeddings_batches


def trajectory_cache_key(trajectory: TrajectoryRecord) -> TrajectoryKey:
    return (
        trajectory.split,
        trajectory.trajectory_id,
        trajectory_step_count(trajectory),
        bool(trajectory.final_success),
    )


def _trajectory_step_batch_limit(encoder: BaseSegmentEncoder) -> int:
    encoder_limit = encoder.runtime_cache_batch_item_limit()
    if encoder_limit is not None:
        return max(int(encoder_limit), 1)
    return DEFAULT_TRAJECTORY_STEP_BATCH_LIMIT


def _payloads_for_batch_cost(
    trajectory: TrajectoryRecord,
    *,
    representation_mode: str,
    max_observation_lines: int,
    dataset_name: str | None,
    tau2_refinement_profile: str | None,
    skillsbench_process_profile: str | None,
    cache: RuntimeCache,
) -> tuple[Payload, ...]:
    if not _disable_runtime_cache(trajectory):
        return cache.get_payloads(
            trajectory,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
    return tuple(
        build_step_payload(
            step,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        for step in trajectory.steps
    )


def _payload_batch_cost(
    encoder: BaseSegmentEncoder,
    payloads: tuple[Payload, ...],
) -> int:
    costs = encoder.runtime_cache_batch_costs(list(payloads))
    if costs is None:
        return len(payloads)
    return max(sum(int(cost) for cost in costs), len(payloads))


def iter_trajectory_batches(
    encoder: BaseSegmentEncoder,
    trajectories: list[TrajectoryRecord],
    *,
    representation_mode: str,
    max_observation_lines: int,
    dataset_name: str | None = None,
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    runtime_cache: RuntimeCache | None = None,
) -> Iterable[list[TrajectoryRecord]]:
    if not trajectories:
        return
    cache = runtime_cache or RuntimeCache()
    batch_limit = _trajectory_step_batch_limit(encoder)
    cost_limit = encoder.runtime_cache_batch_cost_limit()
    current_batch: list[TrajectoryRecord] = []
    current_steps = 0
    current_cost = 0
    for trajectory in trajectories:
        payloads = (
            _payloads_for_batch_cost(
                trajectory,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                dataset_name=dataset_name,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
                cache=cache,
            )
            if cost_limit is not None or not _disable_runtime_cache(trajectory)
            else tuple()
        )
        step_count = len(payloads) if payloads else trajectory_step_count(trajectory)
        trajectory_cost = _payload_batch_cost(encoder, payloads) if payloads else step_count
        cost_would_overflow = (
            cost_limit is not None and current_cost + trajectory_cost > cost_limit
        )
        if current_batch and (
            current_steps + step_count > batch_limit or cost_would_overflow
        ):
            yield current_batch
            current_batch = []
            current_steps = 0
            current_cost = 0
        current_batch.append(trajectory)
        current_steps += step_count
        current_cost += trajectory_cost
        if current_steps >= batch_limit or (
            cost_limit is not None and current_cost >= cost_limit
        ):
            yield current_batch
            current_batch = []
            current_steps = 0
            current_cost = 0
    if current_batch:
        yield current_batch


def _slice_encoded_rows(
    encoded: torch.Tensor,
    *,
    start: int,
    length: int,
) -> torch.Tensor:
    if length <= 0:
        feature_dim = encoded.size(1)
        if encoded.layout in {torch.sparse_coo, torch.sparse_csr}:
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=encoded.dtype),
                size=(0, feature_dim),
                device=encoded.device,
            ).coalesce()
        return torch.empty((0, feature_dim), dtype=encoded.dtype, device=encoded.device)
    if encoded.layout not in {torch.sparse_coo, torch.sparse_csr}:
        return encoded[start : start + length]

    trailing = int(encoded.size(0)) - start - length
    segments = split_encoded_trajectories(encoded, [start, length, trailing])
    return segments[1]


@dataclass
class RuntimeCache:
    payloads: dict[PayloadKey, tuple[Payload, ...]] = field(default_factory=dict)
    future_failure_labels: dict[tuple[TrajectoryKey, int], tuple[int, ...]] = field(default_factory=dict)
    prefix_label_masks: dict[TrajectoryKey, tuple[bool, ...]] = field(default_factory=dict)
    future_signature_keys: dict[
        tuple[TrajectoryKey, int],
        tuple[FutureSignatureKey, ...],
    ] = field(default_factory=dict)
    cached_embeddings: dict[tuple[int, PayloadKey], torch.Tensor] = field(default_factory=dict)

    @staticmethod
    def _payload_key(
        trajectory: TrajectoryRecord,
        *,
        representation_mode: str,
        max_observation_lines: int,
        dataset_name: str | None,
        tau2_refinement_profile: str | None,
        skillsbench_process_profile: str | None,
    ) -> PayloadKey:
        resolved_dataset_name = resolve_step_view_dataset_name(dataset_name)
        resolved_tau2_refinement_profile = resolve_tau2_refinement_profile(
            tau2_refinement_profile
        )
        resolved_skillsbench_process_profile = resolve_skillsbench_process_profile(
            skillsbench_process_profile
        )
        return (
            trajectory_cache_key(trajectory),
            representation_mode,
            max_observation_lines,
            resolved_dataset_name or "inferred",
            resolved_tau2_refinement_profile or "baseline",
            resolved_skillsbench_process_profile or "baseline",
        )

    def get_payloads(
        self,
        trajectory: TrajectoryRecord,
        *,
        representation_mode: str,
        max_observation_lines: int,
        dataset_name: str | None = None,
        tau2_refinement_profile: str | None = None,
        skillsbench_process_profile: str | None = None,
    ) -> tuple[Payload, ...]:
        if is_scrambled_prefix_view(trajectory):
            source_payloads = self.get_payloads(
                trajectory.source_trajectory,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                dataset_name=dataset_name,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
            )
            return tuple(
                source_payloads[index]
                for index in trajectory.shuffled_step_indices
            )
        key = self._payload_key(
            trajectory,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        resolved_dataset_name = resolve_step_view_dataset_name(dataset_name)
        resolved_tau2_refinement_profile = resolve_tau2_refinement_profile(
            tau2_refinement_profile
        )
        resolved_skillsbench_process_profile = resolve_skillsbench_process_profile(
            skillsbench_process_profile
        )
        cached = self.payloads.get(key)
        if cached is not None:
            return cached
        payloads = tuple(
            build_step_payload(
                step,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                dataset_name=resolved_dataset_name,
                tau2_refinement_profile=resolved_tau2_refinement_profile,
                skillsbench_process_profile=resolved_skillsbench_process_profile,
            )
            for step in trajectory.steps
        )
        if _disable_runtime_cache(trajectory):
            return payloads
        self.payloads[key] = payloads
        return payloads

    def get_future_failure_labels(
        self,
        trajectory: TrajectoryRecord,
        *,
        horizon: int,
    ) -> tuple[int, ...]:
        key = (trajectory_cache_key(trajectory), horizon)
        if _disable_runtime_cache(trajectory):
            return future_failure_labels_for_trajectory(trajectory, horizon=horizon)
        cached = self.future_failure_labels.get(key)
        if cached is not None:
            return cached
        labels = future_failure_labels_for_trajectory(trajectory, horizon=horizon)
        self.future_failure_labels[key] = labels
        return labels

    def get_prefix_label_mask(
        self,
        trajectory: TrajectoryRecord,
    ) -> tuple[bool, ...]:
        key = trajectory_cache_key(trajectory)
        if _disable_runtime_cache(trajectory):
            return prefix_label_mask_for_trajectory(trajectory)
        cached = self.prefix_label_masks.get(key)
        if cached is not None:
            return cached
        mask = prefix_label_mask_for_trajectory(trajectory)
        self.prefix_label_masks[key] = mask
        return mask

    def get_future_signature_keys(
        self,
        trajectory: TrajectoryRecord,
        *,
        horizon: int,
    ) -> tuple[FutureSignatureKey, ...]:
        key = (trajectory_cache_key(trajectory), horizon)
        if _disable_runtime_cache(trajectory):
            return future_signature_keys_for_trajectory(trajectory, horizon=horizon)
        cached = self.future_signature_keys.get(key)
        if cached is not None:
            return cached
        signatures = future_signature_keys_for_trajectory(trajectory, horizon=horizon)
        self.future_signature_keys[key] = signatures
        return signatures

    def get_cached_embeddings(
        self,
        encoder: BaseSegmentEncoder,
        trajectory: TrajectoryRecord,
        *,
        representation_mode: str,
        max_observation_lines: int,
        dataset_name: str | None = None,
        tau2_refinement_profile: str | None = None,
        skillsbench_process_profile: str | None = None,
    ) -> torch.Tensor | None:
        if _disable_runtime_cache(trajectory):
            return None
        payload_key = self._payload_key(
            trajectory,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        return self.cached_embeddings.get((id(encoder), payload_key))

    def store_cached_embeddings(
        self,
        encoder: BaseSegmentEncoder,
        trajectory: TrajectoryRecord,
        embeddings: torch.Tensor,
        *,
        representation_mode: str,
        max_observation_lines: int,
        dataset_name: str | None = None,
        tau2_refinement_profile: str | None = None,
        skillsbench_process_profile: str | None = None,
    ) -> None:
        if _disable_runtime_cache(trajectory):
            return
        payload_key = self._payload_key(
            trajectory,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        self.cached_embeddings[(id(encoder), payload_key)] = embeddings.detach().cpu()

    def get_cached_tfidf_embeddings(
        self,
        encoder: BaseSegmentEncoder,
        trajectory: TrajectoryRecord,
        *,
        representation_mode: str,
        max_observation_lines: int,
        dataset_name: str | None = None,
        tau2_refinement_profile: str | None = None,
        skillsbench_process_profile: str | None = None,
    ) -> torch.Tensor | None:
        return self.get_cached_embeddings(
            encoder,
            trajectory,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )

    def store_tfidf_embeddings(
        self,
        encoder: BaseSegmentEncoder,
        trajectory: TrajectoryRecord,
        embeddings: torch.Tensor,
        *,
        representation_mode: str,
        max_observation_lines: int,
        dataset_name: str | None = None,
        tau2_refinement_profile: str | None = None,
        skillsbench_process_profile: str | None = None,
    ) -> None:
        self.store_cached_embeddings(
            encoder,
            trajectory,
            embeddings,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )


def _materialize_missing_cached_embeddings(
    encoder: BaseSegmentEncoder,
    trajectories: list[TrajectoryRecord],
    payload_groups: list[tuple[Payload, ...]],
    *,
    device: torch.device,
    representation_mode: str,
    max_observation_lines: int,
    dataset_name: str | None,
    tau2_refinement_profile: str | None,
    skillsbench_process_profile: str | None,
    runtime_cache: RuntimeCache,
    progress_label: str | None,
) -> None:
    missing_payloads: list[Payload] = []
    missing_trajectories: list[TrajectoryRecord] = []
    missing_lengths: list[int] = []
    for trajectory, payloads in zip(trajectories, payload_groups):
        if runtime_cache.get_cached_embeddings(
            encoder,
            trajectory,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        ) is not None:
            continue
        missing_payloads.extend(payloads)
        missing_trajectories.append(trajectory)
        missing_lengths.append(len(payloads))
    if not missing_payloads:
        if progress_label:
            total_steps = sum(len(payloads) for payloads in payload_groups)
            print(
                f"[{progress_label}] reused cached embeddings for "
                f"{len(trajectories)} trajectories / {total_steps} steps",
                flush=True,
            )
        return

    missing_embedding_batches = _iter_encoded_payload_batches_for_runtime_cache(
        encoder,
        list(missing_payloads),
        device=device,
        progress_label=progress_label,
    )
    trajectory_index = 0
    trajectory_cursor = 0
    current_parts: list[torch.Tensor] = []
    for batch_embeddings in missing_embedding_batches:
        batch_cursor = 0
        batch_length = int(batch_embeddings.size(0))
        while batch_cursor < batch_length and trajectory_index < len(missing_trajectories):
            target_length = missing_lengths[trajectory_index]
            remaining = target_length - trajectory_cursor
            take = min(remaining, batch_length - batch_cursor)
            if take > 0:
                current_parts.append(
                    _slice_encoded_rows(
                        batch_embeddings,
                        start=batch_cursor,
                        length=take,
                    )
                )
            batch_cursor += take
            trajectory_cursor += take
            if trajectory_cursor < target_length:
                continue
            embeddings = (
                current_parts[0]
                if len(current_parts) == 1
                else torch.cat(current_parts, dim=0)
            )
            runtime_cache.store_cached_embeddings(
                encoder,
                missing_trajectories[trajectory_index],
                embeddings,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                dataset_name=dataset_name,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
            )
            trajectory_index += 1
            trajectory_cursor = 0
            current_parts = []
    if trajectory_index != len(missing_trajectories) or trajectory_cursor != 0 or current_parts:
        raise RuntimeError("Runtime cache batching failed to reassemble all trajectory embeddings")


def precompute_trajectory_embeddings(
    encoder: BaseSegmentEncoder,
    trajectories: list[TrajectoryRecord],
    *,
    device: torch.device,
    representation_mode: str,
    max_observation_lines: int,
    dataset_name: str | None = None,
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    runtime_cache: RuntimeCache | None = None,
    progress_label: str | None = None,
) -> None:
    if runtime_cache is None or not encoder.supports_runtime_embedding_cache() or not trajectories:
        return

    payload_groups = [
        runtime_cache.get_payloads(
            trajectory,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        for trajectory in trajectories
    ]
    _materialize_missing_cached_embeddings(
        encoder,
        trajectories,
        payload_groups,
        device=device,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        dataset_name=dataset_name,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
        runtime_cache=runtime_cache,
        progress_label=progress_label,
    )


def _unique_source_trajectories(
    views: list[ScrambledPrefixView],
) -> list[TrajectoryRecord]:
    sources_by_key: dict[TrajectoryKey, TrajectoryRecord] = {}
    for view in views:
        source = view.source_trajectory
        sources_by_key.setdefault(trajectory_cache_key(source), source)
    return list(sources_by_key.values())


def _gather_encoded_rows(
    encoded: torch.Tensor,
    indices: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    if encoded.layout in {torch.sparse_coo, torch.sparse_csr}:
        encoded = encoded.to_dense()
    encoded = encoded.to(device)
    return encoded.index_select(0, index_tensor)


def _encode_scrambled_prefix_views(
    encoder: BaseSegmentEncoder,
    views: list[ScrambledPrefixView],
    *,
    device: torch.device,
    representation_mode: str,
    max_observation_lines: int,
    dataset_name: str | None,
    tau2_refinement_profile: str | None,
    skillsbench_process_profile: str | None,
    runtime_cache: RuntimeCache,
    progress_label: str | None,
) -> tuple[torch.Tensor, list[int], list[str]]:
    lengths = [view.prefix_index for view in views]
    trajectory_ids = [view.trajectory_id for view in views]
    if encoder.supports_runtime_embedding_cache():
        source_trajectories = _unique_source_trajectories(views)
        source_payload_groups = [
            runtime_cache.get_payloads(
                source,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                dataset_name=dataset_name,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
            )
            for source in source_trajectories
        ]
        _materialize_missing_cached_embeddings(
            encoder,
            source_trajectories,
            source_payload_groups,
            device=device,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            runtime_cache=runtime_cache,
            progress_label=progress_label,
        )
        gathered = []
        source_embeddings_by_key: dict[TrajectoryKey, torch.Tensor] = {}
        for view in views:
            source_key = trajectory_cache_key(view.source_trajectory)
            source_embeddings = source_embeddings_by_key.get(source_key)
            if source_embeddings is None:
                cached_embeddings = runtime_cache.get_cached_embeddings(
                    encoder,
                    view.source_trajectory,
                    representation_mode=representation_mode,
                    max_observation_lines=max_observation_lines,
                    dataset_name=dataset_name,
                    tau2_refinement_profile=tau2_refinement_profile,
                    skillsbench_process_profile=skillsbench_process_profile,
                )
                if cached_embeddings is not None:
                    source_embeddings = cached_embeddings.to(device)
                    source_embeddings_by_key[source_key] = source_embeddings
            if source_embeddings is None:
                raise RuntimeError(
                    f"Missing cached source embeddings for scrambled prefix view "
                    f"'{view.trajectory_id}'"
                )
            gathered.append(
                _gather_encoded_rows(
                    source_embeddings,
                    view.shuffled_step_indices,
                    device=device,
                )
            )
        flat_embeddings = torch.cat(gathered, dim=0)
        return flat_embeddings, lengths, trajectory_ids

    payload_groups = [
        runtime_cache.get_payloads(
            view,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        for view in views
    ]
    flat_payloads = [payload for payloads in payload_groups for payload in payloads]
    flat_embeddings = encoder.encode(
        flat_payloads,
        device=device,
        progress_label=progress_label,
    ).embeddings
    return flat_embeddings, lengths, trajectory_ids


def encode_trajectories(
    encoder: BaseSegmentEncoder,
    trajectories: list[TrajectoryRecord],
    *,
    device: torch.device,
    representation_mode: str,
    max_observation_lines: int,
    dataset_name: str | None = None,
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    runtime_cache: RuntimeCache | None = None,
    progress_label: str | None = None,
) -> tuple[torch.Tensor, list[int], list[str]]:
    if not trajectories:
        output_dim = getattr(encoder, "output_dim", 0)
        empty = torch.empty((0, output_dim), dtype=torch.float32, device=device)
        return empty, [], []

    if all(is_scrambled_prefix_view(trajectory) for trajectory in trajectories):
        return _encode_scrambled_prefix_views(
            encoder,
            [trajectory for trajectory in trajectories if is_scrambled_prefix_view(trajectory)],
            device=device,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            runtime_cache=runtime_cache or RuntimeCache(),
            progress_label=progress_label,
        )

    lengths: list[int] = []
    trajectory_ids: list[str] = []
    payload_groups: list[tuple[Payload, ...]] = []
    for trajectory in trajectories:
        payloads = (
            runtime_cache.get_payloads(
                trajectory,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                dataset_name=dataset_name,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
            )
            if runtime_cache is not None
            else tuple(
                build_step_payload(
                    step,
                    representation_mode=representation_mode,
                    max_observation_lines=max_observation_lines,
                    dataset_name=dataset_name,
                    tau2_refinement_profile=tau2_refinement_profile,
                    skillsbench_process_profile=skillsbench_process_profile,
                )
                for step in trajectory.steps
            )
        )
        payload_groups.append(payloads)
        lengths.append(len(payloads))
        trajectory_ids.append(trajectory.trajectory_id)

    has_transient_trajectory = any(_disable_runtime_cache(trajectory) for trajectory in trajectories)
    if (
        runtime_cache is not None
        and encoder.supports_runtime_embedding_cache()
        and not has_transient_trajectory
    ):
        _materialize_missing_cached_embeddings(
            encoder,
            trajectories,
            payload_groups,
            device=device,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=dataset_name,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            runtime_cache=runtime_cache,
            progress_label=progress_label,
        )
        per_trajectory_embeddings = [
            runtime_cache.get_cached_embeddings(
                encoder,
                trajectory,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                dataset_name=dataset_name,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
            ).to(device)
            for trajectory in trajectories
        ]
        flat_embeddings = torch.cat(per_trajectory_embeddings, dim=0)
        return flat_embeddings, lengths, trajectory_ids

    flat_payloads = [payload for payloads in payload_groups for payload in payloads]
    flat_embeddings = encoder.encode(
        flat_payloads,
        device=device,
        progress_label=progress_label,
    ).embeddings
    return flat_embeddings, lengths, trajectory_ids


def split_encoded_trajectories(
    encoded: torch.Tensor,
    lengths: list[int],
) -> list[torch.Tensor]:
    if encoded.layout not in {torch.sparse_coo, torch.sparse_csr}:
        outputs: list[torch.Tensor] = []
        cursor = 0
        for length in lengths:
            outputs.append(encoded[cursor : cursor + length])
            cursor += length
        return outputs

    coalesced = encoded.coalesce() if encoded.layout == torch.sparse_coo else encoded.to_sparse_coo().coalesce()
    indices = coalesced.indices()
    values = coalesced.values()
    outputs = []
    cursor = 0
    feature_dim = encoded.size(1)
    for length in lengths:
        mask = (indices[0] >= cursor) & (indices[0] < cursor + length)
        local_indices = indices[:, mask].clone()
        local_indices[0] -= cursor
        outputs.append(
            torch.sparse_coo_tensor(
                local_indices,
                values[mask],
                size=(length, feature_dim),
                device=values.device,
            ).coalesce()
        )
        cursor += length
    return outputs

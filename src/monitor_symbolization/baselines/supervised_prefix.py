from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
import torch

from monitor_symbolization.data.schema import StepView, TrajectoryRecord
from monitor_symbolization.data.serialization import build_step_payload
from monitor_symbolization.models.encoders import TfidfSegmentEncoder
from monitor_symbolization.monitor.evaluation import (
    _compute_brier_score,
    _compute_expected_calibration_error,
    _select_threshold,
)
from monitor_symbolization.runtime_cache import (
    RuntimeCache,
    encode_trajectories,
    split_encoded_trajectories,
    _iter_encoded_payload_batches_for_runtime_cache,
)


PREFIX_STEP_DELIMITER = "\n<STEP_BOUNDARY>\n"
Payload = str | StepView
FORBIDDEN_FEATURE_MARKERS = (
    "split",
    "final_success",
    "failure_bucket",
    "future_signature",
    "remaining_steps",
    "remaining_steps_bin",
    "full_length",
    "post_prefix",
)
FORBIDDEN_FEATURE_LINE_RE = re.compile(
    r"^\s*("
    + "|".join(re.escape(marker) for marker in FORBIDDEN_FEATURE_MARKERS)
    + r")\s*[:=]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PrefixExample:
    trajectory_id: str
    task_id: str
    split: str
    prefix_index: int
    label: int
    trajectory_step_texts: tuple[Payload, ...]
    step_view_text_mode: str = "full"
    trajectory: TrajectoryRecord | None = field(default=None, compare=False, repr=False)

    @property
    def step_payloads(self) -> tuple[Payload, ...]:
        return self.trajectory_step_texts[: self.prefix_index]

    @property
    def step_texts(self) -> tuple[str, ...]:
        return tuple(
            _payload_to_baseline_text(
                payload,
                step_view_text_mode=self.step_view_text_mode,
            )
            for payload in self.step_payloads
        )

    @property
    def document(self) -> str:
        return PREFIX_STEP_DELIMITER.join(self.step_texts)


@dataclass(frozen=True)
class SupervisedBaselineConfig:
    model_type: str
    representation_mode: str
    horizon: int = 3
    step_view_frontend: str = "inferred"
    step_view_text_mode: str = "full"
    max_observation_lines: int = 8
    tau2_refinement_profile: str | None = None
    skillsbench_process_profile: str | None = None
    max_features: int = 4096
    tfidf_backend: str = "auto"
    tfidf_metadata_sidechannel: str | None = None
    seed: int = 13
    calibration_bins: int = 10
    fast_runtime: str | None = None
    fast_epochs: int | None = None
    feature_batch_size: int | None = None
    fast_device: str | None = None
    sequence_hidden_dim: int | None = None
    sequence_num_layers: int | None = None


@dataclass(frozen=True)
class TrainedBaseline:
    model_type: str
    vectorizer: "SegmentTfidfVectorizer"
    model: object
    config: SupervisedBaselineConfig
    selected_c: float | None
    validation_metrics: dict[str, float | int | str]


@dataclass(frozen=True)
class FeatureBatch:
    features: sparse.csr_matrix
    labels: list[int]
    row_indices: list[int]


@dataclass(frozen=True)
class SequenceFeatureBatch:
    features: torch.Tensor
    sequence_mask: torch.Tensor
    label_mask: torch.Tensor
    labels: torch.Tensor
    row_indices: list[int]


@dataclass
class TorchBinaryClassifier:
    module: torch.nn.Module
    device: str
    predict_batch_size: int = 4096

    def predict_proba(self, features):
        matrix = _to_scipy_csr(features)
        if matrix.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        device = torch.device(self.device)
        self.module.eval()
        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, matrix.shape[0], self.predict_batch_size):
                batch = matrix[start : start + self.predict_batch_size].toarray()
                tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
                logits = self.module(tensor).reshape(-1)
                probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        positive = np.concatenate(probabilities).astype(np.float32, copy=False)
        return np.stack([1.0 - positive, positive], axis=1)


class ContinuousStepViewSequenceRiskHead(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        sequence_model: str,
        num_layers: int,
    ) -> None:
        super().__init__()
        if sequence_model not in {"gru", "transformer"}:
            raise ValueError(f"Unsupported sequence_model: {sequence_model}")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.sequence_model = sequence_model
        self.num_layers = int(num_layers)
        self.input_projection = torch.nn.Linear(self.input_dim, self.hidden_dim)
        if sequence_model == "gru":
            self.sequence_encoder = torch.nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
            )
        else:
            attention_heads = next(
                candidate
                for candidate in (8, 4, 2, 1)
                if self.hidden_dim % candidate == 0
            )
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=attention_heads,
                dim_feedforward=max(4 * self.hidden_dim, 64),
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = torch.nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.num_layers,
            )
        self.risk_head = torch.nn.Linear(self.hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.input_projection.weight)
        torch.nn.init.zeros_(self.input_projection.bias)
        if isinstance(self.sequence_encoder, torch.nn.GRU):
            for name, parameter in self.sequence_encoder.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(parameter)
                else:
                    torch.nn.init.zeros_(parameter)
        torch.nn.init.xavier_uniform_(self.risk_head.weight)
        torch.nn.init.zeros_(self.risk_head.bias)

    def forward(self, features: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected features with shape [B, T, D], got {tuple(features.shape)}")
        if sequence_mask.shape != features.shape[:2]:
            raise ValueError("sequence_mask must have shape [B, T]")
        encoded_inputs = torch.nn.functional.gelu(self.input_projection(features))
        if isinstance(self.sequence_encoder, torch.nn.GRU):
            encoded_states, _ = self.sequence_encoder(encoded_inputs)
        else:
            sequence_length = encoded_inputs.size(1)
            causal_mask = torch.triu(
                torch.ones(
                    (sequence_length, sequence_length),
                    device=encoded_inputs.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            encoded_states = self.sequence_encoder(
                encoded_inputs,
                mask=causal_mask,
                src_key_padding_mask=~sequence_mask,
            )
        encoded_states = encoded_states * sequence_mask.unsqueeze(-1).to(encoded_states.dtype)
        return self.risk_head(encoded_states).squeeze(-1)


@dataclass
class TorchSequencePrefixClassifier:
    module: ContinuousStepViewSequenceRiskHead
    device: str
    predict_batch_size: int = 2048


@dataclass
class SegmentTfidfVectorizer:
    encoder: TfidfSegmentEncoder

    @property
    def vocabulary_(self):
        return self.encoder.vectorizer.vocabulary_

    @property
    def resolved_backend(self) -> str:
        return self.encoder.resolved_backend

    @property
    def output_dim(self) -> int:
        return int(self.encoder.output_dim)

    def fit(self, payloads: Iterable[Payload]) -> None:
        self.encoder.fit(list(payloads))

    def transform(self, payloads: Sequence[Payload], *, progress_label: str | None = None):
        if not payloads:
            return sparse.csr_matrix((0, self.output_dim), dtype=np.float32)
        encoded = self.encoder.encode(
            list(payloads),
            device=_tfidf_encode_device(self.encoder),
            progress_label=progress_label,
        ).embeddings
        return _torch_to_scipy_csr(encoded)

    def transform_payload_groups(
        self,
        payload_groups: Sequence[Sequence[Payload]],
        *,
        progress_label: str | None = None,
    ) -> list[sparse.csr_matrix]:
        if not payload_groups:
            return []
        lengths = [len(payloads) for payloads in payload_groups]
        flat_payloads = [payload for payloads in payload_groups for payload in payloads]
        if not flat_payloads:
            empty = sparse.csr_matrix((0, self.output_dim), dtype=np.float32)
            return [empty for _ in payload_groups]
        encoded_batches = _iter_encoded_payload_batches_for_runtime_cache(
            self.encoder,
            flat_payloads,
            device=_tfidf_encode_device(self.encoder),
            progress_label=progress_label,
        )
        flat_matrix = sparse.vstack(
            [_torch_to_scipy_csr(batch) for batch in encoded_batches],
            format="csr",
        )
        matrices: list[sparse.csr_matrix] = []
        cursor = 0
        for length in lengths:
            matrices.append(flat_matrix[cursor : cursor + length])
            cursor += length
        return matrices

    def transform_trajectories(
        self,
        trajectories: Sequence[TrajectoryRecord],
        *,
        runtime_cache: RuntimeCache,
        representation_mode: str,
        max_observation_lines: int,
        step_view_frontend: str = "inferred",
        tau2_refinement_profile: str | None = None,
        skillsbench_process_profile: str | None = None,
        progress_label: str | None = None,
    ) -> list[sparse.csr_matrix]:
        if not trajectories:
            return []
        encoded, lengths, _ = encode_trajectories(
            self.encoder,
            list(trajectories),
            device=_tfidf_encode_device(self.encoder),
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            runtime_cache=runtime_cache,
            progress_label=progress_label,
        )
        return [
            _torch_to_scipy_csr(trajectory_embeddings)
            for trajectory_embeddings in split_encoded_trajectories(encoded.detach().cpu(), lengths)
        ]


def _payload_to_baseline_text(payload: str | StepView, *, step_view_text_mode: str) -> str:
    if isinstance(payload, StepView):
        return payload.render_text(step_view_text_mode)
    return payload


def _to_scipy_csr(features):
    if sparse.issparse(features):
        return features.tocsr().astype(np.float32)
    if hasattr(features, "get"):
        features = features.get()
        if sparse.issparse(features):
            return features.tocsr().astype(np.float32)
    if hasattr(features, "toarray"):
        return sparse.csr_matrix(np.asarray(features.toarray(), dtype=np.float32))
    return sparse.csr_matrix(np.asarray(features, dtype=np.float32))


def _torch_to_scipy_csr(encoded: torch.Tensor):
    if encoded.layout in {torch.sparse_coo, torch.sparse_csr}:
        coo = encoded.to_sparse_coo().coalesce().cpu()
        indices = coo.indices().numpy()
        values = coo.values().numpy().astype(np.float32, copy=False)
        return sparse.coo_matrix(
            (values, (indices[0], indices[1])),
            shape=tuple(coo.shape),
        ).tocsr()
    dense = encoded.detach().cpu().numpy().astype(np.float32, copy=False)
    return sparse.csr_matrix(dense)


def _tfidf_encode_device(encoder: TfidfSegmentEncoder) -> torch.device:
    if encoder.resolved_backend == "cuml" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _step_payloads_for_trajectory(
    trajectory: TrajectoryRecord,
    *,
    representation_mode: str,
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    step_view_text_mode: str = "full",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    runtime_cache: RuntimeCache | None = None,
) -> tuple[Payload, ...]:
    if runtime_cache is not None:
        return runtime_cache.get_payloads(
            trajectory,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
    return tuple(
        build_step_payload(
            step,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            dataset_name=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        for step in trajectory.steps
    )


def assert_no_forbidden_feature_markers(
    texts: Iterable[str],
    *,
    forbidden_markers: Sequence[str] = FORBIDDEN_FEATURE_MARKERS,
) -> None:
    if tuple(forbidden_markers) == FORBIDDEN_FEATURE_MARKERS:
        pattern = FORBIDDEN_FEATURE_LINE_RE
    else:
        pattern = re.compile(
            r"^\s*("
            + "|".join(re.escape(marker) for marker in forbidden_markers)
            + r")\s*[:=]",
            re.IGNORECASE,
        )
    for index, text in enumerate(texts):
        for line in text.splitlines():
            match = pattern.search(line)
            if match:
                raise ValueError(
                    f"Forbidden future/label marker {match.group(1)!r} found in feature text {index}"
                )


def build_prefix_examples(
    trajectories: list[TrajectoryRecord],
    *,
    horizon: int,
    representation_mode: str,
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    step_view_text_mode: str = "full",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    audit_features: bool = True,
    runtime_cache: RuntimeCache | None = None,
) -> list[PrefixExample]:
    examples: list[PrefixExample] = []
    cache = runtime_cache or RuntimeCache()
    for trajectory in trajectories:
        step_payloads = _step_payloads_for_trajectory(
            trajectory,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            step_view_text_mode=step_view_text_mode,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            runtime_cache=cache,
        )
        labels = cache.get_future_failure_labels(
            trajectory,
            horizon=horizon,
        )
        for prefix_index, label in enumerate(labels, start=1):
            examples.append(
                PrefixExample(
                    trajectory_id=trajectory.trajectory_id,
                    task_id=trajectory.task_id,
                    split=trajectory.split,
                    prefix_index=prefix_index,
                    label=int(label),
                    trajectory_step_texts=step_payloads,
                    step_view_text_mode=step_view_text_mode,
                    trajectory=trajectory,
                )
            )
    if audit_features:
        seen_step_texts: set[int] = set()
        texts_to_audit = []
        for example in examples:
            key = id(example.trajectory_step_texts)
            if key in seen_step_texts:
                continue
            seen_step_texts.add(key)
            texts_to_audit.extend(
                _payload_to_baseline_text(
                    payload,
                    step_view_text_mode=step_view_text_mode,
                )
                for payload in example.trajectory_step_texts
            )
        assert_no_forbidden_feature_markers(texts_to_audit)
    return examples


def labels_for_examples(examples: Sequence[PrefixExample]) -> list[int]:
    return [int(example.label) for example in examples]


def documents_for_examples(examples: Sequence[PrefixExample]) -> list[str]:
    return [example.document for example in examples]


def _iter_distinct_trajectory_step_payloads(examples: Sequence[PrefixExample]) -> Iterable[Payload]:
    seen: set[int] = set()
    for example in examples:
        key = id(example.trajectory_step_texts)
        if key in seen:
            continue
        seen.add(key)
        yield from example.trajectory_step_texts


def fit_prefix_tfidf_vectorizer(
    examples: Sequence[PrefixExample],
    *,
    max_features: int,
    tfidf_backend: str = "auto",
    step_view_text_mode: str = "full",
    tfidf_metadata_sidechannel: str | None = None,
) -> SegmentTfidfVectorizer:
    vectorizer = SegmentTfidfVectorizer(
        TfidfSegmentEncoder(
            max_features=max_features,
            backend=tfidf_backend,
            step_view_text_mode=step_view_text_mode,
            sparse_output=True,
            metadata_sidechannel_mode=tfidf_metadata_sidechannel,
        )
    )
    vectorizer.fit(_iter_distinct_trajectory_step_payloads(examples))
    return vectorizer


def transform_prefix_documents(
    examples: Sequence[PrefixExample],
    vectorizer: SegmentTfidfVectorizer,
    *,
    runtime_cache: RuntimeCache | None = None,
    representation_mode: str = "legacy",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
):
    return build_prefix_tfidf_features(
        examples,
        vectorizer,
        runtime_cache=runtime_cache,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
    )


def fit_stepview_tfidf_vectorizer(
    examples: Sequence[PrefixExample],
    *,
    max_features: int,
    tfidf_backend: str = "auto",
    step_view_text_mode: str = "full",
    tfidf_metadata_sidechannel: str | None = None,
) -> SegmentTfidfVectorizer:
    vectorizer = SegmentTfidfVectorizer(
        TfidfSegmentEncoder(
            max_features=max_features,
            backend=tfidf_backend,
            step_view_text_mode=step_view_text_mode,
            sparse_output=True,
            metadata_sidechannel_mode=tfidf_metadata_sidechannel,
        )
    )
    vectorizer.fit(_iter_distinct_trajectory_step_payloads(examples))
    return vectorizer


def _group_examples_by_trajectory(
    examples: Sequence[PrefixExample],
) -> dict[int, list[tuple[int, PrefixExample]]]:
    groups: dict[int, list[tuple[int, PrefixExample]]] = {}
    for row_index, example in enumerate(examples):
        groups.setdefault(id(example.trajectory_step_texts), []).append((row_index, example))
    return groups


def build_prefix_tfidf_features(
    examples: Sequence[PrefixExample],
    vectorizer: SegmentTfidfVectorizer,
    *,
    runtime_cache: RuntimeCache | None = None,
    representation_mode: str = "legacy",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
):
    if not examples:
        return sparse.csr_matrix((0, vectorizer.output_dim), dtype=np.float32)
    rows: list[sparse.csr_matrix | None] = [None] * len(examples)
    grouped_examples = list(_group_examples_by_trajectory(examples).values())
    group_trajectories = [grouped[0][1].trajectory for grouped in grouped_examples]
    if runtime_cache is not None and all(trajectory is not None for trajectory in group_trajectories):
        step_matrices = vectorizer.transform_trajectories(
            [trajectory for trajectory in group_trajectories if trajectory is not None],
            runtime_cache=runtime_cache,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            progress_label="supervised-prefix/prefix-step-tfidf",
        )
    else:
        step_matrices = vectorizer.transform_payload_groups(
            [grouped[0][1].trajectory_step_texts for grouped in grouped_examples],
            progress_label="supervised-prefix/prefix-step-tfidf",
        )
    for grouped, step_matrix in zip(grouped_examples, step_matrices):
        running = sparse.csr_matrix((1, vectorizer.output_dim), dtype=np.float32)
        cursor = 0
        for row_index, example in sorted(grouped, key=lambda item: item[1].prefix_index):
            while cursor < example.prefix_index:
                running = running + step_matrix[cursor]
                cursor += 1
            rows[row_index] = normalize(running, norm="l2", copy=True)
    return sparse.vstack([row for row in rows if row is not None], format="csr")


def build_stepview_pooled_prefix_features(
    examples: Sequence[PrefixExample],
    vectorizer: SegmentTfidfVectorizer,
    *,
    runtime_cache: RuntimeCache | None = None,
    representation_mode: str = "reduced-dense",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
):
    rows: list[sparse.csr_matrix | None] = [None] * len(examples)
    feature_dim = vectorizer.output_dim
    empty = sparse.csr_matrix((1, feature_dim), dtype=np.float32)
    grouped_examples = list(_group_examples_by_trajectory(examples).values())
    group_trajectories = [grouped[0][1].trajectory for grouped in grouped_examples]
    if runtime_cache is not None and all(trajectory is not None for trajectory in group_trajectories):
        step_matrices = vectorizer.transform_trajectories(
            [trajectory for trajectory in group_trajectories if trajectory is not None],
            runtime_cache=runtime_cache,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            progress_label="supervised-prefix/pooled-step-tfidf",
        )
    else:
        step_matrices = vectorizer.transform_payload_groups(
            [grouped[0][1].trajectory_step_texts for grouped in grouped_examples],
            progress_label="supervised-prefix/pooled-step-tfidf",
        )
    for grouped, step_matrix in zip(grouped_examples, step_matrices):
        running_sum = sparse.csr_matrix((1, feature_dim), dtype=np.float32)
        running_max = sparse.csr_matrix((1, feature_dim), dtype=np.float32)
        cursor = 0
        for row_index, example in sorted(grouped, key=lambda item: item[1].prefix_index):
            while cursor < example.prefix_index:
                step_row = step_matrix[cursor]
                running_sum = running_sum + step_row
                running_max = running_max.maximum(step_row)
                cursor += 1
            if example.prefix_index == 0:
                rows[row_index] = sparse.hstack([empty, empty, empty], format="csr")
                continue
            last = step_matrix[example.prefix_index - 1]
            mean = running_sum * (1.0 / float(example.prefix_index))
            rows[row_index] = sparse.hstack([last, mean, running_max], format="csr")
    if not rows:
        return sparse.csr_matrix((0, feature_dim * 3), dtype=np.float32)
    return sparse.vstack([row for row in rows if row is not None], format="csr")


def _iter_example_group_batches(
    examples: Sequence[PrefixExample],
    *,
    max_prefixes_per_batch: int,
    shuffle: bool = False,
    seed: int | None = None,
) -> Iterable[list[list[tuple[int, PrefixExample]]]]:
    if max_prefixes_per_batch <= 0:
        raise ValueError("max_prefixes_per_batch must be positive")
    groups = list(_group_examples_by_trajectory(examples).values())
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
    batch: list[list[tuple[int, PrefixExample]]] = []
    batch_prefix_count = 0
    for group in groups:
        group_prefix_count = len(group)
        if batch and batch_prefix_count + group_prefix_count > max_prefixes_per_batch:
            yield batch
            batch = []
            batch_prefix_count = 0
        batch.append(group)
        batch_prefix_count += group_prefix_count
    if batch:
        yield batch


def _examples_for_group_batch(
    group_batch: Sequence[Sequence[tuple[int, PrefixExample]]],
) -> tuple[list[int], list[PrefixExample]]:
    pairs = sorted(
        [pair for group in group_batch for pair in group],
        key=lambda item: item[0],
    )
    row_indices = [row_index for row_index, _ in pairs]
    batch_examples = [example for _, example in pairs]
    return row_indices, batch_examples


def iter_prefix_tfidf_feature_batches(
    examples: Sequence[PrefixExample],
    vectorizer: SegmentTfidfVectorizer,
    *,
    max_prefixes_per_batch: int = 2048,
    shuffle: bool = False,
    seed: int | None = None,
    runtime_cache: RuntimeCache | None = None,
    representation_mode: str = "legacy",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
) -> Iterable[FeatureBatch]:
    for group_batch in _iter_example_group_batches(
        examples,
        max_prefixes_per_batch=max_prefixes_per_batch,
        shuffle=shuffle,
        seed=seed,
    ):
        row_indices, batch_examples = _examples_for_group_batch(group_batch)
        features = build_prefix_tfidf_features(
            batch_examples,
            vectorizer,
            runtime_cache=runtime_cache,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        yield FeatureBatch(
            features=features,
            labels=labels_for_examples(batch_examples),
            row_indices=row_indices,
        )


def iter_stepview_pooled_feature_batches(
    examples: Sequence[PrefixExample],
    vectorizer: SegmentTfidfVectorizer,
    *,
    max_prefixes_per_batch: int = 2048,
    shuffle: bool = False,
    seed: int | None = None,
    runtime_cache: RuntimeCache | None = None,
    representation_mode: str = "reduced-dense",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
) -> Iterable[FeatureBatch]:
    for group_batch in _iter_example_group_batches(
        examples,
        max_prefixes_per_batch=max_prefixes_per_batch,
        shuffle=shuffle,
        seed=seed,
    ):
        row_indices, batch_examples = _examples_for_group_batch(group_batch)
        features = build_stepview_pooled_prefix_features(
            batch_examples,
            vectorizer,
            runtime_cache=runtime_cache,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        yield FeatureBatch(
            features=features,
            labels=labels_for_examples(batch_examples),
            row_indices=row_indices,
        )


def iter_stepview_sequence_feature_batches(
    examples: Sequence[PrefixExample],
    vectorizer: SegmentTfidfVectorizer,
    *,
    max_prefixes_per_batch: int = 2048,
    shuffle: bool = False,
    seed: int | None = None,
    runtime_cache: RuntimeCache | None = None,
    representation_mode: str = "reduced-dense",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
) -> Iterable[SequenceFeatureBatch]:
    for group_batch in _iter_example_group_batches(
        examples,
        max_prefixes_per_batch=max_prefixes_per_batch,
        shuffle=shuffle,
        seed=seed,
    ):
        group_trajectories = [grouped[0][1].trajectory for grouped in group_batch]
        if runtime_cache is not None and all(trajectory is not None for trajectory in group_trajectories):
            step_matrices = vectorizer.transform_trajectories(
                [trajectory for trajectory in group_trajectories if trajectory is not None],
                runtime_cache=runtime_cache,
                representation_mode=representation_mode,
                max_observation_lines=max_observation_lines,
                step_view_frontend=step_view_frontend,
                tau2_refinement_profile=tau2_refinement_profile,
                skillsbench_process_profile=skillsbench_process_profile,
                progress_label="supervised-prefix/sequence-step-tfidf",
            )
        else:
            step_matrices = vectorizer.transform_payload_groups(
                [grouped[0][1].trajectory_step_texts for grouped in group_batch],
                progress_label="supervised-prefix/sequence-step-tfidf",
            )

        batch_size = len(group_batch)
        max_steps = max((matrix.shape[0] for matrix in step_matrices), default=0)
        feature_dim = vectorizer.output_dim
        features = np.zeros((batch_size, max_steps, feature_dim), dtype=np.float32)
        sequence_mask = np.zeros((batch_size, max_steps), dtype=bool)
        label_mask = np.zeros((batch_size, max_steps), dtype=bool)
        labels = np.zeros((batch_size, max_steps), dtype=np.float32)
        row_indices: list[int] = []
        for batch_index, (grouped, step_matrix) in enumerate(zip(group_batch, step_matrices)):
            step_count = step_matrix.shape[0]
            if step_count:
                features[batch_index, :step_count] = step_matrix.toarray()
                sequence_mask[batch_index, :step_count] = True
            for row_index, example in sorted(grouped, key=lambda item: item[1].prefix_index):
                position = example.prefix_index - 1
                if position < 0 or position >= step_count:
                    raise ValueError(
                        f"prefix_index={example.prefix_index} is outside trajectory length {step_count}"
                    )
                label_mask[batch_index, position] = True
                labels[batch_index, position] = float(example.label)
                row_indices.append(row_index)

        yield SequenceFeatureBatch(
            features=torch.as_tensor(features, dtype=torch.float32),
            sequence_mask=torch.as_tensor(sequence_mask, dtype=torch.bool),
            label_mask=torch.as_tensor(label_mask, dtype=torch.bool),
            labels=torch.as_tensor(labels, dtype=torch.float32),
            row_indices=row_indices,
        )


def _predict_scores(model: object, features) -> list[float]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.shape[1] == 1:
            return [float(probabilities[0, 0])] * features.shape[0]
        return [float(value) for value in probabilities[:, 1]]
    if hasattr(model, "decision_function"):
        margins = np.asarray(model.decision_function(features), dtype=np.float64)
        return [float(value) for value in 1.0 / (1.0 + np.exp(-margins))]
    raise TypeError(f"Model {type(model)!r} does not expose scores")


def compute_prefix_metrics(
    examples: Sequence[PrefixExample],
    scores: Sequence[float],
    *,
    threshold_scores: Sequence[float] | None = None,
    threshold_labels: Sequence[int] | None = None,
    calibration_bins: int = 10,
    threshold_selection_split: str = "unavailable",
) -> dict[str, float | int | str]:
    labels = labels_for_examples(examples)
    score_list = [float(score) for score in scores]
    if len(score_list) != len(labels):
        raise ValueError(f"score count {len(score_list)} does not match labels {len(labels)}")
    auroc = roc_auc_score(labels, score_list) if len(set(labels)) > 1 else 0.5
    auprc = average_precision_score(labels, score_list) if labels else 0.0
    threshold_source_scores = (
        [float(score) for score in threshold_scores]
        if threshold_scores is not None
        else score_list
    )
    threshold_source_labels = (
        [int(label) for label in threshold_labels]
        if threshold_labels is not None
        else labels
    )
    threshold = _select_threshold(threshold_source_scores, threshold_source_labels)
    predictions = [int(score >= threshold) for score in score_list]
    true_positive_alerts = sum(1 for pred, label in zip(predictions, labels) if pred and label)
    false_positive_alerts = sum(1 for pred, label in zip(predictions, labels) if pred and not label)
    positive_count = int(sum(labels))
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "soft_auroc": float(auroc),
        "soft_auprc": float(auprc),
        "calibration_error": float(
            _compute_expected_calibration_error(score_list, labels, num_bins=calibration_bins)
        ),
        "ece": float(_compute_expected_calibration_error(score_list, labels, num_bins=calibration_bins)),
        "brier_score": float(_compute_brier_score(score_list, labels)),
        "threshold": float(threshold),
        "threshold_selection_split": threshold_selection_split,
        "prefix_count": int(len(labels)),
        "positive_prefix_count": int(positive_count),
        "negative_prefix_count": int(len(labels) - positive_count),
        "positive_prefix_rate": float(positive_count / len(labels)) if labels else 0.0,
        "true_positive_alert_count": int(true_positive_alerts),
        "false_positive_alert_count": int(false_positive_alerts),
        "calibration_bins": int(calibration_bins),
    }


def _resolve_fast_device(fast_device: str | None) -> torch.device:
    if fast_device in {None, "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(fast_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("fast_device='cuda' requested but CUDA is unavailable")
    return device


def _make_torch_binary_model(input_dim: int, hidden_layers: tuple[int, ...]) -> torch.nn.Module:
    if not hidden_layers:
        return torch.nn.Linear(input_dim, 1)
    layers: list[torch.nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(torch.nn.Linear(previous_dim, int(hidden_dim)))
        layers.append(torch.nn.ReLU())
        previous_dim = int(hidden_dim)
    layers.append(torch.nn.Linear(previous_dim, 1))
    return torch.nn.Sequential(*layers)


def _positive_class_weight(labels: Sequence[int], device: torch.device) -> torch.Tensor:
    positive_count = max(1, int(sum(labels)))
    negative_count = max(1, int(len(labels) - sum(labels)))
    return torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)


def _train_torch_binary_model(
    model: torch.nn.Module,
    *,
    train_examples: Sequence[PrefixExample],
    vectorizer: SegmentTfidfVectorizer,
    batch_iterator,
    runtime_cache: RuntimeCache | None,
    representation_mode: str,
    max_observation_lines: int,
    step_view_frontend: str,
    tau2_refinement_profile: str | None,
    skillsbench_process_profile: str | None,
    seed: int,
    fast_epochs: int,
    feature_batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    if fast_epochs <= 0:
        raise ValueError("fast_epochs must be positive")
    if feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be positive")
    torch.manual_seed(seed)
    model.to(device)
    labels = labels_for_examples(train_examples)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=_positive_class_weight(labels, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    for epoch in range(fast_epochs):
        for batch in batch_iterator(
            train_examples,
            vectorizer,
            max_prefixes_per_batch=feature_batch_size,
            shuffle=True,
            seed=seed + epoch,
            runtime_cache=runtime_cache,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        ):
            features = batch.features.toarray()
            inputs = torch.as_tensor(features, dtype=torch.float32, device=device)
            targets = torch.as_tensor(batch.labels, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs).reshape(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()


def _score_feature_batches(
    model: object,
    examples: Sequence[PrefixExample],
    vectorizer: SegmentTfidfVectorizer,
    *,
    batch_iterator,
    feature_batch_size: int,
    runtime_cache: RuntimeCache | None,
    representation_mode: str,
    max_observation_lines: int,
    step_view_frontend: str,
    tau2_refinement_profile: str | None,
    skillsbench_process_profile: str | None,
) -> list[float]:
    scores = [0.0] * len(examples)
    seen = [False] * len(examples)
    for batch in batch_iterator(
        examples,
        vectorizer,
        max_prefixes_per_batch=feature_batch_size,
        runtime_cache=runtime_cache,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
    ):
        batch_scores = _predict_scores(model, batch.features)
        for row_index, score in zip(batch.row_indices, batch_scores):
            scores[row_index] = float(score)
            seen[row_index] = True
    if not all(seen):
        missing = [index for index, was_seen in enumerate(seen) if not was_seen]
        raise RuntimeError(f"missing streamed scores for prefix rows: {missing[:5]}")
    return scores


def _train_torch_sequence_model(
    model: ContinuousStepViewSequenceRiskHead,
    *,
    train_examples: Sequence[PrefixExample],
    vectorizer: SegmentTfidfVectorizer,
    runtime_cache: RuntimeCache | None,
    representation_mode: str,
    max_observation_lines: int,
    step_view_frontend: str,
    tau2_refinement_profile: str | None,
    skillsbench_process_profile: str | None,
    seed: int,
    fast_epochs: int,
    feature_batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    if fast_epochs <= 0:
        raise ValueError("fast_epochs must be positive")
    if feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be positive")
    torch.manual_seed(seed)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=_positive_class_weight(labels_for_examples(train_examples), device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    for epoch in range(fast_epochs):
        for batch in iter_stepview_sequence_feature_batches(
            train_examples,
            vectorizer,
            max_prefixes_per_batch=feature_batch_size,
            shuffle=True,
            seed=seed + epoch,
            runtime_cache=runtime_cache,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        ):
            inputs = batch.features.to(device)
            sequence_mask = batch.sequence_mask.to(device)
            label_mask = batch.label_mask.to(device)
            targets = batch.labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, sequence_mask)
            loss = criterion(logits[label_mask], targets[label_mask])
            loss.backward()
            optimizer.step()


def _score_sequence_feature_batches(
    model: TorchSequencePrefixClassifier,
    examples: Sequence[PrefixExample],
    vectorizer: SegmentTfidfVectorizer,
    *,
    feature_batch_size: int,
    runtime_cache: RuntimeCache | None,
    representation_mode: str,
    max_observation_lines: int,
    step_view_frontend: str,
    tau2_refinement_profile: str | None,
    skillsbench_process_profile: str | None,
) -> list[float]:
    scores = [0.0] * len(examples)
    seen = [False] * len(examples)
    device = torch.device(model.device)
    model.module.eval()
    with torch.no_grad():
        for batch in iter_stepview_sequence_feature_batches(
            examples,
            vectorizer,
            max_prefixes_per_batch=feature_batch_size,
            runtime_cache=runtime_cache,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        ):
            logits = model.module(
                batch.features.to(device),
                batch.sequence_mask.to(device),
            )
            batch_scores = torch.sigmoid(logits[batch.label_mask.to(device)]).detach().cpu().numpy()
            if len(batch_scores) != len(batch.row_indices):
                raise RuntimeError(
                    f"sequence score count {len(batch_scores)} does not match rows {len(batch.row_indices)}"
                )
            for row_index, score in zip(batch.row_indices, batch_scores):
                scores[row_index] = float(score)
                seen[row_index] = True
    if not all(seen):
        missing = [index for index, was_seen in enumerate(seen) if not was_seen]
        raise RuntimeError(f"missing streamed sequence scores for prefix rows: {missing[:5]}")
    return scores


def _train_fast_torch_baseline(
    train_examples: Sequence[PrefixExample],
    val_examples: Sequence[PrefixExample],
    *,
    vectorizer: SegmentTfidfVectorizer,
    hidden_layers: tuple[int, ...],
    model_type: str,
    batch_iterator,
    calibration_examples: Sequence[PrefixExample] | None,
    calibration_bins: int,
    config: SupervisedBaselineConfig,
    runtime_cache: RuntimeCache | None,
    representation_mode: str,
    max_observation_lines: int,
    step_view_frontend: str,
    tau2_refinement_profile: str | None,
    skillsbench_process_profile: str | None,
    seed: int,
    fast_epochs: int,
    feature_batch_size: int,
    learning_rate: float,
    fast_device: str | None,
) -> TrainedBaseline:
    device = _resolve_fast_device(fast_device)
    model = _make_torch_binary_model(vectorizer.output_dim * (3 if "pooled" in model_type else 1), hidden_layers)
    _train_torch_binary_model(
        model,
        train_examples=train_examples,
        vectorizer=vectorizer,
        batch_iterator=batch_iterator,
        runtime_cache=runtime_cache,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
        seed=seed,
        fast_epochs=fast_epochs,
        feature_batch_size=feature_batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    torch_model = TorchBinaryClassifier(
        module=model,
        device=str(device),
        predict_batch_size=feature_batch_size,
    )
    val_scores = _score_feature_batches(
        torch_model,
        val_examples,
        vectorizer,
        batch_iterator=batch_iterator,
        feature_batch_size=feature_batch_size,
        runtime_cache=runtime_cache,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
    )
    cal_scores = (
        _score_feature_batches(
            torch_model,
            calibration_examples,
            vectorizer,
            batch_iterator=batch_iterator,
            feature_batch_size=feature_batch_size,
            runtime_cache=runtime_cache,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        if calibration_examples is not None
        else None
    )
    cal_labels = labels_for_examples(calibration_examples) if calibration_examples is not None else None
    validation_metrics = compute_prefix_metrics(
        val_examples,
        val_scores,
        threshold_scores=cal_scores,
        threshold_labels=cal_labels,
        calibration_bins=calibration_bins,
        threshold_selection_split="cal" if calibration_examples is not None else "val",
    )
    return TrainedBaseline(
        model_type=model_type,
        vectorizer=vectorizer,
        model=torch_model,
        config=config,
        selected_c=None,
        validation_metrics=validation_metrics,
    )


def train_tfidf_linear_baseline(
    train_examples: Sequence[PrefixExample],
    val_examples: Sequence[PrefixExample],
    *,
    max_features: int = 4096,
    tfidf_backend: str = "auto",
    step_view_text_mode: str = "full",
    tfidf_metadata_sidechannel: str | None = None,
    seed: int = 13,
    c_values: Sequence[float] = (0.1, 1.0, 10.0),
    classifier: str = "logistic",
    calibration_examples: Sequence[PrefixExample] | None = None,
    calibration_bins: int = 10,
    config: SupervisedBaselineConfig | None = None,
    runtime_cache: RuntimeCache | None = None,
    representation_mode: str = "legacy",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    fast_runtime: str | None = None,
    fast_epochs: int = 3,
    feature_batch_size: int = 2048,
    learning_rate: float = 1e-3,
    fast_device: str | None = None,
) -> TrainedBaseline:
    if classifier not in {"logistic", "linear-svm"}:
        raise ValueError(f"Unsupported linear classifier: {classifier}")
    if fast_runtime is not None and classifier != "logistic":
        raise ValueError("fast_runtime is currently supported only for tfidf logistic")
    if fast_runtime not in {None, "torch-minibatch"}:
        raise ValueError(f"Unsupported fast_runtime: {fast_runtime}")
    vectorizer = fit_prefix_tfidf_vectorizer(
        train_examples,
        max_features=max_features,
        tfidf_backend=tfidf_backend,
        step_view_text_mode=step_view_text_mode,
        tfidf_metadata_sidechannel=tfidf_metadata_sidechannel,
    )
    if fast_runtime == "torch-minibatch":
        fast_config = config or SupervisedBaselineConfig(
            model_type=f"tfidf-prefix-{classifier}-fast-torch",
            representation_mode=representation_mode,
            max_features=max_features,
            tfidf_backend=tfidf_backend,
            step_view_text_mode=step_view_text_mode,
            step_view_frontend=step_view_frontend,
            max_observation_lines=max_observation_lines,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            tfidf_metadata_sidechannel=tfidf_metadata_sidechannel,
            seed=seed,
            calibration_bins=calibration_bins,
            fast_runtime=fast_runtime,
            fast_epochs=fast_epochs,
            feature_batch_size=feature_batch_size,
            fast_device=fast_device or "auto",
        )
        return _train_fast_torch_baseline(
            train_examples,
            val_examples,
            vectorizer=vectorizer,
            hidden_layers=(),
            model_type=f"tfidf-prefix-{classifier}-fast-torch",
            batch_iterator=iter_prefix_tfidf_feature_batches,
            calibration_examples=calibration_examples,
            calibration_bins=calibration_bins,
            config=fast_config,
            runtime_cache=runtime_cache,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            seed=seed,
            fast_epochs=fast_epochs,
            feature_batch_size=feature_batch_size,
            learning_rate=learning_rate,
            fast_device=fast_device,
        )
    train_features = transform_prefix_documents(
        train_examples,
        vectorizer,
        runtime_cache=runtime_cache,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
    )
    train_labels = labels_for_examples(train_examples)
    val_features = transform_prefix_documents(
        val_examples,
        vectorizer,
        runtime_cache=runtime_cache,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
    )
    val_labels = labels_for_examples(val_examples)
    cal_features = (
        transform_prefix_documents(
            calibration_examples,
            vectorizer,
            runtime_cache=runtime_cache,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        if calibration_examples is not None
        else None
    )
    cal_labels = (
        labels_for_examples(calibration_examples)
        if calibration_examples is not None
        else None
    )

    best_model: object | None = None
    best_c: float | None = None
    best_metrics: dict[str, float | int | str] | None = None
    best_score = -np.inf
    for c_value in c_values:
        if classifier == "logistic":
            model: object = LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
            )
            model.fit(train_features, train_labels)
        else:
            if cal_features is None or cal_labels is None:
                raise ValueError("linear-svm requires calibration_examples for probability calibration")
            base = LinearSVC(C=float(c_value), class_weight="balanced", random_state=seed)
            base.fit(train_features, train_labels)
            model = CalibratedClassifierCV(base, cv="prefit", method="sigmoid")
            model.fit(cal_features, cal_labels)
        val_scores = _predict_scores(model, val_features)
        metrics = compute_prefix_metrics(
            val_examples,
            val_scores,
            threshold_scores=_predict_scores(model, cal_features) if cal_features is not None else None,
            threshold_labels=cal_labels,
            calibration_bins=calibration_bins,
            threshold_selection_split="cal" if cal_features is not None else "val",
        )
        if float(metrics["auprc"]) > best_score:
            best_model = model
            best_c = float(c_value)
            best_metrics = metrics
            best_score = float(metrics["auprc"])
    assert best_model is not None and best_metrics is not None
    return TrainedBaseline(
        model_type=f"tfidf-prefix-{classifier}",
        vectorizer=vectorizer,
        model=best_model,
        config=config
        or SupervisedBaselineConfig(
            model_type=f"tfidf-prefix-{classifier}",
            representation_mode=representation_mode,
            max_features=max_features,
            tfidf_backend=tfidf_backend,
            step_view_text_mode=step_view_text_mode,
            step_view_frontend=step_view_frontend,
            max_observation_lines=max_observation_lines,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            tfidf_metadata_sidechannel=tfidf_metadata_sidechannel,
            seed=seed,
            calibration_bins=calibration_bins,
        ),
        selected_c=best_c,
        validation_metrics=best_metrics,
    )


def train_stepview_pooled_mlp_baseline(
    train_examples: Sequence[PrefixExample],
    val_examples: Sequence[PrefixExample],
    *,
    max_features: int = 4096,
    tfidf_backend: str = "auto",
    step_view_text_mode: str = "full",
    tfidf_metadata_sidechannel: str | None = None,
    seed: int = 13,
    hidden_layer_sizes: Sequence[tuple[int, ...]] = ((64,),),
    calibration_examples: Sequence[PrefixExample] | None = None,
    calibration_bins: int = 10,
    config: SupervisedBaselineConfig | None = None,
    runtime_cache: RuntimeCache | None = None,
    representation_mode: str = "reduced-dense",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    fast_runtime: str | None = None,
    fast_epochs: int = 3,
    feature_batch_size: int = 2048,
    learning_rate: float = 1e-3,
    fast_device: str | None = None,
) -> TrainedBaseline:
    if fast_runtime not in {None, "torch-minibatch"}:
        raise ValueError(f"Unsupported fast_runtime: {fast_runtime}")
    vectorizer = fit_stepview_tfidf_vectorizer(
        train_examples,
        max_features=max_features,
        tfidf_backend=tfidf_backend,
        step_view_text_mode=step_view_text_mode,
        tfidf_metadata_sidechannel=tfidf_metadata_sidechannel,
    )
    if fast_runtime == "torch-minibatch":
        layers = tuple(hidden_layer_sizes[0])
        fast_config = config or SupervisedBaselineConfig(
            model_type="stepview-pooled-mlp-fast-torch",
            representation_mode=representation_mode,
            max_features=max_features,
            tfidf_backend=tfidf_backend,
            step_view_text_mode=step_view_text_mode,
            step_view_frontend=step_view_frontend,
            max_observation_lines=max_observation_lines,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            tfidf_metadata_sidechannel=tfidf_metadata_sidechannel,
            seed=seed,
            calibration_bins=calibration_bins,
            fast_runtime=fast_runtime,
            fast_epochs=fast_epochs,
            feature_batch_size=feature_batch_size,
            fast_device=fast_device or "auto",
        )
        trained = _train_fast_torch_baseline(
            train_examples,
            val_examples,
            vectorizer=vectorizer,
            hidden_layers=layers,
            model_type="stepview-pooled-mlp-fast-torch",
            batch_iterator=iter_stepview_pooled_feature_batches,
            calibration_examples=calibration_examples,
            calibration_bins=calibration_bins,
            config=fast_config,
            runtime_cache=runtime_cache,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            seed=seed,
            fast_epochs=fast_epochs,
            feature_batch_size=feature_batch_size,
            learning_rate=learning_rate,
            fast_device=fast_device,
        )
        trained.validation_metrics["hidden_layer_sizes"] = str(layers)
        return trained
    train_features = build_stepview_pooled_prefix_features(
        train_examples,
        vectorizer,
        runtime_cache=runtime_cache,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
    )
    train_labels = labels_for_examples(train_examples)
    val_features = build_stepview_pooled_prefix_features(
        val_examples,
        vectorizer,
        runtime_cache=runtime_cache,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
    )
    cal_features = (
        build_stepview_pooled_prefix_features(
            calibration_examples,
            vectorizer,
            runtime_cache=runtime_cache,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        if calibration_examples is not None
        else None
    )
    cal_labels = labels_for_examples(calibration_examples) if calibration_examples is not None else None

    best_model: MLPClassifier | None = None
    best_layers: tuple[int, ...] | None = None
    best_metrics: dict[str, float | int | str] | None = None
    best_score = -np.inf
    for layers in hidden_layer_sizes:
        model = MLPClassifier(
            hidden_layer_sizes=tuple(layers),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=False,
            random_state=seed,
        )
        model.fit(train_features, train_labels)
        val_scores = _predict_scores(model, val_features)
        metrics = compute_prefix_metrics(
            val_examples,
            val_scores,
            threshold_scores=_predict_scores(model, cal_features) if cal_features is not None else None,
            threshold_labels=cal_labels,
            calibration_bins=calibration_bins,
            threshold_selection_split="cal" if cal_features is not None else "val",
        )
        if float(metrics["auprc"]) > best_score:
            best_model = model
            best_layers = tuple(layers)
            best_metrics = metrics
            best_score = float(metrics["auprc"])
    assert best_model is not None and best_metrics is not None
    trained = TrainedBaseline(
        model_type="stepview-pooled-mlp",
        vectorizer=vectorizer,
        model=best_model,
        config=config
        or SupervisedBaselineConfig(
            model_type="stepview-pooled-mlp",
            representation_mode=representation_mode,
            max_features=max_features,
            tfidf_backend=tfidf_backend,
            step_view_text_mode=step_view_text_mode,
            step_view_frontend=step_view_frontend,
            max_observation_lines=max_observation_lines,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            tfidf_metadata_sidechannel=tfidf_metadata_sidechannel,
            seed=seed,
            calibration_bins=calibration_bins,
        ),
        selected_c=None,
        validation_metrics={**best_metrics, "hidden_layer_sizes": str(best_layers)},
    )
    return trained


def train_stepview_sequence_baseline(
    train_examples: Sequence[PrefixExample],
    val_examples: Sequence[PrefixExample],
    *,
    sequence_model: str,
    max_features: int = 4096,
    tfidf_backend: str = "auto",
    step_view_text_mode: str = "full",
    tfidf_metadata_sidechannel: str | None = None,
    seed: int = 13,
    hidden_dim: int = 64,
    num_layers: int = 1,
    calibration_examples: Sequence[PrefixExample] | None = None,
    calibration_bins: int = 10,
    config: SupervisedBaselineConfig | None = None,
    runtime_cache: RuntimeCache | None = None,
    representation_mode: str = "reduced-dense",
    max_observation_lines: int = 8,
    step_view_frontend: str = "inferred",
    tau2_refinement_profile: str | None = None,
    skillsbench_process_profile: str | None = None,
    fast_epochs: int = 3,
    feature_batch_size: int = 2048,
    learning_rate: float = 1e-3,
    fast_device: str | None = None,
) -> TrainedBaseline:
    if sequence_model not in {"gru", "transformer"}:
        raise ValueError(f"Unsupported sequence_model: {sequence_model}")
    model_type = f"stepview-{sequence_model}-continuous"
    vectorizer = fit_stepview_tfidf_vectorizer(
        train_examples,
        max_features=max_features,
        tfidf_backend=tfidf_backend,
        step_view_text_mode=step_view_text_mode,
        tfidf_metadata_sidechannel=tfidf_metadata_sidechannel,
    )
    device = _resolve_fast_device(fast_device)
    torch.manual_seed(seed)
    module = ContinuousStepViewSequenceRiskHead(
        input_dim=vectorizer.output_dim,
        hidden_dim=hidden_dim,
        sequence_model=sequence_model,
        num_layers=num_layers,
    )
    _train_torch_sequence_model(
        module,
        train_examples=train_examples,
        vectorizer=vectorizer,
        runtime_cache=runtime_cache,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
        seed=seed,
        fast_epochs=fast_epochs,
        feature_batch_size=feature_batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    torch_model = TorchSequencePrefixClassifier(
        module=module,
        device=str(device),
        predict_batch_size=feature_batch_size,
    )
    val_scores = _score_sequence_feature_batches(
        torch_model,
        val_examples,
        vectorizer,
        feature_batch_size=feature_batch_size,
        runtime_cache=runtime_cache,
        representation_mode=representation_mode,
        max_observation_lines=max_observation_lines,
        step_view_frontend=step_view_frontend,
        tau2_refinement_profile=tau2_refinement_profile,
        skillsbench_process_profile=skillsbench_process_profile,
    )
    cal_scores = (
        _score_sequence_feature_batches(
            torch_model,
            calibration_examples,
            vectorizer,
            feature_batch_size=feature_batch_size,
            runtime_cache=runtime_cache,
            representation_mode=representation_mode,
            max_observation_lines=max_observation_lines,
            step_view_frontend=step_view_frontend,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
        )
        if calibration_examples is not None
        else None
    )
    cal_labels = labels_for_examples(calibration_examples) if calibration_examples is not None else None
    validation_metrics = compute_prefix_metrics(
        val_examples,
        val_scores,
        threshold_scores=cal_scores,
        threshold_labels=cal_labels,
        calibration_bins=calibration_bins,
        threshold_selection_split="cal" if calibration_examples is not None else "val",
    )
    validation_metrics["sequence_model"] = sequence_model
    validation_metrics["sequence_hidden_dim"] = int(hidden_dim)
    validation_metrics["sequence_num_layers"] = int(num_layers)
    return TrainedBaseline(
        model_type=model_type,
        vectorizer=vectorizer,
        model=torch_model,
        config=config
        or SupervisedBaselineConfig(
            model_type=model_type,
            representation_mode=representation_mode,
            max_features=max_features,
            tfidf_backend=tfidf_backend,
            step_view_text_mode=step_view_text_mode,
            step_view_frontend=step_view_frontend,
            max_observation_lines=max_observation_lines,
            tau2_refinement_profile=tau2_refinement_profile,
            skillsbench_process_profile=skillsbench_process_profile,
            tfidf_metadata_sidechannel=tfidf_metadata_sidechannel,
            seed=seed,
            calibration_bins=calibration_bins,
            fast_runtime="torch-sequence",
            fast_epochs=fast_epochs,
            feature_batch_size=feature_batch_size,
            fast_device=fast_device or "auto",
            sequence_hidden_dim=hidden_dim,
            sequence_num_layers=num_layers,
        ),
        selected_c=None,
        validation_metrics=validation_metrics,
    )


def score_trained_baseline(
    trained: TrainedBaseline,
    examples: Sequence[PrefixExample],
    *,
    calibration_examples: Sequence[PrefixExample] | None = None,
    runtime_cache: RuntimeCache | None = None,
) -> tuple[list[float], dict[str, float | int | str]]:
    feature_kwargs = {
        "runtime_cache": runtime_cache,
        "representation_mode": trained.config.representation_mode,
        "max_observation_lines": trained.config.max_observation_lines,
        "step_view_frontend": trained.config.step_view_frontend,
        "tau2_refinement_profile": trained.config.tau2_refinement_profile,
        "skillsbench_process_profile": trained.config.skillsbench_process_profile,
    }
    if isinstance(trained.model, TorchSequencePrefixClassifier):
        feature_batch_size = trained.config.feature_batch_size or trained.model.predict_batch_size
        scores = _score_sequence_feature_batches(
            trained.model,
            examples,
            trained.vectorizer,
            feature_batch_size=feature_batch_size,
            **feature_kwargs,
        )
        cal_scores = (
            _score_sequence_feature_batches(
                trained.model,
                calibration_examples,
                trained.vectorizer,
                feature_batch_size=feature_batch_size,
                **feature_kwargs,
            )
            if calibration_examples is not None
            else None
        )
        cal_labels = labels_for_examples(calibration_examples) if calibration_examples is not None else None
        metrics = compute_prefix_metrics(
            examples,
            scores,
            threshold_scores=cal_scores,
            threshold_labels=cal_labels,
            calibration_bins=trained.config.calibration_bins,
            threshold_selection_split="cal" if cal_scores is not None else "eval",
        )
        return scores, metrics
    if isinstance(trained.model, TorchBinaryClassifier):
        batch_iterator = (
            iter_stepview_pooled_feature_batches
            if trained.model_type == "stepview-pooled-mlp-fast-torch"
            else iter_prefix_tfidf_feature_batches
        )
        feature_batch_size = trained.config.feature_batch_size or trained.model.predict_batch_size
        scores = _score_feature_batches(
            trained.model,
            examples,
            trained.vectorizer,
            batch_iterator=batch_iterator,
            feature_batch_size=feature_batch_size,
            **feature_kwargs,
        )
        cal_scores = (
            _score_feature_batches(
                trained.model,
                calibration_examples,
                trained.vectorizer,
                batch_iterator=batch_iterator,
                feature_batch_size=feature_batch_size,
                **feature_kwargs,
            )
            if calibration_examples is not None
            else None
        )
        cal_labels = labels_for_examples(calibration_examples) if calibration_examples is not None else None
        metrics = compute_prefix_metrics(
            examples,
            scores,
            threshold_scores=cal_scores,
            threshold_labels=cal_labels,
            calibration_bins=trained.config.calibration_bins,
            threshold_selection_split="cal" if cal_scores is not None else "eval",
        )
        return scores, metrics
    if trained.model_type == "stepview-pooled-mlp":
        features = build_stepview_pooled_prefix_features(examples, trained.vectorizer, **feature_kwargs)
        cal_features = (
            build_stepview_pooled_prefix_features(calibration_examples, trained.vectorizer, **feature_kwargs)
            if calibration_examples is not None
            else None
        )
    else:
        features = transform_prefix_documents(examples, trained.vectorizer, **feature_kwargs)
        cal_features = (
            transform_prefix_documents(calibration_examples, trained.vectorizer, **feature_kwargs)
            if calibration_examples is not None
            else None
        )
    scores = _predict_scores(trained.model, features)
    cal_labels = labels_for_examples(calibration_examples) if calibration_examples is not None else None
    metrics = compute_prefix_metrics(
        examples,
        scores,
        threshold_scores=_predict_scores(trained.model, cal_features) if cal_features is not None else None,
        threshold_labels=cal_labels,
        calibration_bins=trained.config.calibration_bins,
        threshold_selection_split="cal" if cal_features is not None else "eval",
    )
    return scores, metrics


def write_baseline_artifacts(
    output_dir: str | Path,
    *,
    trained: TrainedBaseline,
    split_metrics: dict[str, dict[str, float | int | str]],
    scores_by_split: dict[str, Sequence[float]] | None = None,
    extra_metadata: dict | None = None,
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_type": trained.model_type,
        "config": asdict(trained.config),
        "selected_c": trained.selected_c,
        "validation_metrics": trained.validation_metrics,
        "vectorizer_vocabulary_size": int(len(trained.vectorizer.vocabulary_)),
        "tfidf_resolved_backend": trained.vectorizer.resolved_backend,
        "extra": extra_metadata or {},
    }
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps(split_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if scores_by_split is not None:
        serializable_scores = {
            split: [float(score) for score in scores]
            for split, scores in scores_by_split.items()
        }
        (path / "scores.json").write_text(
            json.dumps(serializable_scores, indent=2, sort_keys=True),
            encoding="utf-8",
        )

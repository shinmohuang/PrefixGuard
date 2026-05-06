from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from math import ceil
import pickle
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional GPU dependency
    cp = None

try:
    import cupyx.scipy.sparse as cpx_sparse
except ImportError:  # pragma: no cover - optional GPU dependency
    cpx_sparse = None

try:
    import cudf
except ImportError:  # pragma: no cover - optional GPU dependency
    cudf = None

try:
    from cuml.feature_extraction.text import TfidfVectorizer as CuMlTfidfVectorizer
except ImportError:  # pragma: no cover - optional GPU dependency
    CuMlTfidfVectorizer = None

from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from scipy import sparse as scipy_sparse

from monitor_symbolization.data.schema import StepView

# cuDF's sizes_to_offsets_iterator uses int32 offsets; total ngram output chars must stay
# below INT32_MAX (2_147_483_647).  With ngram_range=(1, 2), each input character produces
# roughly 3.5 output characters on average (unigrams + bigrams, including spaces/separators).
# We target 1.5 B to leave a comfortable margin; if the corpus exceeds this, fit() will
# subsample uniformly before passing to cuML.
_CUML_NGRAM_SAFE_CHARS: int = 1_500_000_000
_CUML_NGRAM_EXPANSION_FACTOR: float = 3.5


DEFAULT_TRANSFORMER_MODEL = "nomic-ai/nomic-embed-text-v1.5"
# Nomic v1.5 supports longer contexts, but 2048 is a practical default for
# fine-tuning on 40GB GPUs in this project.
DEFAULT_TRANSFORMER_MAX_LENGTH = 2048
_NOMIC_TASK_PREFIXES = (
    "search_document:",
    "search_query:",
    "clustering:",
    "classification:",
)


@dataclass
class EncoderOutput:
    embeddings: torch.Tensor


@dataclass(frozen=True)
class PreparedTfidfPayloads:
    texts: list[str]
    sidechannel_tokens: list[tuple[str, ...]]


_METADATA_SIDECHANNEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "tau2-semantic-v1": (
        "tool_role=",
        "verification_obligation=",
        "argument_risk=",
        "result_state=",
        "query_collecting_subtype=",
        "query_collecting_mode=",
        "query_collecting_has_identifier=",
    ),
    "tau2-semantic-v2": (
        "tool_role=",
        "verification_obligation=",
        "obligation_state=",
    ),
    "skillsbench-process-v1": (
        "skillsbench_phase=",
        "skillsbench_error_persistence=",
        "skillsbench_retry_pattern=",
        "skillsbench_progress_state=",
    ),
    "skillsbench-task-v1": (
        "skillsbench_task_type=",
        "skillsbench_exit_code=",
        "skillsbench_outcome=",
    ),
    "skillsbench-exec-v2": (
        "skillsbench_task_type=",
        "skillsbench_exit_code_fine=",
        "skillsbench_outcome=",
    ),
    "terminalbench-task-v1": (
        "task_name=",
    ),
    "terminalbench-meta-v1": (
        "task_name=",
        "agent=",
        "model=",
    ),
}


class BaseSegmentEncoder(nn.Module):
    output_dim: int

    def fit(self, texts: Iterable[str | StepView]) -> None:
        return None

    def supports_runtime_embedding_cache(self) -> bool:
        return False

    def runtime_cache_batch_cost_limit(self) -> int | None:
        return None

    def runtime_cache_batch_item_limit(self) -> int | None:
        return None

    def runtime_cache_batch_costs(
        self,
        payloads: list[str | StepView],
    ) -> list[int] | None:
        return None

    def export_artifact_state(self) -> dict | None:
        return None

    def load_artifact_state(self, artifact_state: dict) -> None:
        raise RuntimeError(f"{self.__class__.__name__} does not support artifact restoration")

    def encode(
        self,
        payloads: list[str | StepView],
        device: torch.device,
        progress_label: str | None = None,
    ) -> EncoderOutput:
        raise NotImplementedError


class TransformerSegmentEncoder(BaseSegmentEncoder):
    def __init__(
        self,
        model_name: str,
        fine_tune: bool = False,
        batch_size: int = 16,
        max_length: int = DEFAULT_TRANSFORMER_MAX_LENGTH,
        step_view_text_mode: str = "dense",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        if step_view_text_mode not in {
            "dense",
            "lexical",
            "transfer-full-lexical",
            "fieldwise",
            "grouped-fieldwise",
        }:
            raise ValueError(f"Unsupported step_view_text_mode: {step_view_text_mode}")
        self.step_view_text_mode = step_view_text_mode
        self._uses_nomic_recipe = self._is_nomic_embed_model(model_name)
        self._uses_qwen_recipe = self._is_qwen3_embedding_model(model_name)
        tokenizer_name = self._resolve_tokenizer_name(model_name)
        tokenizer_kwargs = self._tokenizer_kwargs_for_model(
            model_name,
            max_length=max_length,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

        model_kwargs = self._model_kwargs_for_model(model_name, fine_tune=fine_tune)
        if self._uses_nomic_recipe:
            model_kwargs["trust_remote_code"] = True
            if max_length > 2048:
                # Official model card shows `rotary_scaling_factor=2` for long-context use.
                model_kwargs["rotary_scaling_factor"] = 2
        if self._is_jina_code_model(model_name):
            model_kwargs["trust_remote_code"] = True
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.output_dim = int(self.model.config.hidden_size)
        if fine_tune:
            self._enable_gradient_checkpointing()
        else:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()
        self.fine_tune = fine_tune

    def _enable_gradient_checkpointing(self) -> None:
        # Preserve the existing training objective and data semantics while
        # reducing activation memory for long-context encoder fine-tuning.
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model, "_set_gradient_checkpointing"):
            self.model._set_gradient_checkpointing(self.model, True)

    @staticmethod
    def _is_nomic_embed_model(model_name: str) -> bool:
        return model_name.startswith("nomic-ai/nomic-embed-text-v1.5")

    @staticmethod
    def _is_qwen3_embedding_model(model_name: str) -> bool:
        return model_name.startswith("Qwen/Qwen3-Embedding-")

    @staticmethod
    def _is_jina_code_model(model_name: str) -> bool:
        return model_name.startswith("jinaai/jina-embeddings-v2-base-code")

    @staticmethod
    def _resolve_tokenizer_name(model_name: str) -> str:
        if TransformerSegmentEncoder._is_nomic_embed_model(model_name):
            return "bert-base-uncased"
        return model_name

    @staticmethod
    def _tokenizer_kwargs_for_model(model_name: str, *, max_length: int) -> dict:
        if TransformerSegmentEncoder._is_nomic_embed_model(model_name):
            return {"model_max_length": max_length}
        if TransformerSegmentEncoder._is_qwen3_embedding_model(model_name):
            return {"padding_side": "left"}
        return {}

    @staticmethod
    def _model_kwargs_for_model(model_name: str, *, fine_tune: bool) -> dict:
        if TransformerSegmentEncoder._is_qwen3_embedding_model(model_name) and not fine_tune:
            # Follow the official frozen-embedding recipe closely enough to keep
            # the 8B model operational on project GPUs without changing the
            # downstream monitoring contract.
            kwargs = {"dtype": torch.float16}
            kwargs["attn_implementation"] = (
                "flash_attention_2"
                if TransformerSegmentEncoder._flash_attention_2_available()
                else "sdpa"
            )
            return kwargs
        return {}

    @staticmethod
    def _flash_attention_2_available() -> bool:
        return importlib.util.find_spec("flash_attn") is not None

    @staticmethod
    def _prefix_texts_for_model(model_name: str, texts: list[str]) -> list[str]:
        if not TransformerSegmentEncoder._is_nomic_embed_model(model_name):
            return texts

        prefixed: list[str] = []
        for text in texts:
            if text.startswith(_NOMIC_TASK_PREFIXES):
                prefixed.append(text)
            else:
                prefixed.append(f"classification: {text}")
        return prefixed

    def _postprocess_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_uses_nomic_recipe", False):
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = F.normalize(embeddings, p=2, dim=1)
        elif getattr(self, "_uses_qwen_recipe", False):
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    @staticmethod
    def _pool_token_embeddings(
        model_name: str,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if TransformerSegmentEncoder._is_qwen3_embedding_model(model_name):
            left_padding = bool(torch.all(attention_mask[:, -1] > 0).item())
            if left_padding:
                return hidden[:, -1, :]
            sequence_lengths = attention_mask.sum(dim=1).to(torch.long).clamp_min(1) - 1
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_indices, sequence_lengths, :]

        expanded_mask = attention_mask.unsqueeze(-1)
        summed = (hidden * expanded_mask).sum(dim=1)
        lengths = expanded_mask.sum(dim=1).clamp_min(1)
        return summed / lengths

    def _encode_flat_texts(
        self,
        texts: list[str],
        device: torch.device,
        progress_label: str | None = None,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.output_dim), dtype=torch.float32, device=device)

        self.model.to(device)
        pooled_batches: list[torch.Tensor] = []
        total_batches = ceil(len(texts) / self.batch_size)
        if progress_label:
            print(
                f"[{progress_label}] encoding {len(texts)} segments on {device} "
                f"with encoder_batch_size={self.batch_size} ({total_batches} batches)",
                flush=True,
            )
        for batch_index, start in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch_texts = texts[start : start + self.batch_size]
            batch_texts = self._prefix_texts_for_model(self.model_name, batch_texts)
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            # Respect the outer autograd context so validation-time
            # re-symbolization can stay inference-only even when the encoder
            # is fine-tuned during training.
            with torch.set_grad_enabled(self.fine_tune and torch.is_grad_enabled()):
                model_output = self.model(**encoded)
                hidden = model_output.last_hidden_state
                pooled = self._pool_token_embeddings(
                    self.model_name,
                    hidden,
                    encoded["attention_mask"],
                )
                # Keep downstream projection layers on a consistent dtype even
                # when the frozen transformer emits fp16/bf16 activations.
                pooled_batches.append(self._postprocess_embeddings(pooled).to(torch.float32))
            if progress_label and (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % 100 == 0
            ):
                print(
                    f"[{progress_label}] encoded batch {batch_index}/{total_batches}",
                    flush=True,
                )
        if progress_label:
            print(f"[{progress_label}] finished encoding", flush=True)
        return torch.cat(pooled_batches, dim=0)

    def supports_runtime_embedding_cache(self) -> bool:
        return not self.fine_tune

    def encode(
        self,
        payloads: list[str | StepView],
        device: torch.device,
        progress_label: str | None = None,
    ) -> EncoderOutput:
        if not payloads:
            empty = torch.empty((0, self.output_dim), dtype=torch.float32, device=device)
            return EncoderOutput(embeddings=empty)

        if all(isinstance(payload, str) for payload in payloads):
            embeddings = self._encode_flat_texts(
                [payload for payload in payloads if isinstance(payload, str)],
                device=device,
                progress_label=progress_label,
            )
            return EncoderOutput(embeddings=embeddings)

        payload_route_groups: list[tuple[tuple[str, ...], ...]] = []
        for payload in payloads:
            if isinstance(payload, StepView):
                if self.step_view_text_mode == "lexical":
                    payload_route_groups.append(((payload.lexical_text,),))
                elif self.step_view_text_mode == "transfer-full-lexical":
                    payload_route_groups.append(((payload.render_text("transfer-full"),),))
                elif self.step_view_text_mode == "fieldwise":
                    payload_route_groups.append((payload.field_chunks(),))
                elif self.step_view_text_mode == "grouped-fieldwise":
                    payload_route_groups.append(payload.field_route_groups())
                else:
                    payload_route_groups.append((payload.dense_chunks,))
            else:
                payload_route_groups.append((((payload,),))[0:1])
        flat_chunks = [
            chunk
            for route_groups in payload_route_groups
            for route in route_groups
            for chunk in route
        ]
        chunk_embeddings = self._encode_flat_texts(
            flat_chunks,
            device=device,
            progress_label=progress_label,
        )
        aggregated: list[torch.Tensor] = []
        cursor = 0
        for route_groups in payload_route_groups:
            route_embeddings: list[torch.Tensor] = []
            for route in route_groups:
                route_chunk_embeddings = chunk_embeddings[cursor : cursor + len(route)]
                route_embeddings.append(route_chunk_embeddings.mean(dim=0))
                cursor += len(route)
            aggregated.append(torch.stack(route_embeddings, dim=0).mean(dim=0))
        return EncoderOutput(embeddings=torch.stack(aggregated, dim=0))


class TfidfSegmentEncoder(BaseSegmentEncoder):
    def __init__(
        self,
        max_features: int = 4096,
        step_view_text_mode: str = "full",
        backend: str = "auto",
        sparse_output: bool = False,
        metadata_sidechannel_mode: str | None = None,
        metadata_sidechannel_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if backend not in {"auto", "cuml", "sklearn"}:
            raise ValueError(f"Unsupported TF-IDF backend: {backend}")
        if metadata_sidechannel_mode not in {None, *sorted(_METADATA_SIDECHANNEL_PREFIXES.keys())}:
            raise ValueError(
                f"Unsupported TF-IDF metadata sidechannel mode: {metadata_sidechannel_mode}"
            )
        self.backend = backend
        self.resolved_backend = self._resolve_backend(backend)
        vectorizer_cls = (
            CuMlTfidfVectorizer
            if self.resolved_backend == "cuml"
            else SklearnTfidfVectorizer
        )
        self.vectorizer = vectorizer_cls(max_features=max_features, ngram_range=(1, 2))
        self.output_dim = max_features
        self._is_fitted = False
        self.step_view_text_mode = step_view_text_mode
        self.sparse_output = sparse_output
        self.metadata_sidechannel_mode = metadata_sidechannel_mode
        self.metadata_sidechannel_scale = float(metadata_sidechannel_scale)
        self.metadata_sidechannel_prefixes = (
            tuple()
            if metadata_sidechannel_mode is None
            else _METADATA_SIDECHANNEL_PREFIXES[metadata_sidechannel_mode]
        )
        self.metadata_sidechannel_vocabulary: tuple[str, ...] = tuple()

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend == "sklearn":
            return "sklearn"
        if backend == "cuml":
            if CuMlTfidfVectorizer is None or cudf is None:
                raise RuntimeError(
                    "TF-IDF backend 'cuml' requested but cuml/cudf is not installed"
                )
            return "cuml"
        if CuMlTfidfVectorizer is not None and cudf is not None:
            return "cuml"
        return "sklearn"

    @staticmethod
    def _prepare_payloads(
        texts: Iterable[str | StepView],
        *,
        step_view_text_mode: str,
        metadata_sidechannel_mode: str | None = None,
    ) -> PreparedTfidfPayloads:
        sidechannel_prefixes = (
            tuple()
            if metadata_sidechannel_mode is None
            else _METADATA_SIDECHANNEL_PREFIXES[metadata_sidechannel_mode]
        )
        prepared_texts: list[str] = []
        prepared_sidechannels: list[tuple[str, ...]] = []
        for text in texts:
            if not isinstance(text, StepView):
                prepared_texts.append(text)
                prepared_sidechannels.append(tuple())
                continue
            if not sidechannel_prefixes:
                prepared_texts.append(text.render_text(step_view_text_mode))
                prepared_sidechannels.append(tuple())
                continue
            prepared_texts.append(
                text.render_text(
                    step_view_text_mode,
                    exclude_metadata_prefixes=sidechannel_prefixes,
                )
            )
            prepared_sidechannels.append(
                tuple(
                    line
                    for line in text.metadata_lines
                    if any(line.startswith(prefix) for prefix in sidechannel_prefixes)
                )
            )
        return PreparedTfidfPayloads(
            texts=prepared_texts,
            sidechannel_tokens=prepared_sidechannels,
        )

    @staticmethod
    def _normalize_payloads(
        texts: Iterable[str | StepView],
        *,
        step_view_text_mode: str,
        metadata_sidechannel_mode: str | None = None,
    ) -> list[str]:
        return TfidfSegmentEncoder._prepare_payloads(
            texts,
            step_view_text_mode=step_view_text_mode,
            metadata_sidechannel_mode=metadata_sidechannel_mode,
        ).texts

    def _prepare_text_input(self, texts: list[str]):
        if self.resolved_backend == "cuml":
            assert cudf is not None
            return cudf.Series(texts)
        return texts

    @staticmethod
    def _build_sidechannel_vocabulary(
        sidechannel_tokens: list[tuple[str, ...]],
    ) -> tuple[str, ...]:
        values = {
            token
            for tokens in sidechannel_tokens
            for token in tokens
            if token
        }
        return tuple(sorted(values))

    @staticmethod
    def _encode_sidechannel_dense(
        sidechannel_tokens: list[tuple[str, ...]],
        vocabulary: tuple[str, ...],
        *,
        device: torch.device,
        scale: float = 1.0,
    ) -> torch.Tensor:
        if not sidechannel_tokens:
            return torch.empty((0, len(vocabulary)), dtype=torch.float32, device=device)
        matrix = torch.zeros(
            (len(sidechannel_tokens), len(vocabulary)),
            dtype=torch.float32,
            device=device,
        )
        if not vocabulary:
            return matrix
        feature_index = {token: index for index, token in enumerate(vocabulary)}
        for row_index, tokens in enumerate(sidechannel_tokens):
            for token in tokens:
                column_index = feature_index.get(token)
                if column_index is not None:
                    matrix[row_index, column_index] = scale
        return matrix

    @staticmethod
    def _encode_sidechannel_sparse(
        sidechannel_tokens: list[tuple[str, ...]],
        vocabulary: tuple[str, ...],
        *,
        device: torch.device,
        scale: float = 1.0,
    ) -> torch.Tensor:
        if not sidechannel_tokens:
            with torch.sparse.check_sparse_tensor_invariants(False):
                return torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.int64, device=device),
                    torch.empty((0,), dtype=torch.float32, device=device),
                    size=(0, len(vocabulary)),
                    device=device,
                ).coalesce()
        if not vocabulary:
            with torch.sparse.check_sparse_tensor_invariants(False):
                return torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.int64, device=device),
                    torch.empty((0,), dtype=torch.float32, device=device),
                    size=(len(sidechannel_tokens), 0),
                    device=device,
                ).coalesce()
        feature_index = {token: index for index, token in enumerate(vocabulary)}
        rows: list[int] = []
        cols: list[int] = []
        for row_index, tokens in enumerate(sidechannel_tokens):
            for token in tokens:
                column_index = feature_index.get(token)
                if column_index is not None:
                    rows.append(row_index)
                    cols.append(column_index)
        if not rows:
            with torch.sparse.check_sparse_tensor_invariants(False):
                return torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.int64, device=device),
                    torch.empty((0,), dtype=torch.float32, device=device),
                    size=(len(sidechannel_tokens), len(vocabulary)),
                    device=device,
                ).coalesce()
        indices = torch.tensor([rows, cols], dtype=torch.int64, device=device)
        values = torch.full((len(rows),), float(scale), dtype=torch.float32, device=device)
        with torch.sparse.check_sparse_tensor_invariants(False):
            return torch.sparse_coo_tensor(
                indices,
                values,
                size=(len(sidechannel_tokens), len(vocabulary)),
                device=device,
            ).coalesce()

    @staticmethod
    def _hstack_sparse(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        left = left.coalesce()
        right = right.coalesce()
        if left.size(0) != right.size(0):
            raise ValueError("Sparse sidechannel features must align on the row dimension")
        left_indices = left.indices()
        right_indices = right.indices().clone()
        right_indices[1] += left.size(1)
        indices = torch.cat([left_indices, right_indices], dim=1)
        values = torch.cat([left.values(), right.values()], dim=0)
        with torch.sparse.check_sparse_tensor_invariants(False):
            return torch.sparse_coo_tensor(
                indices,
                values,
                size=(left.size(0), left.size(1) + right.size(1)),
                device=left.device,
            ).coalesce()

    @staticmethod
    def _infer_backend_from_vectorizer(vectorizer) -> str:
        module_name = type(vectorizer).__module__
        return "cuml" if module_name.startswith("cuml.") else "sklearn"

    @staticmethod
    def _feature_count(vectorizer, features) -> int:
        if hasattr(vectorizer, "get_feature_names_out"):
            try:
                return int(len(vectorizer.get_feature_names_out()))
            except TypeError:
                pass
        return int(features.shape[1])

    @staticmethod
    def _to_torch_dense(features, device: torch.device) -> torch.Tensor:
        dense = features.toarray() if hasattr(features, "toarray") else features.todense()
        if cp is not None and isinstance(dense, cp.ndarray):
            tensor = torch.from_dlpack(dense.astype(cp.float32, copy=False))
            return tensor if tensor.device == device else tensor.to(device)
        if hasattr(dense, "get"):
            dense = dense.get()
        dense_np = np.asarray(dense, dtype=np.float32)
        return torch.from_numpy(dense_np).to(device)

    @staticmethod
    def _to_torch_sparse(features, device: torch.device) -> torch.Tensor:
        coo = features.tocoo() if hasattr(features, "tocoo") else features
        if cpx_sparse is not None and isinstance(coo, cpx_sparse.coo_matrix):
            row = torch.utils.dlpack.from_dlpack(coo.row.astype(cp.int64, copy=False))
            col = torch.utils.dlpack.from_dlpack(coo.col.astype(cp.int64, copy=False))
            values = torch.utils.dlpack.from_dlpack(coo.data.astype(cp.float32, copy=False))
            indices = torch.stack([row, col], dim=0)
            with torch.sparse.check_sparse_tensor_invariants(False):
                sparse = torch.sparse_coo_tensor(
                    indices,
                    values,
                    size=coo.shape,
                    device=values.device,
                )
            return sparse.coalesce() if sparse.device == device else sparse.coalesce().to(device)
        if scipy_sparse.issparse(coo):
            row = torch.from_numpy(np.asarray(coo.row, dtype=np.int64))
            col = torch.from_numpy(np.asarray(coo.col, dtype=np.int64))
            values = torch.from_numpy(np.asarray(coo.data, dtype=np.float32))
            indices = torch.stack([row, col], dim=0)
            with torch.sparse.check_sparse_tensor_invariants(False):
                return torch.sparse_coo_tensor(
                    indices,
                    values,
                    size=coo.shape,
                    device=device,
                ).coalesce()
        raise TypeError(f"Unsupported sparse feature type: {type(features)!r}")

    @staticmethod
    def _safe_subsample_for_cuml(texts: list[str], *, seed: int = 42) -> list[str]:
        """Return a subsample of *texts* whose total estimated cuDF ngram output stays under
        ``_CUML_NGRAM_SAFE_CHARS``.

        cuDF's ``sizes_to_offsets_iterator`` uses int32 offsets, so the total number of output
        characters produced by ``ngrams_tokenize`` must not exceed INT32_MAX.  With
        ``ngram_range=(1, 2)`` the output is roughly ``_CUML_NGRAM_EXPANSION_FACTOR``× the input
        character count.  When the estimated output would overflow we draw a uniform random
        subsample (deterministic via *seed*) that keeps the total comfortably below the limit.

        The subsample is used **only** for vocabulary fitting; the full corpus is still used
        for the sidechannel vocabulary and for transform().  Because max_features caps the
        vocabulary at a small number (default 4096), the top terms are stable across
        representative subsamples of large corpora.
        """
        total_chars = sum(len(t) for t in texts)
        if total_chars * _CUML_NGRAM_EXPANSION_FACTOR <= _CUML_NGRAM_SAFE_CHARS:
            return texts  # no overflow risk
        safe_chars = _CUML_NGRAM_SAFE_CHARS / _CUML_NGRAM_EXPANSION_FACTOR
        keep_fraction = safe_chars / total_chars
        n_keep = max(1, int(len(texts) * keep_fraction))
        rng = np.random.default_rng(seed=seed)
        indices = np.sort(rng.choice(len(texts), size=n_keep, replace=False))
        return [texts[i] for i in indices]

    def fit(self, texts: Iterable[str | StepView]) -> None:
        prepared = self._prepare_payloads(
            texts,
            step_view_text_mode=self.step_view_text_mode,
            metadata_sidechannel_mode=self.metadata_sidechannel_mode,
        )
        # For cuML backend: guard against INT32 overflow in cuDF ngrams_tokenize by
        # subsampling the fit corpus when the estimated output size would exceed the limit.
        # The full corpus is still used for the sidechannel vocabulary (no subsampling needed
        # there, as it is a simple set union, not a cuDF operation).
        texts_for_fit = (
            self._safe_subsample_for_cuml(prepared.texts)
            if self.resolved_backend == "cuml"
            else prepared.texts
        )
        vectorizer_input = self._prepare_text_input(texts_for_fit)
        self.vectorizer.fit(vectorizer_input)
        # Use a single doc for the dimension probe (no overflow risk).
        probe_input = self._prepare_text_input(texts_for_fit[:1] if texts_for_fit else texts_for_fit)
        transformed = self.vectorizer.transform(probe_input)
        text_output_dim = self._feature_count(self.vectorizer, transformed)
        self.metadata_sidechannel_vocabulary = self._build_sidechannel_vocabulary(
            prepared.sidechannel_tokens
        )
        self.output_dim = text_output_dim + len(self.metadata_sidechannel_vocabulary)
        self._is_fitted = True

    def supports_runtime_embedding_cache(self) -> bool:
        return True

    def runtime_cache_batch_cost_limit(self) -> int | None:
        if self.resolved_backend != "cuml":
            return None
        return int(_CUML_NGRAM_SAFE_CHARS / _CUML_NGRAM_EXPANSION_FACTOR)

    def runtime_cache_batch_item_limit(self) -> int | None:
        if self.resolved_backend != "cuml":
            return None
        # cuML TF-IDF can still hit large temporary allocations in transform() even when
        # the total text-character budget is safe, so cap the number of payloads per
        # runtime-cache batch as a second guardrail.
        return 50_000

    def runtime_cache_batch_costs(
        self,
        payloads: list[str | StepView],
    ) -> list[int] | None:
        if self.resolved_backend != "cuml":
            return None
        prepared = self._prepare_payloads(
            payloads,
            step_view_text_mode=self.step_view_text_mode,
            metadata_sidechannel_mode=self.metadata_sidechannel_mode,
        )
        return [max(len(text), 1) for text in prepared.texts]

    def export_artifact_state(self) -> dict:
        if not self._is_fitted:
            raise RuntimeError("TfidfSegmentEncoder must be fitted before exporting artifacts")
        return {
            "kind": "tfidf_vectorizer_pickle",
            "step_view_text_mode": self.step_view_text_mode,
            "backend": self.backend,
            "resolved_backend": self.resolved_backend,
            "sparse_output": self.sparse_output,
            "metadata_sidechannel_mode": self.metadata_sidechannel_mode,
            "metadata_sidechannel_scale": self.metadata_sidechannel_scale,
            "metadata_sidechannel_vocabulary": self.metadata_sidechannel_vocabulary,
            "vectorizer_pickle": pickle.dumps(self.vectorizer),
        }

    def load_artifact_state(self, artifact_state: dict) -> None:
        if artifact_state.get("kind") != "tfidf_vectorizer_pickle":
            raise ValueError("Unsupported TF-IDF artifact state")
        vectorizer = pickle.loads(artifact_state["vectorizer_pickle"])
        self.vectorizer = vectorizer
        self.step_view_text_mode = artifact_state.get("step_view_text_mode", self.step_view_text_mode)
        self.backend = artifact_state.get("backend", self.backend)
        self.resolved_backend = artifact_state.get(
            "resolved_backend",
            self._infer_backend_from_vectorizer(vectorizer),
        )
        self.sparse_output = artifact_state.get("sparse_output", self.sparse_output)
        text_output_dim = self._feature_count(
            vectorizer,
            vectorizer.transform(self._prepare_text_input(["warmup"])),
        )
        self.metadata_sidechannel_mode = artifact_state.get(
            "metadata_sidechannel_mode",
            self.metadata_sidechannel_mode,
        )
        self.metadata_sidechannel_scale = float(
            artifact_state.get("metadata_sidechannel_scale", self.metadata_sidechannel_scale)
        )
        self.metadata_sidechannel_prefixes = (
            tuple()
            if self.metadata_sidechannel_mode is None
            else _METADATA_SIDECHANNEL_PREFIXES[self.metadata_sidechannel_mode]
        )
        self.metadata_sidechannel_vocabulary = tuple(
            artifact_state.get("metadata_sidechannel_vocabulary", ())
        )
        self.output_dim = text_output_dim + len(self.metadata_sidechannel_vocabulary)
        self._is_fitted = True

    def encode(
        self,
        payloads: list[str | StepView],
        device: torch.device,
        progress_label: str | None = None,
    ) -> EncoderOutput:
        if not self._is_fitted:
            raise RuntimeError("TfidfSegmentEncoder must be fitted before encoding")
        prepared = self._prepare_payloads(
            payloads,
            step_view_text_mode=self.step_view_text_mode,
            metadata_sidechannel_mode=self.metadata_sidechannel_mode,
        )
        if progress_label:
            print(
                f"[{progress_label}] encoding {len(prepared.texts)} segments with TF-IDF "
                f"({self.resolved_backend}, {'sparse' if self.sparse_output else 'dense'}) on {device}",
                flush=True,
            )
        features = self.vectorizer.transform(self._prepare_text_input(prepared.texts))
        text_embeddings = (
            self._to_torch_sparse(features, device)
            if self.sparse_output
            else self._to_torch_dense(features, device)
        )
        if self.sparse_output:
            sidechannel_embeddings = self._encode_sidechannel_sparse(
                prepared.sidechannel_tokens,
                self.metadata_sidechannel_vocabulary,
                device=device,
                scale=self.metadata_sidechannel_scale,
            )
            embeddings = self._hstack_sparse(text_embeddings, sidechannel_embeddings)
        else:
            sidechannel_embeddings = self._encode_sidechannel_dense(
                prepared.sidechannel_tokens,
                self.metadata_sidechannel_vocabulary,
                device=device,
                scale=self.metadata_sidechannel_scale,
            )
            embeddings = torch.cat([text_embeddings, sidechannel_embeddings], dim=1)
        if progress_label:
            print(f"[{progress_label}] finished encoding", flush=True)
        return EncoderOutput(embeddings=embeddings)


class HybridStepEncoder(BaseSegmentEncoder):
    def __init__(
        self,
        model_name: str,
        lexical_max_features: int = 4096,
        step_view_text_mode: str = "full",
        lexical_metadata_sidechannel_mode: str | None = None,
        lexical_metadata_sidechannel_scale: float = 1.0,
        transformer_step_view_mode: str = "dense",
        fine_tune: bool = False,
        batch_size: int = 16,
        max_length: int = DEFAULT_TRANSFORMER_MAX_LENGTH,
    ) -> None:
        super().__init__()
        self.lexical_encoder = TfidfSegmentEncoder(
            max_features=lexical_max_features,
            step_view_text_mode=step_view_text_mode,
            metadata_sidechannel_mode=lexical_metadata_sidechannel_mode,
            metadata_sidechannel_scale=lexical_metadata_sidechannel_scale,
        )
        self.observation_encoder = TransformerSegmentEncoder(
            model_name=model_name,
            fine_tune=fine_tune,
            batch_size=batch_size,
            max_length=max_length,
            step_view_text_mode=transformer_step_view_mode,
        )
        self._projection: nn.Linear | None = None
        self.output_dim = 0
        self.fine_tune = fine_tune

    def fit(self, texts: Iterable[str | StepView]) -> None:
        step_views = [text for text in texts if isinstance(text, StepView)]
        self.lexical_encoder.fit(step_views)
        combined_dim = self.lexical_encoder.output_dim + self.observation_encoder.output_dim
        self._projection = nn.Linear(combined_dim, self.observation_encoder.output_dim)
        self.output_dim = self.observation_encoder.output_dim

    def encode(
        self,
        payloads: list[str | StepView],
        device: torch.device,
        progress_label: str | None = None,
    ) -> EncoderOutput:
        if not payloads:
            empty = torch.empty((0, self.output_dim), dtype=torch.float32, device=device)
            return EncoderOutput(embeddings=empty)
        if any(not isinstance(payload, StepView) for payload in payloads):
            raise TypeError("HybridStepEncoder expects StepView payloads")
        if self._projection is None:
            raise RuntimeError("HybridStepEncoder must be fitted before encoding")

        lexical_output = self.lexical_encoder.encode(
            payloads,
            device=device,
            progress_label=f"{progress_label}/lexical" if progress_label else None,
        ).embeddings
        observation_output = self.observation_encoder.encode(
            payloads,
            device=device,
            progress_label=f"{progress_label}/observation" if progress_label else None,
        ).embeddings
        self._projection.to(device)
        with torch.set_grad_enabled(self.fine_tune and torch.is_grad_enabled()):
            fused = torch.cat([lexical_output, observation_output], dim=1)
            embeddings = self._projection(fused)
        return EncoderOutput(embeddings=embeddings)

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError


class OpenAIAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAIChatConfig:
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    timeout_seconds: float = 120.0


def config_from_env(
    *,
    model: str | None,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    temperature: float = 0.0,
    timeout_seconds: float = 120.0,
) -> OpenAIChatConfig:
    resolved_model = model or os.environ.get("OPENAI_MODEL", "").strip()
    if not resolved_model:
        raise OpenAIAdapterError("No OpenAI model configured. Pass --openai-model or set OPENAI_MODEL.")
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise OpenAIAdapterError(f"Missing API key in environment variable {api_key_env}.")
    base_url = os.environ.get(base_url_env, "").strip() or "https://api.openai.com/v1"
    return OpenAIChatConfig(
        model=resolved_model,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )


def request_json_chat_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    config: OpenAIChatConfig,
    json_schema: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": json_schema,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = request.Request(
        f"{config.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as response:
            raw_response = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise OpenAIAdapterError(
            f"OpenAI-compatible request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise OpenAIAdapterError(f"OpenAI-compatible request failed: {exc}") from exc

    response_obj = json.loads(raw_response)
    refusal = response_obj.get("choices", [{}])[0].get("message", {}).get("refusal")
    if refusal:
        raise OpenAIAdapterError(f"Model refused structured output request: {refusal}")
    try:
        content = response_obj["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenAIAdapterError(
            "Malformed chat completion response; missing choices[0].message.content"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise OpenAIAdapterError("Chat completion returned empty content.")
    parsed_obj = json.loads(content)
    return response_obj, json.dumps(parsed_obj, indent=2, sort_keys=True, ensure_ascii=True)

"""Provider and capability resolution for Montferrand LLM backends.

This module keeps provider transport, model capabilities, and structured-output
strategy separate so the rest of the app can swap between direct Inception and
Claude via OpenRouter without changing conversation logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, cast

from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

BackendProvider = Literal["openrouter", "inception"]
BackendRole = Literal["agent", "judge"]
StructuredOutputStrategy = Literal["native", "tool"]

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_INCEPTION_BASE_URL = "https://api.inceptionlabs.ai/v1"

_OPENROUTER_NATIVE_MODEL_PREFIXES = (
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.1",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-opus-4.6",
    "inception/",
)


@dataclass(frozen=True)
class BackendSpec:
    """Fully-resolved backend connection settings."""

    provider: BackendProvider
    model_name: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class BackendCapabilities:
    """Backend capabilities relevant to agent construction."""

    supports_native_structured_output: bool
    supports_required_tool_choice: bool
    structured_output_strategy: StructuredOutputStrategy


@dataclass(frozen=True)
class ResolvedBackend:
    """A resolved backend, its base profile, and the chosen strategy."""

    spec: BackendSpec
    base_profile: OpenAIModelProfile
    capabilities: BackendCapabilities


def _resolve_env(*names: str, default: str) -> str:
    """Return the first non-empty env var found, or *default*."""

    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _require_env(env_var: str, message: str) -> str:
    """Return a non-empty env var or raise RuntimeError."""

    value = os.getenv(env_var, "").strip()
    if not value:
        raise RuntimeError(message)
    return value


def _resolve_model_name(*env_names: str, model_name: str | None = None) -> str:
    """Return the requested model name or the first configured env var."""

    if model_name:
        return model_name
    for name in env_names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise RuntimeError(f"No model configured. Set one of: {', '.join(env_names)}")


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _resolve_provider(role: BackendRole) -> BackendProvider:
    env_names = (
        ("MONTFERRAND_PROVIDER",)
        if role == "agent"
        else ("MONTFERRAND_JUDGE_PROVIDER", "MONTFERRAND_PROVIDER")
    )
    provider = _resolve_env(*env_names, default="openrouter").lower()
    if provider not in {"openrouter", "inception"}:
        raise RuntimeError(
            "Unsupported provider. "
            "Set MONTFERRAND_PROVIDER/MONTFERRAND_JUDGE_PROVIDER to "
            "'openrouter' or 'inception'."
        )
    if provider == "openrouter":
        return cast(BackendProvider, provider)
    return cast(BackendProvider, provider)


def _coerce_provider(provider: str) -> BackendProvider:
    """Validate and normalize a provider override string."""

    normalized = provider.strip().lower()
    if normalized not in {"openrouter", "inception"}:
        raise RuntimeError(
            "Unsupported provider override. Use 'openrouter' or 'inception'."
        )
    return cast(BackendProvider, normalized)


def _resolve_spec(
    role: BackendRole,
    model_name: str | None = None,
    provider: BackendProvider | None = None,
) -> BackendSpec:
    provider = provider or _resolve_provider(role)
    model_env_names = (
        ("MONTFERRAND_MODEL",)
        if role == "agent"
        else ("MONTFERRAND_JUDGE_MODEL", "MONTFERRAND_MODEL")
    )
    resolved_model_name = _resolve_model_name(*model_env_names, model_name=model_name)

    if provider == "openrouter":
        base_url = _normalize_base_url(
            _resolve_env("OPENROUTER_BASE_URL", default=DEFAULT_OPENROUTER_BASE_URL)
        )
        api_key = _require_env(
            "OPENROUTER_API_KEY",
            "OPENROUTER_API_KEY is not set. Copy .env.template to .env and fill "
            "in your key.",
        )
    else:
        base_url = _normalize_base_url(
            _resolve_env("INCEPTION_BASE_URL", default=DEFAULT_INCEPTION_BASE_URL)
        )
        api_key = _require_env(
            "INCEPTION_API_KEY",
            "INCEPTION_API_KEY is not set. Copy .env.template to .env and fill "
            "in your key.",
        )
        if "/" in resolved_model_name and not resolved_model_name.startswith(
            "inception/"
        ):
            raise RuntimeError(
                "Direct Inception expects an Inception model name like 'mercury-2' "
                "or 'inception/mercury-2'. Use MONTFERRAND_PROVIDER=openrouter for "
                "models such as anthropic/claude-sonnet-4.6."
            )
        if resolved_model_name.startswith("inception/"):
            resolved_model_name = resolved_model_name.split("/", 1)[1]

    return BackendSpec(
        provider=provider,
        model_name=resolved_model_name,
        base_url=base_url,
        api_key=api_key,
    )


def _supports_openrouter_native_profile(spec: BackendSpec) -> bool:
    return (
        spec.base_url == DEFAULT_OPENROUTER_BASE_URL and spec.provider == "openrouter"
    )


def _resolve_base_profile(spec: BackendSpec) -> OpenAIModelProfile:
    if spec.provider == "openrouter" and _supports_openrouter_native_profile(spec):
        profile = OpenRouterProvider.model_profile(spec.model_name)
    else:
        profile = OpenAIProvider.model_profile(spec.model_name)

    resolved = OpenAIModelProfile.from_profile(profile)

    if spec.provider == "inception" or spec.model_name.startswith("inception/"):
        resolved = resolved.update(
            OpenAIModelProfile(
                supports_json_schema_output=True,
                supports_json_object_output=True,
                default_structured_output_mode="native",
                openai_supports_tool_choice_required=False,
            )
        )
    elif spec.provider == "openrouter" and spec.model_name.startswith(
        _OPENROUTER_NATIVE_MODEL_PREFIXES
    ):
        resolved = resolved.update(
            OpenAIModelProfile(
                supports_json_schema_output=True,
                default_structured_output_mode="native",
            )
        )

    return resolved


def _resolve_capabilities(profile: OpenAIModelProfile) -> BackendCapabilities:
    structured_output_strategy: StructuredOutputStrategy = (
        "native" if profile.supports_json_schema_output else "tool"
    )
    return BackendCapabilities(
        supports_native_structured_output=profile.supports_json_schema_output,
        supports_required_tool_choice=profile.openai_supports_tool_choice_required,
        structured_output_strategy=structured_output_strategy,
    )


def resolve_backend(
    role: BackendRole = "agent",
    *,
    model_name: str | None = None,
    provider: BackendProvider | None = None,
) -> ResolvedBackend:
    """Resolve the configured backend for the given role."""

    spec = _resolve_spec(role, model_name=model_name, provider=provider)
    base_profile = _resolve_base_profile(spec)
    capabilities = _resolve_capabilities(base_profile)
    return ResolvedBackend(
        spec=spec,
        base_profile=base_profile,
        capabilities=capabilities,
    )


def build_provider(spec: BackendSpec) -> OpenRouterProvider | OpenAIProvider:
    """Build the provider for a resolved backend spec."""

    if spec.provider == "openrouter" and _supports_openrouter_native_profile(spec):
        return OpenRouterProvider(api_key=spec.api_key)
    return OpenAIProvider(base_url=spec.base_url, api_key=spec.api_key)


def build_model_profile(backend: ResolvedBackend) -> OpenAIModelProfile:
    """Build the runtime profile used when creating the model instance."""

    profile = backend.base_profile
    strategy = backend.capabilities.structured_output_strategy

    if strategy == "native":
        return OpenAIModelProfile(
            default_structured_output_mode="native",
            openai_supports_tool_choice_required=False,
        ).update(profile)

    if not backend.capabilities.supports_required_tool_choice:
        raise RuntimeError(
            "This model/backend combination does not support native structured "
            "output or required tool-choice output tools. Choose a different "
            "provider/model combination."
        )

    return OpenAIModelProfile(default_structured_output_mode="tool").update(profile)

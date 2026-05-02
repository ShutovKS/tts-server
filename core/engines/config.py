# FILE: core/engines/config.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define typed and discriminated engine configuration models for future engine registry filtering and compatibility wiring.
#   SCOPE: engine config base models, explicit disabled config, enabled engine variants, collection validation, and parsing helpers
#   DEPENDS: pydantic
#   LINKS: M-ENGINE-CONFIG, M-ENGINE-CONTRACTS
#   ROLE: CONFIG
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   DisabledEngineConfig - Explicit disabled engine config shape that still parses deterministically.
#   TorchEngineConfig - Typed enabled config for torch-routed engines.
#   MlxEngineConfig - Typed enabled config for mlx-routed engines.
#   OnnxEngineConfig - Typed enabled config for onnx-routed engines.
#   QwenFastEngineConfig - Typed enabled config for qwen_fast-routed engines.
#   EngineConfig - Discriminated union over all supported engine config shapes.
#   EngineSettings - Validated collection of engine configs with duplicate name/alias protection.
#   parse_engine_config - Parse one engine config payload into a typed model.
#   parse_engine_settings - Parse a collection payload into EngineSettings.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 2 engine wave: introduced discriminated engine config models with a deterministic disabled case, typed shared fields, and params escape hatches]
# END_CHANGE_SUMMARY

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator  # pyright: ignore[reportMissingImports]


def _normalize_identifier_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, Iterable):
        values = [str(item).strip() for item in raw]
    else:
        raise TypeError("Expected a string or iterable of strings")

    normalized: list[str] = []
    for value in values:
        if value and value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _normalize_required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


class _EngineModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _EnabledEngineConfig(_EngineModel):
    name: str
    aliases: tuple[str, ...] = ()
    family: str
    capabilities: tuple[str, ...]
    priority: int = 100
    enabled: Literal[True] = True
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "family", mode="before")
    @classmethod
    def _validate_required_text(cls, value: Any, info: Any) -> Any:
        return _normalize_required_text(value, field_name=info.field_name)

    @field_validator("aliases", "capabilities", mode="before")
    @classmethod
    def _parse_identifier_tuple(cls, value: Any) -> Any:
        return _normalize_identifier_tuple(value)

    @field_validator("params", mode="before")
    @classmethod
    def _validate_params(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("params must be a mapping")
        return dict(value)

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, value: int) -> int:
        if value < 0:
            raise ValueError("priority must be >= 0")
        return value

    @model_validator(mode="after")
    def _validate_capabilities_present(self) -> _EnabledEngineConfig:
        if not self.capabilities:
            raise ValueError("capabilities must contain at least one capability")
        return self


# START_CONTRACT: DisabledEngineConfig
#   PURPOSE: Represent a deterministic disabled-engine configuration entry that future registry filtering can skip without ambiguity.
#   INPUTS: { kind: Literal["disabled"] - Discriminator value, name: str - Stable engine config identifier, aliases: tuple[str, ...] - Optional alternate identifiers, enabled: Literal[False] - Explicit disabled flag, reason: str | None - Optional operator-facing disable reason, params: dict[str, Any] - Escape hatch for future disabled-state metadata }
#   OUTPUTS: { instance - Typed disabled engine config model }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-CONFIG
# END_CONTRACT: DisabledEngineConfig
class DisabledEngineConfig(_EngineModel):
    kind: Literal["disabled"]
    name: str
    aliases: tuple[str, ...] = ()
    enabled: Literal[False] = False
    reason: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: Any) -> Any:
        return _normalize_required_text(value, field_name="name")

    @field_validator("aliases", mode="before")
    @classmethod
    def _parse_aliases(cls, value: Any) -> Any:
        return _normalize_identifier_tuple(value)

    @field_validator("reason", mode="before")
    @classmethod
    def _normalize_reason(cls, value: Any) -> Any:
        if value is None:
            return None
        normalized = _normalize_required_text(value, field_name="reason")
        return normalized

    @field_validator("params", mode="before")
    @classmethod
    def _validate_params(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("params must be a mapping")
        return dict(value)


class TorchEngineConfig(_EnabledEngineConfig):
    kind: Literal["torch"]
    backend: Literal["torch"] = "torch"


class MlxEngineConfig(_EnabledEngineConfig):
    kind: Literal["mlx"]
    backend: Literal["mlx"] = "mlx"


class OnnxEngineConfig(_EnabledEngineConfig):
    kind: Literal["onnx"]
    backend: Literal["onnx"] = "onnx"


class QwenFastEngineConfig(_EnabledEngineConfig):
    kind: Literal["qwen_fast"]
    backend: Literal["qwen_fast"] = "qwen_fast"


EngineConfig: TypeAlias = Annotated[
    DisabledEngineConfig
    | TorchEngineConfig
    | MlxEngineConfig
    | OnnxEngineConfig
    | QwenFastEngineConfig,
    Field(discriminator="kind"),
]

_ENGINE_CONFIG_ADAPTER = TypeAdapter(EngineConfig)


# START_CONTRACT: EngineSettings
#   PURPOSE: Hold a validated collection of engine configs and reject duplicate names or aliases before future registry wiring begins.
#   INPUTS: { engines: tuple[EngineConfig, ...] - Parsed engine config entries }
#   OUTPUTS: { instance - Typed engine config collection with helper views }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-CONFIG
# END_CONTRACT: EngineSettings
class EngineSettings(_EngineModel):
    engines: tuple[EngineConfig, ...] = ()

    @field_validator("engines", mode="before")
    @classmethod
    def _parse_engines(cls, value: Any) -> Any:
        if value is None:
            return ()
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError("engines must be an iterable of engine config payloads")
        parsed: list[EngineConfig] = []
        for item in value:
            parsed.append(parse_engine_config(item))
        return tuple(parsed)

    @model_validator(mode="after")
    def _validate_unique_tokens(self) -> EngineSettings:
        seen: dict[str, str] = {}
        for config in self.engines:
            tokens = (config.name, *config.aliases)
            for token in tokens:
                normalized = token.casefold()
                previous = seen.get(normalized)
                if previous is not None:
                    raise ValueError(
                        f"Duplicate engine alias/name '{token}' conflicts with '{previous}'"
                    )
                seen[normalized] = config.name
        return self

    @property
    def enabled_engines(self) -> tuple[EngineConfig, ...]:
        return tuple(config for config in self.engines if getattr(config, "enabled", False))

    @property
    def disabled_engines(self) -> tuple[DisabledEngineConfig, ...]:
        return tuple(
            config for config in self.engines if isinstance(config, DisabledEngineConfig)
        )


# START_CONTRACT: parse_engine_config
#   PURPOSE: Parse one engine config payload into the discriminated typed engine-config union.
#   INPUTS: { payload: Mapping[str, Any] - Raw config payload }
#   OUTPUTS: { EngineConfig - Parsed typed engine config }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-CONFIG
# END_CONTRACT: parse_engine_config
def parse_engine_config(payload: Mapping[str, Any]) -> EngineConfig:
    return _ENGINE_CONFIG_ADAPTER.validate_python(payload)


# START_CONTRACT: parse_engine_settings
#   PURPOSE: Parse a collection payload into EngineSettings.
#   INPUTS: { payload: Mapping[str, Any] - Raw engine settings payload }
#   OUTPUTS: { EngineSettings - Parsed engine settings collection }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-CONFIG
# END_CONTRACT: parse_engine_settings
def parse_engine_settings(payload: Mapping[str, Any]) -> EngineSettings:
    return EngineSettings.model_validate(payload)


__all__ = [
    "DisabledEngineConfig",
    "EngineConfig",
    "EngineSettings",
    "MlxEngineConfig",
    "OnnxEngineConfig",
    "QwenFastEngineConfig",
    "TorchEngineConfig",
    "parse_engine_config",
    "parse_engine_settings",
]

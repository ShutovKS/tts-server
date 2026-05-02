# FILE: core/engines/registry.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Register TTSEngine implementations, optionally load them from entry points, and resolve deterministic engine selections by key or capability/language match.
#   SCOPE: EngineRegistry, loader/discovery helpers, duplicate-key protection, optional entry-point loading, and deterministic selection ordering
#   DEPENDS: M-ENGINE-CONTRACTS, M-ENGINE-CONFIG, M-ERRORS, M-OBSERVABILITY
#   LINKS: M-ENGINE-REGISTRY, M-DISCOVERY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   ENGINE_ENTRY_POINT_GROUP - Importlib entry-point group for external TTSEngine registrations.
#   EngineRegistryError - Typed registry failure for duplicate registration and unresolved selection paths.
#   EngineRegistry - Process-local registry for TTSEngine registration, lookup, and deterministic selection.
#   load_engine_registry - Build an EngineRegistry from explicit engines, built-ins, settings, and optional entry points.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Task 7 engine wave: introduced registry/discovery loading for TTSEngine with deterministic duplicate handling, disabled-config filtering, and warning-isolated optional entry-point loading]
# END_CHANGE_SUMMARY

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import TypeAlias

from core.engines.config import DisabledEngineConfig, EngineConfig, EngineSettings
from core.engines.contracts import TTSEngine
from core.errors import CoreError
from core.observability import log_event

LOGGER = logging.getLogger(__name__)
ENGINE_ENTRY_POINT_GROUP = "tts_server.engines"

EngineRegistrationSource: TypeAlias = str
EngineRegistrationCandidate: TypeAlias = TTSEngine | type[TTSEngine]


@dataclass(frozen=True)
class _RegisteredEngine:
    engine: TTSEngine
    config: EngineConfig | None
    source: EngineRegistrationSource
    registration_index: int
    aliases: tuple[str, ...]
    capabilities: tuple[str, ...]
    families: tuple[str, ...]
    backends: tuple[str, ...]
    languages: tuple[str, ...]


# START_CONTRACT: EngineRegistryError
#   PURPOSE: Represent deterministic engine-registry failures without reusing unrelated backend or model error types.
#   INPUTS: { reason: str - Human-readable registry failure message }
#   OUTPUTS: { instance - Typed engine registry error }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-REGISTRY
# END_CONTRACT: EngineRegistryError
class EngineRegistryError(CoreError):
    pass


# START_CONTRACT: EngineRegistry
#   PURPOSE: Hold a process-local set of TTSEngine instances and resolve deterministic matches by key or capability/language constraints.
#   INPUTS: { engines: Sequence[tuple[TTSEngine, EngineConfig | None, str]] | None - Optional initial registrations with config and source metadata }
#   OUTPUTS: { instance - Registry ready for lookup and resolution }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-REGISTRY
# END_CONTRACT: EngineRegistry
class EngineRegistry:
    def __init__(
        self,
        engines: Sequence[tuple[TTSEngine, EngineConfig | None, EngineRegistrationSource]] | None = None,
    ) -> None:
        self._registrations: dict[str, _RegisteredEngine] = {}
        self._token_index: dict[str, str] = {}
        self._registration_counter = 0
        for engine, config, source in engines or ():
            self.register(engine, config=config, source=source)

    @property
    def registered_engines(self) -> tuple[TTSEngine, ...]:
        return tuple(registration.engine for registration in self._sorted_registrations())

    # START_CONTRACT: register
    #   PURPOSE: Register one TTSEngine under its declared key and optional aliases while rejecting duplicates deterministically.
    #   INPUTS: { engine: TTSEngine - Engine instance to register, config: EngineConfig | None - Optional typed config matched to the engine, source: str - Registration source label for diagnostics }
    #   OUTPUTS: { None }
    #   SIDE_EFFECTS: Mutates the registry
    #   LINKS: M-ENGINE-REGISTRY
    # END_CONTRACT: register
    def register(
        self,
        engine: TTSEngine,
        *,
        config: EngineConfig | None = None,
        source: EngineRegistrationSource = "explicit",
    ) -> None:
        key = _normalize_token(getattr(engine, "key", ""))
        if not key:
            raise EngineRegistryError("TTSEngine must declare a non-empty 'key'")
        if key in self._registrations:
            raise EngineRegistryError(f"EngineRegistry duplicate engine key '{key}'")

        aliases = _collect_aliases(engine=engine, config=config)
        for token in (key, *aliases):
            existing = self._token_index.get(token)
            if existing is not None:
                raise EngineRegistryError(f"EngineRegistry duplicate engine key '{token}'")

        capabilities = _effective_capabilities(engine=engine, config=config)
        families = _effective_families(engine=engine, config=config)
        backends = tuple(_normalize_tokens(engine.capabilities().backends))
        languages = _effective_languages(engine=engine, config=config)
        registration = _RegisteredEngine(
            engine=engine,
            config=config,
            source=source,
            registration_index=self._registration_counter,
            aliases=aliases,
            capabilities=capabilities,
            families=families,
            backends=backends,
            languages=languages,
        )
        self._registration_counter += 1
        self._registrations[key] = registration
        for token in (key, *aliases):
            self._token_index[token] = key

    # START_CONTRACT: get
    #   PURPOSE: Resolve a registered engine by key or alias without applying capability/language filtering.
    #   INPUTS: { key: str - Engine key or configured alias }
    #   OUTPUTS: { TTSEngine | None - Registered engine when present }
    #   SIDE_EFFECTS: none
    #   LINKS: M-ENGINE-REGISTRY
    # END_CONTRACT: get
    def get(self, key: str) -> TTSEngine | None:
        registration = self._lookup_registration(key)
        if registration is None:
            return None
        return registration.engine

    # START_CONTRACT: keys
    #   PURPOSE: List registered engine keys in deterministic registry order.
    #   INPUTS: {}
    #   OUTPUTS: { tuple[str, ...] - Registered engine keys }
    #   SIDE_EFFECTS: none
    #   LINKS: M-ENGINE-REGISTRY
    # END_CONTRACT: keys
    def keys(self) -> tuple[str, ...]:
        return tuple(registration.engine.key for registration in self._sorted_registrations())

    # START_CONTRACT: resolve_engine
    #   PURPOSE: Resolve the best engine by explicit key or by capability/family/backend/language constraints using deterministic priority and registration order.
    #   INPUTS: { engine_key: str | None - Explicit key or alias to prefer, capability: str | None - Required synthesis capability, language: str | None - Optional requested language code, family: str | None - Optional family constraint, backend_key: str | None - Optional backend constraint, require_available: bool - When True only return engines whose availability reports enabled and available state }
    #   OUTPUTS: { TTSEngine - Selected registered engine }
    #   SIDE_EFFECTS: none
    #   LINKS: M-ENGINE-REGISTRY
    # END_CONTRACT: resolve_engine
    def resolve_engine(
        self,
        *,
        engine_key: str | None = None,
        capability: str | None = None,
        language: str | None = None,
        family: str | None = None,
        backend_key: str | None = None,
        require_available: bool = True,
    ) -> TTSEngine:
        # START_BLOCK_RESOLVE_ENGINE
        requested_language = _normalize_optional_token(language)
        requested_family = _normalize_optional_token(family)
        requested_backend = _normalize_optional_token(backend_key)

        if engine_key is not None:
            registration = self._lookup_registration(engine_key)
            if registration is None:
                raise EngineRegistryError(f"No registered engine matches key '{engine_key}'")
            self._ensure_registration_matches(
                registration=registration,
                capability=capability,
                language=requested_language,
                family=requested_family,
                backend_key=requested_backend,
                require_available=require_available,
            )
            return registration.engine

        candidates: list[tuple[tuple[int, int, int, str], _RegisteredEngine]] = []
        normalized_capability = _normalize_optional_token(capability)
        for registration in self._sorted_registrations():
            if not _registration_matches(
                registration=registration,
                capability=normalized_capability,
                language=requested_language,
                family=requested_family,
                backend_key=requested_backend,
                require_available=require_available,
            ):
                continue
            priority = _effective_priority(registration.config)
            language_rank = _language_rank(registration.languages, requested_language)
            candidates.append(
                (
                    (priority, language_rank, registration.registration_index, registration.engine.key),
                    registration,
                )
            )

        if not candidates:
            raise EngineRegistryError(
                "No registered engine matched the requested constraints"
            )

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1].engine
        # END_BLOCK_RESOLVE_ENGINE

    def _lookup_registration(self, key: str) -> _RegisteredEngine | None:
        normalized = _normalize_token(key)
        resolved_key = self._token_index.get(normalized)
        if resolved_key is None:
            return None
        return self._registrations[resolved_key]

    def _sorted_registrations(self) -> tuple[_RegisteredEngine, ...]:
        return tuple(
            sorted(
                self._registrations.values(),
                key=lambda registration: (
                    _effective_priority(registration.config),
                    registration.registration_index,
                    registration.engine.key,
                ),
            )
        )

    def _ensure_registration_matches(
        self,
        *,
        registration: _RegisteredEngine,
        capability: str | None,
        language: str | None,
        family: str | None,
        backend_key: str | None,
        require_available: bool,
    ) -> None:
        if not _registration_matches(
            registration=registration,
            capability=_normalize_optional_token(capability),
            language=language,
            family=family,
            backend_key=backend_key,
            require_available=require_available,
        ):
            raise EngineRegistryError(
                f"Registered engine '{registration.engine.key}' does not satisfy the requested constraints"
            )


# START_CONTRACT: load_engine_registry
#   PURPOSE: Build an EngineRegistry from explicit engines, built-in candidates, parsed engine settings, and optional entry-point registrations with fail-fast control.
#   INPUTS: { explicit_engines: Sequence[TTSEngine | type[TTSEngine]] - Caller-provided engine registrations, built_in_engines: Sequence[TTSEngine | type[TTSEngine]] - Built-in engine registrations, settings: EngineSettings | None - Parsed engine settings used to match configs and skip disabled entries, include_entry_points: bool - Whether to load optional importlib entry points, entry_points_loader: callable | None - Optional loader override for tests, fail_fast: bool - Whether optional entry-point load failures should raise instead of warn-and-skip }
#   OUTPUTS: { EngineRegistry - Built registry }
#   SIDE_EFFECTS: May import entry-point objects and emit warning logs for skipped optional failures
#   LINKS: M-ENGINE-REGISTRY, M-DISCOVERY
# END_CONTRACT: load_engine_registry
def load_engine_registry(
    *,
    explicit_engines: Sequence[EngineRegistrationCandidate] = (),
    built_in_engines: Sequence[EngineRegistrationCandidate] = (),
    settings: EngineSettings | None = None,
    include_entry_points: bool = True,
    entry_points_loader: Callable[[], Iterable[EntryPoint]] | None = None,
    fail_fast: bool = False,
) -> EngineRegistry:
    registry = EngineRegistry()
    config_index = _build_config_index(settings)

    # START_BLOCK_REGISTER_ENGINE_CANDIDATES
    registration_stream: list[tuple[EngineRegistrationCandidate, str]] = []
    registration_stream.extend((candidate, "built_in") for candidate in built_in_engines)
    registration_stream.extend((candidate, "explicit") for candidate in explicit_engines)
    if include_entry_points:
        registration_stream.extend(
            (candidate, "entry_point")
            for candidate in _load_entry_point_engines(
                loader=entry_points_loader,
                fail_fast=fail_fast,
            )
        )

    for candidate, source in registration_stream:
        engine = _coerce_engine_candidate(candidate)
        config = _match_engine_config(engine=engine, config_index=config_index)
        if isinstance(config, DisabledEngineConfig):
            log_event(
                LOGGER,
                level=logging.INFO,
                event="[EngineRegistry][load_engine_registry][REGISTER_ENGINE_CANDIDATES]",
                message="Skipping disabled engine config registration",
                engine_key=engine.key,
                config_name=config.name,
                source=source,
            )
            continue
        registry.register(engine, config=config, source=source)
    # END_BLOCK_REGISTER_ENGINE_CANDIDATES

    return registry


def _build_config_index(
    settings: EngineSettings | None,
) -> dict[str, EngineConfig]:
    index: dict[str, EngineConfig] = {}
    if settings is None:
        return index
    for config in settings.engines:
        tokens = (config.name, *getattr(config, "aliases", ()))
        for token in tokens:
            index[_normalize_token(token)] = config
    return index


def _match_engine_config(
    *,
    engine: TTSEngine,
    config_index: dict[str, EngineConfig],
) -> EngineConfig | None:
    tokens = [engine.key]
    aliases = getattr(engine, "aliases", ())
    if isinstance(aliases, Iterable) and not isinstance(aliases, (str, bytes, bytearray)):
        tokens.extend(str(alias) for alias in aliases)
    for token in tokens:
        matched = config_index.get(_normalize_token(token))
        if matched is not None:
            return matched
    return None


def _coerce_engine_candidate(candidate: EngineRegistrationCandidate) -> TTSEngine:
    if isinstance(candidate, TTSEngine):
        return candidate
    if isinstance(candidate, type):
        if not issubclass(candidate, TTSEngine):
            raise EngineRegistryError(
                f"Entry candidate '{candidate.__module__}.{candidate.__qualname__}' is not a TTSEngine subclass"
            )
        if inspect.isabstract(candidate):
            raise EngineRegistryError(
                f"TTSEngine candidate '{candidate.__module__}.{candidate.__qualname__}' is abstract"
            )
        return candidate()
    raise EngineRegistryError(
        f"Engine candidate must be a TTSEngine instance or subclass, got {type(candidate).__name__}"
    )


def _load_entry_point_engines(
    *,
    loader: Callable[[], Iterable[EntryPoint]] | None,
    fail_fast: bool,
) -> tuple[EngineRegistrationCandidate, ...]:
    loaded: list[EngineRegistrationCandidate] = []
    for entry in _resolve_entry_points(loader=loader):
        try:
            obj = entry.load()
            if isinstance(obj, TTSEngine):
                loaded.append(obj)
                continue
            if not isinstance(obj, type):
                raise EngineRegistryError(
                    f"entry point '{getattr(entry, 'name', '<unknown>')}' did not resolve to a class or TTSEngine instance"
                )
            if not issubclass(obj, TTSEngine):
                raise EngineRegistryError(
                    f"entry point '{getattr(entry, 'name', '<unknown>')}' is not a subclass of TTSEngine"
                )
            if inspect.isabstract(obj):
                raise EngineRegistryError(
                    f"entry point '{getattr(entry, 'name', '<unknown>')}' resolved to an abstract TTSEngine class"
                )
            loaded.append(obj)
        except Exception as exc:
            if fail_fast:
                raise
            log_event(
                LOGGER,
                level=logging.WARNING,
                event="[EngineRegistry][load_engine_registry][LOAD_ENTRY_POINTS]",
                message="Skipping optional engine entry point after load failure",
                entry_point=getattr(entry, "name", "<unknown>"),
                group=ENGINE_ENTRY_POINT_GROUP,
                error=str(exc),
            )
    return tuple(loaded)


def _resolve_entry_points(
    *,
    loader: Callable[[], Iterable[EntryPoint]] | None,
) -> tuple[EntryPoint, ...]:
    if loader is not None:
        if not callable(loader):
            raise TypeError("entry_points_loader must be callable when provided")
        iterable = loader()
        return tuple(iterable)
    selected = entry_points()
    selector = getattr(selected, "select", None)
    if selector is not None:
        return tuple(selector(group=ENGINE_ENTRY_POINT_GROUP))
    return tuple(getattr(selected, "get", lambda _key, default=(): default)(ENGINE_ENTRY_POINT_GROUP, ()))


def _registration_matches(
    *,
    registration: _RegisteredEngine,
    capability: str | None,
    language: str | None,
    family: str | None,
    backend_key: str | None,
    require_available: bool,
) -> bool:
    if capability is not None and capability not in registration.capabilities:
        return False
    if family is not None and registration.families and family not in registration.families:
        return False
    if backend_key is not None and registration.backends and backend_key not in registration.backends:
        return False
    if language is not None and registration.languages and language not in registration.languages:
        return False
    if require_available:
        availability = registration.engine.availability()
        if not availability.is_enabled or not availability.is_available:
            return False
    return True


def _collect_aliases(
    *,
    engine: TTSEngine,
    config: EngineConfig | None,
) -> tuple[str, ...]:
    aliases: list[str] = []
    raw_engine_aliases = getattr(engine, "aliases", ())
    if isinstance(raw_engine_aliases, Iterable) and not isinstance(
        raw_engine_aliases, (str, bytes, bytearray)
    ):
        aliases.extend(str(alias) for alias in raw_engine_aliases)
    if config is not None:
        aliases.extend(getattr(config, "aliases", ()))
        aliases.append(getattr(config, "name", ""))
    normalized = _normalize_tokens(aliases)
    return tuple(token for token in normalized if token != _normalize_token(engine.key))


def _effective_capabilities(
    *,
    engine: TTSEngine,
    config: EngineConfig | None,
) -> tuple[str, ...]:
    if config is not None and not isinstance(config, DisabledEngineConfig):
        return tuple(_normalize_tokens(config.capabilities))
    return tuple(_normalize_tokens(engine.capabilities().capabilities))


def _effective_families(
    *,
    engine: TTSEngine,
    config: EngineConfig | None,
) -> tuple[str, ...]:
    if config is not None and not isinstance(config, DisabledEngineConfig):
        return (_normalize_token(config.family),)
    return tuple(_normalize_tokens(engine.capabilities().families))


def _effective_languages(
    *,
    engine: TTSEngine,
    config: EngineConfig | None,
) -> tuple[str, ...]:
    params = getattr(config, "params", {}) if config is not None else {}
    raw = params.get("languages") or params.get("language")
    if raw is None:
        raw = getattr(engine, "languages", ())
    return _normalize_tokens(raw)


def _effective_priority(config: EngineConfig | None) -> int:
    if config is None or isinstance(config, DisabledEngineConfig):
        return 100
    return int(config.priority)


def _language_rank(languages: tuple[str, ...], requested_language: str | None) -> int:
    if requested_language is None:
        return 0
    if not languages:
        return 1
    return 0 if requested_language in languages else 2


def _normalize_tokens(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, Iterable):
        values = [str(item).strip() for item in raw]
    else:
        values = [str(raw).strip()]

    normalized: list[str] = []
    for value in values:
        token = _normalize_optional_token(value)
        if token is not None and token not in normalized:
            normalized.append(token)
    return tuple(normalized)


def _normalize_optional_token(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def _normalize_token(value: object) -> str:
    normalized = _normalize_optional_token(value)
    return normalized or ""


__all__ = [
    "ENGINE_ENTRY_POINT_GROUP",
    "EngineRegistry",
    "EngineRegistryError",
    "load_engine_registry",
]

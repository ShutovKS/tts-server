# FILE: core/engines/runtime_factory.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Build the runtime EngineRegistry from built-in TTSEngine implementations and optional entry-point discovery outside the service layer.
#   SCOPE: Built-in engine candidate assembly, cache wrapping, engine settings parsing, and deterministic registry construction for runtime bootstrap
#   DEPENDS: M-CONFIG, M-ENGINE-REGISTRY, M-ENGINE-CONTRACTS, M-ENGINE-MODEL-CACHE, M-ENGINE-PIPER, M-ENGINE-QWEN3, M-ENGINE-OMNIVOICE
#   LINKS: M-BOOTSTRAP, M-ENGINE-REGISTRY
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   CachedEngine - Cache-decorating TTSEngine wrapper used by runtime registry composition
#   build_engine_settings - Parse typed runtime engine settings from CoreSettings
#   build_engine_registry - Build the runtime EngineRegistry from built-in engines plus tts_server.engines entry points
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Added cache wrapping for built-in and entry-point TTSEngine registrations through the runtime registry factory]
# END_CHANGE_SUMMARY

from __future__ import annotations

from core.config import CoreSettings
from core.engines.config import DisabledEngineConfig, EngineConfig, EngineSettings, parse_engine_settings
from core.engines.contracts import AudioBuffer, EngineAvailability, EngineCapabilities, ModelHandle, SynthesisJob, TTSEngine
from core.engines.model_cache import ModelCache, ModelCacheKey
from core.engines.omnivoice import OmniVoiceTorchEngine
from core.engines.piper import PiperOnnxEngine
from core.engines.qwen3 import Qwen3TorchEngine
from core.engines.registry import EngineRegistry, load_engine_registry


# START_CONTRACT: CachedEngine
#   PURPOSE: Wrap a TTSEngine with ModelCache-backed load_model reuse while delegating capabilities, availability, and synthesis unchanged.
#   INPUTS: { engine: TTSEngine - Wrapped engine instance, cache: ModelCache - Cache used for load_model reuse }
#   OUTPUTS: { instance - Cache-aware engine wrapper }
#   SIDE_EFFECTS: none on construction
#   LINKS: M-ENGINE-RUNTIME-FACTORY, M-ENGINE-MODEL-CACHE
# END_CONTRACT: CachedEngine
class CachedEngine(TTSEngine):
    def __init__(self, engine: TTSEngine, cache: ModelCache) -> None:
        self._engine = engine
        self._cache = cache
        self.key = engine.key
        self.label = engine.label
        aliases = getattr(engine, "aliases", ())
        self.aliases = tuple(aliases) if isinstance(aliases, tuple) else tuple(aliases or ())

    def capabilities(self) -> EngineCapabilities:
        return self._engine.capabilities()

    def availability(self) -> EngineAvailability:
        return self._engine.availability()

    def load_model(self, *, spec, backend_key: str, model_path) -> ModelHandle:
        key = ModelCacheKey.from_parts(
            engine_key=self.key,
            model_id=spec.model_id,
            backend_key=backend_key,
            model_path=model_path,
        )
        return self._cache.get_or_load(
            key,
            lambda: self._engine.load_model(
                spec=spec,
                backend_key=backend_key,
                model_path=model_path,
            ),
        )

    def synthesize(self, handle: ModelHandle, job: SynthesisJob) -> AudioBuffer:
        return self._engine.synthesize(handle, job)

    def clear_cache(self) -> None:
        self._cache.clear()


def _wrap_engine_with_cache(engine: TTSEngine, settings: EngineSettings) -> TTSEngine:
    config = _find_config_for_engine(engine, settings)
    return _wrap_engine_with_cache_config(engine, config)


def _wrap_engine_with_cache_config(
    engine: TTSEngine,
    config: EngineConfig | None,
) -> TTSEngine:
    cache_size = 1 if config is None or isinstance(config, DisabledEngineConfig) else config.model_cache_size
    return CachedEngine(engine, ModelCache(max_entries=cache_size))


def _find_config_for_engine(engine: TTSEngine, settings: EngineSettings):
    tokens = {engine.key.casefold()}
    aliases = getattr(engine, "aliases", ())
    tokens.update(str(alias).casefold() for alias in aliases or ())
    for config in settings.enabled_engines:
        config_tokens = {config.name.casefold(), *(alias.casefold() for alias in config.aliases)}
        if tokens & config_tokens:
            return config
    return None


# START_CONTRACT: build_engine_settings
#   PURPOSE: Parse typed engine settings from CoreSettings without injecting synthetic disabled-engine entries.
#   INPUTS: { settings: CoreSettings - Runtime settings containing engine_configs }
#   OUTPUTS: { EngineSettings - Parsed typed engine settings collection }
#   SIDE_EFFECTS: none
#   LINKS: M-ENGINE-RUNTIME-FACTORY, M-ENGINE-CONFIG
# END_CONTRACT: build_engine_settings
def build_engine_settings(settings: CoreSettings) -> EngineSettings:
    return parse_engine_settings({"engines": list(settings.engine_configs)})


# START_CONTRACT: build_engine_registry
#   PURPOSE: Build the process-local engine registry from built-in runtime engines and optional external engine entry points.
#   INPUTS: { settings: CoreSettings - Runtime settings containing engine-route toggles }
#   OUTPUTS: { EngineRegistry - Registry populated with built-in engines and successfully loaded entry-point engines }
#   SIDE_EFFECTS: May import optional entry-point objects through load_engine_registry and logs isolated entry-point failures there
#   LINKS: M-BOOTSTRAP, M-ENGINE-REGISTRY
# END_CONTRACT: build_engine_registry
def build_engine_registry(settings: CoreSettings) -> EngineRegistry:
    built_in_engines = [Qwen3TorchEngine, OmniVoiceTorchEngine, PiperOnnxEngine]
    engine_settings = build_engine_settings(settings)
    return load_engine_registry(
        built_in_engines=tuple(built_in_engines),
        settings=engine_settings,
        include_entry_points=True,
        engine_wrapper=_wrap_engine_with_cache_config,
        fail_fast=False,
    )


__all__ = ["CachedEngine", "build_engine_registry", "build_engine_settings"]

# FILE: tests/unit/core/test_engine_proof_family.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Unit tests for a test-only proof family engine that exercises the TTSEngine and EngineRegistry seam without touching production API, service, or dispatcher code.
#   SCOPE: proof-family registration, deterministic selection through EngineRegistry, fake synthesis through TTSEngine, and invalid proof-family config failure coverage
#   DEPENDS: M-ENGINE-CONTRACTS, M-ENGINE-CONFIG, M-ENGINE-REGISTRY
#   LINKS: V-M-ENGINE-REGISTRY, V-M-ENGINE-CONFIG, V-M-ENGINE-STATIC-COUPLING
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _ProofFamilyEngine - Test-only TTSEngine implementation with deterministic fake synthesis output.
#   build_proof_family_engine - Helper that constructs the proof family engine with stable identifiers.
#   test_proof_family_registry_selection_and_fake_synthesis - Verifies the proof family can be discovered, selected, loaded, and synthesized through the engine path.
#   test_proof_family_config_rejects_invalid_payloads - Verifies invalid proof-family config payloads fail deterministically before registry registration.
#   test_proof_family_not_hardcoded_in_runtime_files - Verifies the proof family name is absent from the server, service, and dispatcher runtime files.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Task 9 proof family: added a test-only engine family exercising TTSEngine and EngineRegistry with deterministic fake synthesis and invalid config coverage]
# END_CHANGE_SUMMARY

from __future__ import annotations

from pathlib import Path

import pytest

from core.engines import (
    AudioBuffer,
    EngineAvailability,
    EngineCapabilities,
    EngineRegistry,
    EngineRegistryError,
    ModelHandle,
    SynthesisJob,
    TTSEngine,
    load_engine_registry,
    parse_engine_settings,
)
from core.models.catalog import MODEL_SPECS


pytestmark = pytest.mark.unit


class _ProofFamilyEngine(TTSEngine):
    def __init__(self) -> None:
        self.key = "proof-family"
        self.label = "Proof Family"
        self.aliases = ("proof-family-config", "proof-family-alias")
        self.languages = ("en",)
        self.load_calls: list[tuple[str, str, str | None]] = []
        self.synthesize_calls: list[tuple[str, str]] = []

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            families=("proof_family",),
            backends=("torch",),
            capabilities=("preset_speaker_tts",),
        )

    def availability(self) -> EngineAvailability:
        return EngineAvailability(
            engine_key=self.key,
            is_available=True,
            is_enabled=True,
        )

    def load_model(
        self,
        *,
        spec,
        backend_key: str,
        model_path: Path | None,
    ) -> ModelHandle:
        self.load_calls.append((spec.key, backend_key, None if model_path is None else str(model_path)))
        return ModelHandle(
            spec=spec,
            runtime_model={"proof": spec.key, "backend": backend_key},
            resolved_path=model_path,
            engine_key=self.key,
            backend_key=backend_key,
            family_key="proof_family",
        )

    def synthesize(self, handle: ModelHandle, job: SynthesisJob) -> AudioBuffer:
        self.synthesize_calls.append((handle.spec.key, job.text))
        return AudioBuffer(
            waveform=b"proof-family:" + job.text.encode("utf-8"),
            sample_rate=16000,
            audio_format="wav",
        )


def build_proof_family_engine() -> _ProofFamilyEngine:
    return _ProofFamilyEngine()


def test_proof_family_registry_selection_and_fake_synthesis(tmp_path: Path) -> None:
    engine = build_proof_family_engine()
    settings = parse_engine_settings(
        {
            "engines": [
                {
                    "kind": "torch",
                    "name": "proof-family-config",
                    "aliases": ["proof-family-config-alias"],
                    "family": "proof_family",
                    "capabilities": ["preset_speaker_tts"],
                    "priority": 5,
                    "params": {"languages": ["en"]},
                }
            ]
        }
    )

    registry = load_engine_registry(
        explicit_engines=(engine,),
        settings=settings,
        include_entry_points=False,
    )
    resolved = registry.resolve_engine(
        capability="preset_speaker_tts",
        family="proof_family",
        language="en",
        backend_key="torch",
    )
    spec = next(iter(MODEL_SPECS.values()))
    handle = resolved.load_model(spec=spec, backend_key="torch", model_path=tmp_path / "proof-model")
    job = SynthesisJob(
        capability="preset_speaker_tts",
        execution_mode="custom",
        text="proof synthesis",
        language="en",
        output_dir=tmp_path,
        payload={"speaker": "proof"},
    )
    audio = resolved.synthesize(handle, job)

    assert registry.keys() == ("proof-family",)
    assert registry.get("proof-family") is engine
    assert registry.get("proof-family-config") is engine
    assert registry.get("proof-family-config-alias") is engine
    assert resolved is engine
    assert handle.engine_key == "proof-family"
    assert handle.family_key == "proof_family"
    assert handle.runtime_model == {"proof": spec.key, "backend": "torch"}
    assert engine.load_calls == [(spec.key, "torch", str(tmp_path / "proof-model"))]
    assert engine.synthesize_calls == [(spec.key, "proof synthesis")]
    assert audio.waveform == b"proof-family:proof synthesis"
    assert audio.sample_rate == 16000
    assert audio.audio_format == "wav"


def test_proof_family_config_rejects_invalid_payloads() -> None:
    with pytest.raises((TypeError, ValueError), match="capabilities"):
        parse_engine_settings(
            {
                "engines": [
                    {
                        "kind": "torch",
                        "name": "proof-family-invalid",
                        "family": "proof_family",
                        "capabilities": [],
                    }
                ]
            }
        )


def test_proof_family_not_hardcoded_in_runtime_files() -> None:
    forbidden_files = (
        Path("server/api/routes_tts.py"),
        Path("core/services/tts_service.py"),
        Path("core/backends/torch_backend/dispatcher.py"),
    )
    forbidden_tokens = ("test-family", "TestFamily", "proof_family", "proof-family")

    for file_path in forbidden_files:
        text = file_path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"{token} unexpectedly found in {file_path}"

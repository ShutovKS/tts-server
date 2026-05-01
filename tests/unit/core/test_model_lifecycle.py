# FILE: tests/unit/core/test_model_lifecycle.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Verify ModelLifecycleService delete/refresh/submit_download/get_download semantics for the HTTP control plane.
#   SCOPE: delete_model success and 404 paths, default no-downloader-configured failure, custom downloader success path, refresh hook detection, get_download/list_downloads tracking.
#   DEPENDS: M-MODEL-LIFECYCLE
#   LINKS: V-M-MODEL-LIFECYCLE
#   ROLE: TEST
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   _make_spec - Build a SimpleNamespace stand-in for ModelSpec used by the lifecycle facade.
#   _make_registry - Build a SimpleNamespace registry exposing model_specs and an optional reload_manifest hook.
#   test_delete_model_removes_existing_folder - Verifies delete_model returns True and removes the folder when it exists.
#   test_delete_model_returns_false_for_unknown_model_id - Verifies delete_model returns False for unregistered ids.
#   test_delete_model_returns_false_when_folder_missing - Verifies delete_model returns False when the spec exists but the folder is already absent.
#   test_submit_download_uses_default_downloader - Verifies the default downloader produces a "no_downloader_configured" failure.
#   test_submit_download_uses_injected_downloader - Verifies an injected downloader runs synchronously when run_async is False.
#   test_submit_download_unknown_model_fails - Verifies unknown model_ids fail before the downloader runs.
#   test_refresh_uses_reload_manifest_hook - Verifies refresh prefers reload_manifest when available.
#   test_refresh_reports_unsupported_when_no_hook - Verifies refresh returns supported=False when no hook is exposed.
#   test_get_download_returns_none_for_unknown_id - Verifies get_download returns None for unknown ids.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Phase 4.13: introduced unit coverage for ModelLifecycleService delete/submit_download/get_download/refresh paths]
# END_CHANGE_SUMMARY

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.services.model_lifecycle import (
    MODEL_DOWNLOAD_STATUSES,
    ModelDownloadJob,
    ModelLifecycleService,
)

pytestmark = pytest.mark.unit


def _make_spec(
    *, folder: str, model_id: str | None = None, key: str | None = None
) -> SimpleNamespace:
    resolved_id = model_id or folder
    return SimpleNamespace(
        api_name=folder,
        folder=folder,
        key=key or folder,
        model_id=resolved_id,
    )


def _make_registry(
    specs: tuple[SimpleNamespace, ...], *, reload_calls: list[str] | None = None
) -> SimpleNamespace:
    if reload_calls is None:
        return SimpleNamespace(model_specs=specs)

    def _reload() -> None:
        reload_calls.append("reload_manifest")

    return SimpleNamespace(model_specs=specs, reload_manifest=_reload)


def test_delete_model_removes_existing_folder(tmp_path: Path) -> None:
    spec = _make_spec(folder="ModelA", model_id="model-a")
    registry = _make_registry((spec,))
    folder = tmp_path / "ModelA"
    folder.mkdir()
    (folder / "config.json").write_text("{}")

    service = ModelLifecycleService(models_dir=tmp_path, registry=registry)

    assert service.delete_model("model-a") is True
    assert not folder.exists()


def test_delete_model_returns_false_for_unknown_model_id(tmp_path: Path) -> None:
    registry = _make_registry(())
    service = ModelLifecycleService(models_dir=tmp_path, registry=registry)
    assert service.delete_model("unknown") is False


def test_delete_model_returns_false_when_folder_missing(tmp_path: Path) -> None:
    spec = _make_spec(folder="ModelA")
    registry = _make_registry((spec,))
    service = ModelLifecycleService(models_dir=tmp_path, registry=registry)
    assert service.delete_model("ModelA") is False


def test_submit_download_uses_default_downloader(tmp_path: Path) -> None:
    spec = _make_spec(folder="ModelA")
    registry = _make_registry((spec,))
    service = ModelLifecycleService(models_dir=tmp_path, registry=registry)

    job = service.submit_download("ModelA", run_async=False)

    assert job.status in MODEL_DOWNLOAD_STATUSES
    assert job.status == "failed"
    assert job.error == "no_downloader_configured"
    assert job.completed_at is not None
    assert service.get_download(job.id) == job


def test_submit_download_uses_injected_downloader(tmp_path: Path) -> None:
    spec = _make_spec(folder="ModelA")
    registry = _make_registry((spec,))

    captured: dict[str, Path] = {}

    def downloader(job: ModelDownloadJob, target_dir: Path) -> ModelDownloadJob:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "config.json").write_text("{}")
        captured["target"] = target_dir
        return replace(job, status="succeeded", progress=1.0, completed_at=job.updated_at)

    service = ModelLifecycleService(models_dir=tmp_path, registry=registry, downloader=downloader)
    job = service.submit_download("ModelA", source="hf://example/repo", run_async=False)

    assert job.status == "succeeded"
    assert job.progress == 1.0
    assert captured["target"] == tmp_path / "ModelA"
    assert (captured["target"] / "config.json").read_text() == "{}"


def test_submit_download_unknown_model_fails(tmp_path: Path) -> None:
    registry = _make_registry(())
    service = ModelLifecycleService(models_dir=tmp_path, registry=registry)
    job = service.submit_download("missing", run_async=False)

    assert job.status == "failed"
    assert job.error == "unknown_model_id"


def test_refresh_uses_reload_manifest_hook(tmp_path: Path) -> None:
    spec = _make_spec(folder="ModelA")
    calls: list[str] = []
    registry = _make_registry((spec,), reload_calls=calls)
    service = ModelLifecycleService(models_dir=tmp_path, registry=registry)

    outcome = service.refresh()

    assert outcome["supported"] is True
    assert outcome["method"] == "reload_manifest"
    assert outcome["model_count"] == 1
    assert calls == ["reload_manifest"]


def test_refresh_reports_unsupported_when_no_hook(tmp_path: Path) -> None:
    registry = _make_registry(())
    service = ModelLifecycleService(models_dir=tmp_path, registry=registry)

    outcome = service.refresh()

    assert outcome["supported"] is False
    assert outcome["model_count"] == 0


def test_get_download_returns_none_for_unknown_id(tmp_path: Path) -> None:
    registry = _make_registry(())
    service = ModelLifecycleService(models_dir=tmp_path, registry=registry)
    assert service.get_download("missing-id") is None
    assert service.list_downloads() == ()

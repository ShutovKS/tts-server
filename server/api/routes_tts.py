# FILE: server/api/routes_tts.py
# VERSION: 1.3.0
# START_MODULE_CONTRACT
#   PURPOSE: Expose the public façade for synchronous and async TTS HTTP route registration.
#   SCOPE: register_tts_routes façade plus stable helper re-exports used by existing tests and imports
#   DEPENDS: M-APPLICATION, M-CONTRACTS, M-ERRORS, M-OBSERVABILITY, M-STREAMING
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   T - Generic type variable used by async timeout helper utilities
#   build_text_length_error - Build validation errors for invalid text lengths
#   build_upload_validation_error - Build validation errors for invalid clone uploads
#   enforce_text_length - Validate text against configured length limits
#   current_principal_id - Read current principal id from request state
#   resolve_idempotency_scope - Resolve idempotency scope for async submissions
#   ensure_requested_model_capability - Validate an explicitly requested model against the requested normalized synthesis capability
#   build_job_urls - Build status, result, and cancel URLs for async jobs
#   public_job_status - Convert internal job statuses into the frozen public async lifecycle
#   build_job_snapshot_payload - Convert internal job snapshots to public payloads
#   get_job_snapshot_or_raise - Load a job snapshot and enforce owner access
#   build_idempotency_fingerprint - Build deterministic async job idempotency fingerprints
#   create_custom_job_submission_from_openai - Build async custom submissions from OpenAI payloads
#   create_custom_job_submission_from_custom - Build async custom submissions from custom payloads
#   create_design_job_submission - Build async voice design submissions
#   validate_clone_upload - Validate clone upload metadata and content type
#   build_clone_staged_path - Build a staging path for uploaded clone audio
#   stage_clone_job_submission - Persist uploaded clone media and build async clone submissions
#   run_inference_with_timeout - Run synthesis with timeout handling for sync routes
#   register_tts_routes - Register synchronous and async TTS routes on the FastAPI app through split private registration modules
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.3.0 - Split TTS endpoint registration into private concern modules while keeping register_tts_routes as the public façade and preserving route behavior, route names, and log markers]
# END_CHANGE_SUMMARY

from __future__ import annotations

from fastapi import FastAPI  # pyright: ignore[reportMissingImports]

from server.api.tts._helpers import (  # pyright: ignore[reportMissingImports]
    build_clone_staged_path,
    build_idempotency_fingerprint,
    build_job_snapshot_payload,
    build_text_length_error,
    build_upload_validation_error,
    create_custom_job_submission_from_custom,
    create_custom_job_submission_from_openai,
    create_design_job_submission,
    current_principal_id,
    ensure_requested_model_capability,
    get_job_snapshot_or_raise,
    public_job_status,
    resolve_idempotency_scope,
    stage_clone_job_submission,
    validate_clone_upload,
    enforce_text_length,
)
from server.api.tts._timeout import run_inference_with_timeout  # pyright: ignore[reportMissingImports]
from server.api.tts.clone import register_clone_tts_routes
from server.api.tts.custom import register_custom_tts_routes
from server.api.tts.design import register_design_tts_routes
from server.api.tts.jobs import register_tts_job_routes
from server.api.tts.openai import register_openai_tts_routes
from server.api.tts.stream import register_stream_tts_routes

# START_CONTRACT: register_tts_routes
#   PURPOSE: Register synchronous and asynchronous TTS HTTP endpoints on the FastAPI application.
#   INPUTS: { app: FastAPI - application to attach routes to, logger: Any - structured logger used by endpoint handlers }
#   OUTPUTS: { None - routes are attached in place }
#   SIDE_EFFECTS: Mutates FastAPI routing table by registering TTS endpoints
#   LINKS: M-SERVER, M-APPLICATION
# END_CONTRACT: register_tts_routes
def register_tts_routes(app: FastAPI, logger) -> None:
    register_openai_tts_routes(app, logger)
    register_custom_tts_routes(app, logger)
    register_stream_tts_routes(app, logger)
    register_design_tts_routes(app, logger)
    register_clone_tts_routes(app, logger)
    register_tts_job_routes(app, logger)


__all__ = [
    "build_clone_staged_path",
    "build_idempotency_fingerprint",
    "build_job_snapshot_payload",
    "build_text_length_error",
    "build_upload_validation_error",
    "create_custom_job_submission_from_custom",
    "create_custom_job_submission_from_openai",
    "create_design_job_submission",
    "current_principal_id",
    "ensure_requested_model_capability",
    "enforce_text_length",
    "get_job_snapshot_or_raise",
    "public_job_status",
    "register_tts_routes",
    "resolve_idempotency_scope",
    "run_inference_with_timeout",
    "stage_clone_job_submission",
    "validate_clone_upload",
]

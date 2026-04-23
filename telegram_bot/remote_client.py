# pyright: reportMissingImports=false
# FILE: telegram_bot/remote_client.py
# VERSION: 1.1.0
# START_MODULE_CONTRACT
#   PURPOSE: Provide a Telegram-side HTTP client for the canonical remote server contract.
#   SCOPE: Readiness and model discovery, async submit/status/result flows, controlled error decoding, correlation header capture
#   DEPENDS: M-ERRORS
#   LINKS: M-TELEGRAM, M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   LOGGER - Module logger for Telegram remote server client events
#   RemoteServerRetryConfig - Retry policy for canonical remote server requests
#   RemoteServerCorrelation - Captured request and job correlation headers
#   RemoteServerErrorEnvelope - Structured decoded server error payload
#   RemoteServerRequestError - Base exception for remote server request failures
#   RemoteServerTransportError - Transport failure for unreachable or timed out remote servers
#   RemoteServerAPIError - Controlled remote server error with decoded public envelope
#   RemoteReadinessResponse - Typed readiness response wrapper with correlation metadata
#   RemoteModelsResponse - Typed model discovery response wrapper with correlation metadata
#   RemoteAsyncJobResponse - Typed async job snapshot wrapper with correlation metadata
#   RemoteJobResult - Typed async result wrapper with binary payload and response headers
#   RemoteServerClient - Canonical HTTP client used by Telegram-side remote workflows
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.0 - Added dedicated async submit helpers for Telegram remote design and clone job workflows while preserving correlation merging]
# END_CHANGE_SUMMARY

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from core.observability import get_logger
from telegram_bot.observability import log_telegram_event


LOGGER = get_logger(__name__)


# START_CONTRACT: RemoteServerRetryConfig
#   PURPOSE: Configure retry behavior for canonical remote server requests.
#   INPUTS: { max_attempts: int - maximum retry attempts, initial_delay: float - initial backoff delay, max_delay: float - delay ceiling, multiplier: float - exponential factor }
#   OUTPUTS: { RemoteServerRetryConfig - immutable retry policy }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM, M-SERVER
# END_CONTRACT: RemoteServerRetryConfig
@dataclass(frozen=True)
class RemoteServerRetryConfig:
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 10.0
    multiplier: float = 2.0


# START_CONTRACT: RemoteServerCorrelation
#   PURPOSE: Capture request and job correlation metadata returned by the canonical server.
#   INPUTS: { request_id: str | None - current HTTP interaction request id, job_id: str | None - async job id header, submit_request_id: str | None - original submit request correlation id }
#   OUTPUTS: { RemoteServerCorrelation - immutable correlation metadata }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM, M-SERVER
# END_CONTRACT: RemoteServerCorrelation
@dataclass(frozen=True)
class RemoteServerCorrelation:
    request_id: str | None = None
    job_id: str | None = None
    submit_request_id: str | None = None


# START_CONTRACT: RemoteServerErrorEnvelope
#   PURPOSE: Represent the public structured error envelope returned by the canonical server.
#   INPUTS: { code: str - machine-readable error code, message: str - human-readable summary, details: dict[str, Any] - public diagnostics, request_id: str | None - server correlation id, status_code: int | None - HTTP status code, retry_after: str | None - retry hint header }
#   OUTPUTS: { RemoteServerErrorEnvelope - immutable decoded error payload }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM, M-SERVER
# END_CONTRACT: RemoteServerErrorEnvelope
@dataclass(frozen=True)
class RemoteServerErrorEnvelope:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    status_code: int | None = None
    retry_after: str | None = None


# START_CONTRACT: RemoteServerRequestError
#   PURPOSE: Provide a base exception for Telegram-side canonical server failures.
#   INPUTS: { message: str - failure summary, correlation: RemoteServerCorrelation | None - captured correlation metadata }
#   OUTPUTS: { RemoteServerRequestError - exception instance }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM, M-SERVER
# END_CONTRACT: RemoteServerRequestError
class RemoteServerRequestError(Exception):
    def __init__(
        self,
        message: str,
        *,
        correlation: RemoteServerCorrelation | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.correlation = correlation or RemoteServerCorrelation()


# START_CONTRACT: RemoteServerTransportError
#   PURPOSE: Represent transport-level failures when the canonical server cannot be reached reliably.
#   INPUTS: { message: str - failure summary, correlation: RemoteServerCorrelation | None - captured correlation metadata }
#   OUTPUTS: { RemoteServerTransportError - exception instance }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM, M-SERVER
# END_CONTRACT: RemoteServerTransportError
class RemoteServerTransportError(RemoteServerRequestError):
    pass


# START_CONTRACT: RemoteServerAPIError
#   PURPOSE: Represent a controlled canonical server error decoded from the public error envelope.
#   INPUTS: { envelope: RemoteServerErrorEnvelope - decoded public error payload, correlation: RemoteServerCorrelation | None - captured correlation metadata }
#   OUTPUTS: { RemoteServerAPIError - exception instance }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM, M-SERVER
# END_CONTRACT: RemoteServerAPIError
class RemoteServerAPIError(RemoteServerRequestError):
    def __init__(
        self,
        envelope: RemoteServerErrorEnvelope,
        *,
        correlation: RemoteServerCorrelation | None = None,
    ):
        super().__init__(envelope.message, correlation=correlation)
        self.envelope = envelope

    @property
    def code(self) -> str:
        return self.envelope.code

    @property
    def details(self) -> dict[str, Any]:
        return self.envelope.details

    @property
    def request_id(self) -> str | None:
        return self.envelope.request_id


# START_CONTRACT: RemoteReadinessResponse
#   PURPOSE: Wrap the canonical readiness response with captured correlation metadata.
#   INPUTS: { status: str - readiness status value, checks: dict[str, Any] - readiness diagnostics, correlation: RemoteServerCorrelation - captured response correlation }
#   OUTPUTS: { RemoteReadinessResponse - immutable readiness wrapper }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM, M-SERVER
# END_CONTRACT: RemoteReadinessResponse
@dataclass(frozen=True)
class RemoteReadinessResponse:
    status: str
    checks: dict[str, Any]
    correlation: RemoteServerCorrelation


# START_CONTRACT: RemoteModelsResponse
#   PURPOSE: Wrap the canonical model discovery response with captured correlation metadata.
#   INPUTS: { data: list[dict[str, Any]] - discovered model records, correlation: RemoteServerCorrelation - captured response correlation }
#   OUTPUTS: { RemoteModelsResponse - immutable discovery wrapper }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM, M-SERVER
# END_CONTRACT: RemoteModelsResponse
@dataclass(frozen=True)
class RemoteModelsResponse:
    data: list[dict[str, Any]]
    correlation: RemoteServerCorrelation


# START_CONTRACT: RemoteAsyncJobResponse
#   PURPOSE: Wrap a canonical async job snapshot response with captured correlation metadata.
#   INPUTS: { payload: dict[str, Any] - async job snapshot payload, correlation: RemoteServerCorrelation - captured response correlation }
#   OUTPUTS: { RemoteAsyncJobResponse - immutable async snapshot wrapper }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM, M-SERVER
# END_CONTRACT: RemoteAsyncJobResponse
@dataclass(frozen=True)
class RemoteAsyncJobResponse:
    payload: dict[str, Any]
    correlation: RemoteServerCorrelation


# START_CONTRACT: RemoteJobResult
#   PURPOSE: Wrap canonical async result bytes and response metadata for Telegram delivery flows.
#   INPUTS: { audio_bytes: bytes - returned audio payload, content_type: str - response content type, model_id: str | None - audio model header, tts_mode: str | None - TTS mode header, backend_id: str | None - backend header, saved_output_file: str | None - saved output header, correlation: RemoteServerCorrelation - captured response correlation }
#   OUTPUTS: { RemoteJobResult - immutable binary result wrapper }
#   SIDE_EFFECTS: none
#   LINKS: M-TELEGRAM, M-SERVER
# END_CONTRACT: RemoteJobResult
@dataclass(frozen=True)
class RemoteJobResult:
    audio_bytes: bytes
    content_type: str
    model_id: str | None
    tts_mode: str | None
    backend_id: str | None
    saved_output_file: str | None
    correlation: RemoteServerCorrelation


# START_CONTRACT: RemoteServerClient
#   PURPOSE: Execute canonical remote server HTTP calls for readiness, model discovery, and async TTS lifecycle operations.
#   INPUTS: { base_url: str - canonical server base URL, logger: logging.Logger | None - optional logger, retry_config: RemoteServerRetryConfig | None - retry override, http_client: httpx.AsyncClient | None - optional injected async client }
#   OUTPUTS: { RemoteServerClient - configured canonical remote server client }
#   SIDE_EFFECTS: Lazily creates and uses an async HTTP client for network requests.
#   LINKS: M-TELEGRAM, M-SERVER
# END_CONTRACT: RemoteServerClient
class RemoteServerClient:
    REQUEST_TIMEOUT = 30.0

    def __init__(
        self,
        base_url: str,
        logger: logging.Logger | None = None,
        retry_config: RemoteServerRetryConfig | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._logger = logger or LOGGER
        self._retry_config = retry_config or RemoteServerRetryConfig()
        self._client = http_client
        self._owns_client = http_client is None

    @property
    def base_url(self) -> str:
        return self._base_url

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.REQUEST_TIMEOUT),
                follow_redirects=True,
            )
        return self._client

    # START_CONTRACT: close
    #   PURPOSE: Close the underlying async HTTP client when this instance created it.
    #   INPUTS: {}
    #   OUTPUTS: { None - no return value }
    #   SIDE_EFFECTS: Closes network resources held by the owned HTTP client.
    #   LINKS: M-TELEGRAM, M-SERVER
    # END_CONTRACT: close
    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _build_url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _calculate_delay(self, retry_count: int) -> float:
        delay = self._retry_config.initial_delay * (
            self._retry_config.multiplier**retry_count
        )
        return min(delay, self._retry_config.max_delay)

    def _build_request_headers(
        self,
        *,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        if request_id:
            headers["x-request-id"] = request_id
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def _extract_correlation(self, response: httpx.Response) -> RemoteServerCorrelation:
        return RemoteServerCorrelation(
            request_id=response.headers.get("x-request-id"),
            job_id=response.headers.get("x-job-id"),
            submit_request_id=response.headers.get("x-submit-request-id"),
        )

    def _merge_correlation(
        self,
        primary: RemoteServerCorrelation,
        fallback: RemoteServerCorrelation | None = None,
    ) -> RemoteServerCorrelation:
        fallback = fallback or RemoteServerCorrelation()
        return RemoteServerCorrelation(
            request_id=primary.request_id or fallback.request_id,
            job_id=primary.job_id or fallback.job_id,
            submit_request_id=primary.submit_request_id or fallback.submit_request_id,
        )

    def _decode_error_envelope(
        self,
        response: httpx.Response,
    ) -> RemoteServerErrorEnvelope | None:
        try:
            payload = response.json()
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        code = payload.get("code")
        message = payload.get("message")
        request_id = payload.get("request_id")
        if not isinstance(code, str) or not isinstance(message, str):
            return None
        details = payload.get("details")
        decoded_details = details if isinstance(details, dict) else {}
        return RemoteServerErrorEnvelope(
            code=code,
            message=message,
            details=decoded_details,
            request_id=request_id if isinstance(request_id, str) else None,
            status_code=response.status_code,
            retry_after=response.headers.get("retry-after"),
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        retry_count: int = 0,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        known_correlation: RemoteServerCorrelation | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        # START_BLOCK_PREPARE_REMOTE_REQUEST
        client = await self._get_client()
        url = self._build_url(path)
        headers = self._build_request_headers(
            request_id=request_id,
            idempotency_key=idempotency_key,
        )
        request_headers = dict(kwargs.pop("headers", {}))
        request_headers.update(headers)
        log_telegram_event(
            self._logger,
            level=logging.DEBUG,
            event="[TelegramRemoteServerClient][_request][BLOCK_PREPARE_REMOTE_REQUEST]",
            message="Telegram remote server request",
            method=method,
            path=path,
            retry_count=retry_count,
        )
        # END_BLOCK_PREPARE_REMOTE_REQUEST

        try:
            # START_BLOCK_EXECUTE_REMOTE_REQUEST
            response = await client.request(
                method,
                url,
                headers=request_headers,
                **kwargs,
            )

            if response.status_code == 429 or response.status_code >= 500:
                if retry_count < self._retry_config.max_attempts:
                    retry_after = response.headers.get("retry-after")
                    wait_time = (
                        float(retry_after)
                        if retry_after is not None
                        else self._calculate_delay(retry_count)
                    )
                    await asyncio.sleep(wait_time)
                    return await self._request(
                        method,
                        path,
                        retry_count=retry_count + 1,
                        request_id=request_id,
                        idempotency_key=idempotency_key,
                        known_correlation=known_correlation,
                        headers=request_headers,
                        **kwargs,
                    )

            if response.status_code >= 400:
                response_correlation = self._extract_correlation(response)
                envelope = self._decode_error_envelope(response)
                payload_correlation = RemoteServerCorrelation(
                    request_id=(envelope.request_id if envelope is not None else None),
                    job_id=(
                        str(envelope.details.get("job_id"))
                        if envelope is not None
                        and isinstance(envelope.details.get("job_id"), str)
                        else None
                    ),
                    submit_request_id=(
                        str(envelope.details.get("submit_request_id"))
                        if envelope is not None
                        and isinstance(envelope.details.get("submit_request_id"), str)
                        else None
                    ),
                )
                correlation = self._merge_correlation(
                    response_correlation,
                    self._merge_correlation(payload_correlation, known_correlation),
                )
                if envelope is not None:
                    raise RemoteServerAPIError(envelope, correlation=correlation)
                raise RemoteServerRequestError(
                    f"Remote server request failed with status {response.status_code}",
                    correlation=correlation,
                )

            return response
            # END_BLOCK_EXECUTE_REMOTE_REQUEST
        except RemoteServerRequestError:
            raise
        except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
            # START_BLOCK_HANDLE_REMOTE_TIMEOUT
            if retry_count < self._retry_config.max_attempts:
                await asyncio.sleep(self._calculate_delay(retry_count))
                return await self._request(
                    method,
                    path,
                    retry_count=retry_count + 1,
                    request_id=request_id,
                    idempotency_key=idempotency_key,
                    known_correlation=known_correlation,
                    headers=request_headers,
                    **kwargs,
                )
            raise RemoteServerTransportError(
                f"Remote server request timed out after {retry_count + 1} attempts"
            ) from exc
            # END_BLOCK_HANDLE_REMOTE_TIMEOUT
        except httpx.ConnectError as exc:
            # START_BLOCK_HANDLE_REMOTE_CONNECTION_ERROR
            if retry_count < self._retry_config.max_attempts:
                await asyncio.sleep(self._calculate_delay(retry_count))
                return await self._request(
                    method,
                    path,
                    retry_count=retry_count + 1,
                    request_id=request_id,
                    idempotency_key=idempotency_key,
                    known_correlation=known_correlation,
                    headers=request_headers,
                    **kwargs,
                )
            raise RemoteServerTransportError(
                f"Remote server connection failed after {retry_count + 1} attempts"
            ) from exc
            # END_BLOCK_HANDLE_REMOTE_CONNECTION_ERROR

    # START_CONTRACT: get_readiness
    #   PURPOSE: Fetch the canonical server readiness report for Telegram capability checks.
    #   INPUTS: { request_id: str | None - optional correlation id forwarded to the server }
    #   OUTPUTS: { RemoteReadinessResponse - readiness payload and response correlation }
    #   SIDE_EFFECTS: Performs a canonical server HTTP request.
    #   LINKS: M-TELEGRAM, M-SERVER
    # END_CONTRACT: get_readiness
    async def get_readiness(self, request_id: str | None = None) -> RemoteReadinessResponse:
        response = await self._request(
            "GET",
            "/health/ready",
            request_id=request_id,
        )
        payload = response.json()
        return RemoteReadinessResponse(
            status=str(payload.get("status", "unknown")),
            checks=payload.get("checks", {}),
            correlation=self._extract_correlation(response),
        )

    # START_CONTRACT: list_models
    #   PURPOSE: Fetch canonical model discovery records from the remote server.
    #   INPUTS: { request_id: str | None - optional correlation id forwarded to the server }
    #   OUTPUTS: { RemoteModelsResponse - model discovery payload and response correlation }
    #   SIDE_EFFECTS: Performs a canonical server HTTP request.
    #   LINKS: M-TELEGRAM, M-SERVER
    # END_CONTRACT: list_models
    async def list_models(self, request_id: str | None = None) -> RemoteModelsResponse:
        response = await self._request(
            "GET",
            "/api/v1/models",
            request_id=request_id,
        )
        payload = response.json()
        data = payload.get("data", [])
        return RemoteModelsResponse(
            data=data if isinstance(data, list) else [],
            correlation=self._extract_correlation(response),
        )

    # START_CONTRACT: submit_speech_job
    #   PURPOSE: Submit an OpenAI-compatible async speech request to the canonical server.
    #   INPUTS: { payload: dict[str, Any] - canonical OpenAI-style async speech request body, request_id: str | None - optional forwarded correlation id, idempotency_key: str | None - optional idempotency key }
    #   OUTPUTS: { RemoteAsyncJobResponse - async job snapshot and response correlation }
    #   SIDE_EFFECTS: Performs a canonical server HTTP request.
    #   LINKS: M-TELEGRAM, M-SERVER
    # END_CONTRACT: submit_speech_job
    async def submit_speech_job(
        self,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RemoteAsyncJobResponse:
        response = await self._request(
            "POST",
            "/v1/audio/speech/jobs",
            request_id=request_id,
            idempotency_key=idempotency_key,
            json=payload,
        )
        return RemoteAsyncJobResponse(
            payload=response.json(),
            correlation=self._extract_correlation(response),
        )

    # START_CONTRACT: submit_design_job
    #   PURPOSE: Submit a voice design async job request to the canonical server.
    #   INPUTS: { payload: dict[str, Any] - canonical voice design request body, request_id: str | None - optional forwarded correlation id, idempotency_key: str | None - optional idempotency key }
    #   OUTPUTS: { RemoteAsyncJobResponse - async job snapshot and response correlation }
    #   SIDE_EFFECTS: Performs a canonical server HTTP request.
    #   LINKS: M-TELEGRAM, M-SERVER
    # END_CONTRACT: submit_design_job
    async def submit_design_job(
        self,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RemoteAsyncJobResponse:
        response = await self._request(
            "POST",
            "/api/v1/tts/design/jobs",
            request_id=request_id,
            idempotency_key=idempotency_key,
            json=payload,
        )
        return RemoteAsyncJobResponse(
            payload=response.json(),
            correlation=self._extract_correlation(response),
        )

    # START_CONTRACT: submit_clone_job
    #   PURPOSE: Submit a voice clone async job request with multipart reference audio to the canonical server.
    #   INPUTS: { text: str - synthesis text, ref_audio_bytes: bytes - reference audio payload, ref_audio_filename: str - uploaded filename, ref_audio_content_type: str - uploaded MIME type, ref_text: str | None - optional reference transcript, language: str - requested language code, model: str | None - optional model override, request_id: str | None - optional forwarded correlation id, idempotency_key: str | None - optional idempotency key }
    #   OUTPUTS: { RemoteAsyncJobResponse - async job snapshot and response correlation }
    #   SIDE_EFFECTS: Performs a canonical server HTTP multipart request.
    #   LINKS: M-TELEGRAM, M-SERVER
    # END_CONTRACT: submit_clone_job
    async def submit_clone_job(
        self,
        *,
        text: str,
        ref_audio_bytes: bytes,
        ref_audio_filename: str,
        ref_audio_content_type: str,
        ref_text: str | None = None,
        language: str = "auto",
        model: str | None = None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RemoteAsyncJobResponse:
        form_fields: dict[str, str] = {
            "text": text,
            "language": language,
        }
        if ref_text is not None:
            form_fields["ref_text"] = ref_text
        if model is not None:
            form_fields["model"] = model

        response = await self._request(
            "POST",
            "/api/v1/tts/clone/jobs",
            request_id=request_id,
            idempotency_key=idempotency_key,
            data=form_fields,
            files={
                "ref_audio": (
                    ref_audio_filename,
                    ref_audio_bytes,
                    ref_audio_content_type,
                )
            },
        )
        return RemoteAsyncJobResponse(
            payload=response.json(),
            correlation=self._extract_correlation(response),
        )

    # START_CONTRACT: get_job_status
    #   PURPOSE: Retrieve the canonical async job snapshot for a previously submitted remote job.
    #   INPUTS: { job_id: str - async job identifier, request_id: str | None - optional forwarded correlation id }
    #   OUTPUTS: { RemoteAsyncJobResponse - async job snapshot and response correlation }
    #   SIDE_EFFECTS: Performs a canonical server HTTP request.
    #   LINKS: M-TELEGRAM, M-SERVER
    # END_CONTRACT: get_job_status
    async def get_job_status(
        self,
        job_id: str,
        *,
        request_id: str | None = None,
        submit_request_id: str | None = None,
    ) -> RemoteAsyncJobResponse:
        response = await self._request(
            "GET",
            f"/api/v1/tts/jobs/{job_id}",
            request_id=request_id,
            known_correlation=RemoteServerCorrelation(
                job_id=job_id,
                submit_request_id=submit_request_id,
            ),
        )
        return RemoteAsyncJobResponse(
            payload=response.json(),
            correlation=self._merge_correlation(
                self._extract_correlation(response),
                RemoteServerCorrelation(
                    job_id=job_id,
                    submit_request_id=submit_request_id,
                ),
            ),
        )

    # START_CONTRACT: get_job_result
    #   PURPOSE: Retrieve the binary audio result for a succeeded remote async job.
    #   INPUTS: { job_id: str - async job identifier, request_id: str | None - optional forwarded correlation id }
    #   OUTPUTS: { RemoteJobResult - binary audio payload with response metadata }
    #   SIDE_EFFECTS: Performs a canonical server HTTP request.
    #   LINKS: M-TELEGRAM, M-SERVER
    # END_CONTRACT: get_job_result
    async def get_job_result(
        self,
        job_id: str,
        *,
        request_id: str | None = None,
        submit_request_id: str | None = None,
    ) -> RemoteJobResult:
        response = await self._request(
            "GET",
            f"/api/v1/tts/jobs/{job_id}/result",
            request_id=request_id,
            known_correlation=RemoteServerCorrelation(
                job_id=job_id,
                submit_request_id=submit_request_id,
            ),
        )
        return RemoteJobResult(
            audio_bytes=response.content,
            content_type=response.headers.get("content-type", "application/octet-stream"),
            model_id=response.headers.get("x-model-id"),
            tts_mode=response.headers.get("x-tts-mode"),
            backend_id=response.headers.get("x-backend-id"),
            saved_output_file=response.headers.get("x-saved-output-file"),
            correlation=self._merge_correlation(
                self._extract_correlation(response),
                RemoteServerCorrelation(
                    job_id=job_id,
                    submit_request_id=submit_request_id,
                ),
            ),
        )


__all__ = [
    "LOGGER",
    "RemoteServerRetryConfig",
    "RemoteServerCorrelation",
    "RemoteServerErrorEnvelope",
    "RemoteServerRequestError",
    "RemoteServerTransportError",
    "RemoteServerAPIError",
    "RemoteReadinessResponse",
    "RemoteModelsResponse",
    "RemoteAsyncJobResponse",
    "RemoteJobResult",
    "RemoteServerClient",
]

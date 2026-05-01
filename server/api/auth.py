# FILE: server/api/auth.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Implement HTTP authentication middleware and dependency injection.
#   SCOPE: Bearer token validation, principal resolution
#   DEPENDS: M-CONFIG, M-ERRORS
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   AUTH_MODE_OFF - Auth mode constant for disabled authentication
#   AUTH_MODE_STATIC_BEARER - Auth mode constant for static bearer authentication
#   LOCAL_DEFAULT_PRINCIPAL_ID - Stable principal id used for local unauthenticated requests
#   LOCAL_DEFAULT_SUBJECT_TYPE - Subject type label for local unauthenticated requests
#   STATIC_TOKEN_SUBJECT_TYPE - Subject type label for static bearer token principals
#   RequestPrincipal - Immutable request principal attached to authenticated requests
#   build_local_default_principal - Build the fallback local default principal
#   is_public_route - Detect routes that remain public regardless of auth mode
#   is_protected_route - Detect routes that require authentication under policy
#   resolve_request_principal - Resolve the effective principal for an incoming request
#   ensure_job_owner_access - Enforce owner-only access to async job resources
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from fastapi import Request

from core.errors import ForbiddenError, UnauthorizedError
from server.bootstrap import ServerSettings

AUTH_MODE_OFF = "off"
AUTH_MODE_STATIC_BEARER = "static_bearer"
LOCAL_DEFAULT_PRINCIPAL_ID = "local-default"
LOCAL_DEFAULT_SUBJECT_TYPE = "local_default"
STATIC_TOKEN_SUBJECT_TYPE = "static_token"
PROTECTED_ROUTE_PREFIXES = (
    "/api/v1/models",
    "/api/v1/tts",
    "/v1/audio/speech",
    "/health/ready",
)
PUBLIC_ROUTE_PATHS = {"/health/live"}


@dataclass(frozen=True)
# START_CONTRACT: RequestPrincipal
#   PURPOSE: Represent the authenticated or local principal attached to an HTTP request.
#   INPUTS: { principal_id: str - stable principal identifier, subject_type: str - principal kind, authenticated: bool - whether credentials were validated, credential_id: Optional[str] - optional credential fingerprint }
#   OUTPUTS: { RequestPrincipal - immutable request principal record }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: RequestPrincipal
class RequestPrincipal:
    principal_id: str
    subject_type: str
    authenticated: bool
    credential_id: str | None = None


# START_CONTRACT: build_local_default_principal
#   PURPOSE: Build the fallback local principal used when authentication is disabled or not required.
#   INPUTS: { none: None - no explicit inputs }
#   OUTPUTS: { RequestPrincipal - unauthenticated local default principal }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: build_local_default_principal
def build_local_default_principal() -> RequestPrincipal:
    return RequestPrincipal(
        principal_id=LOCAL_DEFAULT_PRINCIPAL_ID,
        subject_type=LOCAL_DEFAULT_SUBJECT_TYPE,
        authenticated=False,
        credential_id=None,
    )


# START_CONTRACT: is_public_route
#   PURPOSE: Determine whether a request path is explicitly public.
#   INPUTS: { path: str - request path to evaluate }
#   OUTPUTS: { bool - true when the path is treated as public }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: is_public_route
def is_public_route(path: str) -> bool:
    return path in PUBLIC_ROUTE_PATHS


# START_CONTRACT: is_protected_route
#   PURPOSE: Determine whether a request path requires authentication under server policy.
#   INPUTS: { path: str - request path to evaluate }
#   OUTPUTS: { bool - true when the path matches protected route prefixes }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: is_protected_route
def is_protected_route(path: str) -> bool:
    if is_public_route(path):
        return False
    return any(path.startswith(prefix) for prefix in PROTECTED_ROUTE_PREFIXES)


# START_CONTRACT: resolve_request_principal
#   PURPOSE: Resolve the effective request principal according to configured authentication mode and route policy.
#   INPUTS: { request: Request - incoming request with app settings state }
#   OUTPUTS: { RequestPrincipal - resolved principal for authorization and ownership checks }
#   SIDE_EFFECTS: Reads server settings and may raise authentication errors for invalid configuration or credentials
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: resolve_request_principal
def resolve_request_principal(request: Request) -> RequestPrincipal:
    settings: ServerSettings = request.app.state.settings
    if settings.auth_mode == AUTH_MODE_OFF:
        return build_local_default_principal()
    if settings.auth_mode != AUTH_MODE_STATIC_BEARER:
        raise UnauthorizedError(reason=f"Unsupported auth mode: {settings.auth_mode}")
    if not is_protected_route(request.url.path):
        return build_local_default_principal()
    return authenticate_static_bearer(request, settings)


# START_CONTRACT: authenticate_static_bearer
#   PURPOSE: Validate a static bearer token request and produce an authenticated principal.
#   INPUTS: { request: Request - incoming request carrying authorization header, settings: ServerSettings - server auth configuration }
#   OUTPUTS: { RequestPrincipal - authenticated principal derived from the validated token }
#   SIDE_EFFECTS: Reads request headers and raises unauthorized errors when credentials are missing or invalid
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: authenticate_static_bearer
def authenticate_static_bearer(request: Request, settings: ServerSettings) -> RequestPrincipal:
    token = extract_bearer_token(request)
    configured_token = settings.auth_static_bearer_token
    if configured_token is None:
        raise UnauthorizedError(reason="Static bearer auth is enabled but no token is configured")
    if token is None:
        raise UnauthorizedError(reason="Missing bearer token")
    if token != configured_token:
        raise UnauthorizedError(reason="Bearer token is invalid")
    credential_id = settings.auth_static_bearer_credential_id or build_credential_id(token)
    principal_id = settings.auth_static_bearer_principal_id or credential_id
    return RequestPrincipal(
        principal_id=principal_id,
        subject_type=STATIC_TOKEN_SUBJECT_TYPE,
        authenticated=True,
        credential_id=credential_id,
    )


# START_CONTRACT: extract_bearer_token
#   PURPOSE: Extract and validate the bearer token value from the Authorization header.
#   INPUTS: { request: Request - incoming request carrying HTTP headers }
#   OUTPUTS: { Optional[str] - stripped bearer token when present, otherwise none }
#   SIDE_EFFECTS: Raises unauthorized errors for malformed authorization schemes
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: extract_bearer_token
def extract_bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization")
    if authorization is None:
        return None
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credentials.strip():
        raise UnauthorizedError(reason="Authorization header must use Bearer authentication")
    return credentials.strip()


# START_CONTRACT: ensure_job_owner_access
#   PURPOSE: Enforce that the current request principal owns the addressed async job.
#   INPUTS: { request: Request - incoming request containing resolved principal, owner_principal_id: str - expected job owner principal id }
#   OUTPUTS: { None - returns when access is permitted }
#   SIDE_EFFECTS: Raises forbidden errors when the request principal does not own the job
#   LINKS: M-SERVER, M-ERRORS
# END_CONTRACT: ensure_job_owner_access
def ensure_job_owner_access(request: Request, *, owner_principal_id: str) -> None:
    principal: RequestPrincipal = request.state.principal
    if principal.principal_id != owner_principal_id:
        raise ForbiddenError(
            reason="Authenticated principal is not allowed to access this job",
            details={"owner_principal_id": owner_principal_id},
        )


# START_CONTRACT: build_credential_id
#   PURPOSE: Derive a stable redacted credential identifier from a static bearer token.
#   INPUTS: { token: str - validated bearer token value }
#   OUTPUTS: { str - stable credential identifier derived from the token hash }
#   SIDE_EFFECTS: none
#   LINKS: M-SERVER
# END_CONTRACT: build_credential_id
def build_credential_id(token: str) -> str:
    return f"static-bearer-{hashlib.sha256(token.encode('utf-8')).hexdigest()[:16]}"


__all__ = [
    "AUTH_MODE_OFF",
    "AUTH_MODE_STATIC_BEARER",
    "LOCAL_DEFAULT_PRINCIPAL_ID",
    "LOCAL_DEFAULT_SUBJECT_TYPE",
    "RequestPrincipal",
    "STATIC_TOKEN_SUBJECT_TYPE",
    "build_local_default_principal",
    "ensure_job_owner_access",
    "is_protected_route",
    "is_public_route",
    "resolve_request_principal",
]

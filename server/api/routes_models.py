# FILE: server/api/routes_models.py
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define model discovery HTTP endpoints.
#   SCOPE: GET /api/v1/models
#   DEPENDS: M-MODEL-REGISTRY
#   LINKS: M-SERVER
#   ROLE: RUNTIME
#   MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   register_models_routes - Register model discovery routes on the FastAPI app
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - GRACE integration: added MODULE_CONTRACT, MODULE_MAP, and function contracts]
# END_CHANGE_SUMMARY

from __future__ import annotations

from fastapi import FastAPI, Request

from core.observability import log_event, operation_scope
from server.api.policies import enforce_control_plane_admission
from server.schemas.audio import ModelsResponse


# START_CONTRACT: register_models_routes
#   PURPOSE: Register model discovery HTTP routes on the FastAPI application.
#   INPUTS: { app: FastAPI - application to attach routes to, logger: Any - structured logger used by endpoint handlers }
#   OUTPUTS: { None - routes are attached in place }
#   SIDE_EFFECTS: Mutates FastAPI routing table by registering model discovery endpoints
#   LINKS: M-SERVER, M-MODEL-REGISTRY
# END_CONTRACT: register_models_routes
def register_models_routes(app: FastAPI, logger) -> None:
    @app.get("/api/v1/models", response_model=ModelsResponse, tags=["models"])
    # START_CONTRACT: list_models
    #   PURPOSE: Return the list of models currently registered for the API.
    #   INPUTS: { request: Request - incoming request used to access registry state }
    #   OUTPUTS: { ModelsResponse - model discovery payload }
    #   SIDE_EFFECTS: Consumes control-plane admission checks and emits model listing logs
    #   LINKS: M-SERVER, M-MODEL-REGISTRY
    # END_CONTRACT: list_models
    async def list_models(request: Request) -> ModelsResponse:
        with operation_scope("server.list_models"):
            await enforce_control_plane_admission(request)
            models = request.app.state.registry.list_models()
            log_event(
                logger,
                level=20,
                event="[RoutesModels][list_models][LIST_MODELS]",
                message="Model listing completed",
                model_count=len(models),
            )
            return ModelsResponse(data=models)


__all__ = [
    "register_models_routes",
]

from __future__ import annotations

from fastapi import FastAPI, Request

from core.observability import log_event, operation_scope
from server.schemas.audio import ModelsResponse



def register_models_routes(app: FastAPI, logger) -> None:
    @app.get("/api/v1/models", response_model=ModelsResponse, tags=["models"])
    async def list_models(request: Request) -> ModelsResponse:
        with operation_scope("server.list_models"):
            models = request.app.state.registry.list_models()
            log_event(
                logger,
                level=20,
                event="models.list.completed",
                message="Model listing completed",
                model_count=len(models),
            )
            return ModelsResponse(data=models)

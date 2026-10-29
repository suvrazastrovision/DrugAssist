"""DrugAssist: authenticated FDA retrieval and optional cited answer writing."""

import sqlite3
from contextlib import asynccontextmanager
from threading import RLock
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas import AnswerRequest, AnswerResponse, HealthResponse, SearchRequest, SearchResponse
from src.auth.routes import get_current_user
from src.auth.routes import router as auth_router
from src.auth.service import AuthError, build_auth_service
from src.services.answer_service import build_answer_service
from src.services.search_service import ServiceUnavailable, build_search_service


def get_service(request: Request):
    """Return the lazily initialized search service, sanitizing startup failures."""
    try:
        with request.app.state.lock:
            if request.app.state.service is None:
                request.app.state.service = request.app.state.service_factory()
            return request.app.state.service
    except Exception:  # noqa: BLE001 -- Sanitize failures from third-party providers and Chroma.
        raise ServiceUnavailable() from None


def create_app(
    service_factory=build_search_service,
    auth_factory=build_auth_service,
    answer_factory=build_answer_service,
):
    """Build the API with injectable search, authentication, and answer services."""

    @asynccontextmanager
    async def lifespan(app):
        """Close the search service when the application shuts down."""
        try:
            yield
        finally:
            if app.state.service is not None:
                app.state.service.close()

    app = FastAPI(title="DrugAssist", version="0.1.0", lifespan=lifespan)
    app.state.service_factory, app.state.service = service_factory, None
    app.state.auth_factory, app.state.auth = auth_factory, None
    app.state.answer_factory, app.state.answer = answer_factory, None
    app.state.lock = RLock()
    app.include_router(auth_router)

    @app.middleware("http")
    async def private_responses(request, call_next):
        """Prevent caching of account, search, and answer responses."""
        response = await call_next(request)
        if request.url.path.startswith("/auth/") or request.url.path in {"/search", "/answer"}:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(AuthError)
    async def auth_error(request, exc):
        """Convert an authentication error into a public HTTP response."""
        headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else {}
        if exc.retry_after is not None:
            headers["Retry-After"] = str(exc.retry_after)
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail}, headers=headers)

    @app.exception_handler(sqlite3.Error)
    async def auth_unavailable(request, exc):
        """Report account storage failures without exposing database details."""
        return JSONResponse(
            status_code=503, content={"detail": "Account service is unavailable. Please try again later."}
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, exc):
        """Return validation errors without echoing submitted credentials."""
        # Pydantic's default response includes submitted values, including passwords.
        detail = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.exception_handler(ServiceUnavailable)
    async def unavailable(request, exc):
        """Report search failures using the endpoint's public response schema."""
        if request.url.path == "/health":
            body = HealthResponse(
                status="unavailable", chromadb="unavailable", indexed_documents=None, indexed_chunks=None
            ).model_dump()
        else:
            body = {"detail": "Search service is unavailable. Please try again later."}
        return JSONResponse(status_code=503, content=body)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        """Return readiness and index counts from the search service."""
        return get_service(request).health()

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request, user: Annotated[dict, Depends(get_current_user)]):
        """Retrieve source passages for an authenticated user's query."""
        return get_service(request).search(body.query, body.k)

    @app.post("/answer", response_model=AnswerResponse)
    def answer(body: AnswerRequest, request: Request, user: Annotated[dict, Depends(get_current_user)]):
        """Retrieve evidence and write a cited answer for an authenticated user."""
        service = get_service(request)
        with request.app.state.lock:
            if request.app.state.answer is None:
                request.app.state.answer = request.app.state.answer_factory(service)
            writer = request.app.state.answer
        return writer.answer(body.query, body.k, body.audience)

    return app


app = create_app()

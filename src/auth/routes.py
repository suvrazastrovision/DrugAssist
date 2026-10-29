"""Thin HTTP routes and a reusable authentication dependency."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.schemas import Credentials, LoginResponse, UserResponse
from src.auth.service import AuthError, AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])
bearer = HTTPBearer(auto_error=False)
Token = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]


def get_auth(request: Request) -> AuthService:
    """Return the lazily initialized authentication service for this app."""
    with request.app.state.lock:
        if request.app.state.auth is None:
            request.app.state.auth = request.app.state.auth_factory()
        return request.app.state.auth


def get_current_user(request: Request, credentials: Token) -> dict:
    """Resolve bearer credentials to a user or raise an authentication error."""
    if credentials is None:
        raise AuthError(401, "Please log in again.")
    return get_auth(request).current_user(credentials.credentials)


def limit_attempts(request: Request, auth: AuthService) -> None:
    """Apply the authentication attempt limit to the requesting client."""
    auth.limit_attempts(request.client.host if request.client else "unknown")


@router.post("/signup", response_model=UserResponse, status_code=201)
def signup(body: Credentials, request: Request) -> dict:
    """Create an account after checking the client's attempt limit."""
    auth = get_auth(request)
    limit_attempts(request, auth)
    return auth.signup(str(body.email), body.password)


@router.post("/login", response_model=LoginResponse)
def login(body: Credentials, request: Request) -> dict:
    """Verify credentials and issue a session after checking the attempt limit."""
    auth = get_auth(request)
    limit_attempts(request, auth)
    return auth.login(str(body.email), body.password)


@router.get("/me", response_model=UserResponse)
def me(user: Annotated[dict, Depends(get_current_user)]) -> dict:
    """Return the authenticated user's public account details."""
    return user


@router.post("/logout", status_code=204)
def logout(
    request: Request, credentials: Token, user: Annotated[dict, Depends(get_current_user)]
) -> Response:
    """Revoke the current bearer token and return an empty response."""
    get_auth(request).logout(credentials.credentials)
    return Response(status_code=204)

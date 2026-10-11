
import hmac
import os

from fastapi import HTTPException


def authenticate_project(authorization: str | None) -> str:
    """Validate a Bearer API key and return its project ID."""

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, separator, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not separator or not token.strip():
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Map configured API keys to their project IDs.
    project_keys = {
        os.getenv("PROJECT_A_API_KEY", ""): "project-a",
        os.getenv("PROJECT_B_API_KEY", ""): "project-b",
    }

    # Ignore empty environment values so an unset key can never authenticate.
    configured_keys = {
        key: project_id
        for key, project_id in project_keys.items()
        if key
    }

    if not configured_keys:
        raise HTTPException(
            status_code=500,
            detail="Authentication is not configured",
        )

    token = token.strip()

    for api_key, project_id in configured_keys.items():
        if hmac.compare_digest(token, api_key):
            return project_id

    raise HTTPException(
        status_code=401,
        detail="Invalid project API key",
        headers={"WWW-Authenticate": "Bearer"},
    )

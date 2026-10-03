from fastapi import Header, HTTPException


PROJECT_KEYS = {
    "sk-project-a": "project-a",
    "sk-project-b": "project-b",
}


def authenticate_project(
    authorization: str | None,
) -> str:
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
        )

    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header",
        )

    project_id = PROJECT_KEYS.get(token)

    if project_id is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid project key",
        )

    return project_id

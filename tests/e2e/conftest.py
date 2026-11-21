from __future__ import annotations

import socket
import os

import httpx
import pytest


@pytest.fixture(scope="session", autouse=True)
def require_live_stack():
    """Keep the live suite out of ordinary unit runs when its stack is absent."""
    gateway = os.getenv("E2E_GATEWAY_URL", "http://127.0.0.1:28000")

    try:
        with httpx.Client(timeout=1) as client:
            response = client.get(f"{gateway}/health")
        with socket.create_connection(("127.0.0.1", 26379), timeout=1):
            pass
    except (httpx.HTTPError, OSError):
        pytest.skip("start docker-compose.e2e.yml to run the live integration suite")
    if response.status_code != 200:
        pytest.skip("start docker-compose.e2e.yml to run the live integration suite")

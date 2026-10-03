import json
import time
from pathlib import Path


LEDGER_PATH = Path("evidence/requests.jsonl")


def record_request(
    *,
    request_id: str,
    project_id: str,
    requested_model: str,
    selected_model: str,
    status: str,
    latency_ms: float,
    attempts: int,
    failure_class: str | None = None,
    usage: dict | None = None,
):
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": time.time(),
        "request_id": request_id,
        "project_id": project_id,
        "requested_model": requested_model,
        "selected_model": selected_model,
        "status": status,
        "latency_ms": round(latency_ms, 2),
        "attempts": attempts,
        "failure_class": failure_class,
        "usage": usage,
    }

    with LEDGER_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")

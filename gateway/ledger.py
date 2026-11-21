import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path


LEDGER_PATH = Path("evidence/requests.jsonl")
_LEDGER_LOCK = threading.Lock()


def _previous_signature(fd: int) -> str | None:
    size = os.lseek(fd, 0, os.SEEK_END)
    if size == 0:
        return None
    start = max(0, size - 8192)
    os.lseek(fd, start, os.SEEK_SET)
    tail = os.read(fd, size - start)
    lines = [line for line in tail.splitlines() if line]
    if not lines:
        return None
    try:
        return json.loads(lines[-1]).get("signature")
    except (json.JSONDecodeError, AttributeError):
        return None


def record_request(
    *,
    request_id: str,
    requested_model: str,
    selected_model: str,
    status: str,
    latency_ms: float,
    attempts: int,
    failure_class: str | None = None,
    usage: dict | None = None,
):
    LEDGER_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(LEDGER_PATH.parent, 0o700)

    record = {
        "timestamp": time.time(),
        "request_id": request_id,
        "requested_model": requested_model,
        "selected_model": selected_model,
        "status": status,
        "latency_ms": round(latency_ms, 2),
        "attempts": attempts,
        "failure_class": failure_class,
        "usage": usage,
    }

    audit_key = os.getenv("GATEWAY_AUDIT_HMAC_KEY") or os.getenv("LITELLM_SALT_KEY", "")
    with _LEDGER_LOCK:
        fd = os.open(LEDGER_PATH, os.O_CREAT | os.O_APPEND | os.O_RDWR, 0o600)
        try:
            os.fchmod(fd, 0o600)
            record["previous_signature"] = _previous_signature(fd)
            if audit_key:
                canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
                record["signature"] = hmac.new(
                    audit_key.encode(), canonical.encode(), hashlib.sha256
                ).hexdigest()
            os.lseek(fd, 0, os.SEEK_END)
            os.write(fd, (json.dumps(record, separators=(",", ":")) + "\n").encode())
        finally:
            os.close(fd)

#!/usr/bin/env python3
"""Rotate the LiteLLM master key and gateway shared API key in .env."""

from __future__ import annotations

import os
import secrets
import stat
import tempfile
from pathlib import Path


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
ROTATED_KEYS = {
    "LITELLM_MASTER_KEY": "sk-" + secrets.token_urlsafe(48),
    "GATEWAY_API_KEY": secrets.token_urlsafe(48),
}


def main() -> None:
    if not ENV_FILE.is_file():
        raise SystemExit(".env not found; copy .env.example first")
    if ENV_FILE.is_symlink():
        raise SystemExit("Refusing to replace a symlinked .env file")

    old_mode = stat.S_IMODE(ENV_FILE.stat().st_mode)
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    found = set()
    updated = []
    for line in lines:
        key, separator, _value = line.partition("=")
        if separator and key in ROTATED_KEYS:
            updated.append(f"{key}={ROTATED_KEYS[key]}")
            found.add(key)
        else:
            updated.append(line)
    for key, value in ROTATED_KEYS.items():
        if key not in found:
            updated.append(f"{key}={value}")

    fd, temporary_name = tempfile.mkstemp(prefix=".env.", dir=ENV_FILE.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temporary:
            temporary.write("\n".join(updated) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, old_mode & 0o600)
        os.replace(temporary_name, ENV_FILE)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    print("Rotated LITELLM_MASTER_KEY and GATEWAY_API_KEY in .env (values not printed).")
    print("Recreate litellm, semantic-router, and gateway, then update client credentials.")
    print("LITELLM_SALT_KEY was left unchanged to preserve encrypted database credentials.")


if __name__ == "__main__":
    main()

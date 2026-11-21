#!/usr/bin/env python3
"""Verify HMAC signatures and chain links in the gateway JSONL audit ledger."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", nargs="?", type=Path, default=Path("evidence/requests.jsonl"))
    args = parser.parse_args()
    key = os.getenv("GATEWAY_AUDIT_HMAC_KEY") or os.getenv("LITELLM_SALT_KEY", "")
    if not key:
        parser.error("set GATEWAY_AUDIT_HMAC_KEY (or LITELLM_SALT_KEY) to verify signatures")

    previous_signature = None
    verified = 0
    legacy_unsigned = 0
    with args.ledger.open(encoding="utf-8") as ledger:
        for line_number, line in enumerate(ledger, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"invalid JSON at line {line_number}: {error}") from error
            signature = record.pop("signature", None)
            if signature is None:
                if verified:
                    raise SystemExit(f"unsigned entry breaks the signed chain at line {line_number}")
                legacy_unsigned += 1
                continue
            if record.get("previous_signature") != previous_signature:
                raise SystemExit(f"chain link mismatch at line {line_number}")
            canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
            expected = hmac.new(key.encode(), canonical.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise SystemExit(f"signature mismatch at line {line_number}")
            previous_signature = signature
            verified += 1

    if not verified:
        raise SystemExit("no signed entries found")
    print(f"Verified {verified} signed audit entries; found {legacy_unsigned} legacy unsigned entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

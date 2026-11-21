#!/usr/bin/env python3
"""Check DNS records needed by the Caddy public-domain deployment."""

from __future__ import annotations

import os
import socket
import sys


def main() -> int:
    domain = os.getenv("PUBLIC_DOMAIN", "localhost").strip().strip(".")
    expected_ipv4 = os.getenv("PUBLIC_IPV4")
    if domain in {"", "localhost"} or "." not in domain:
        raise SystemExit("Set PUBLIC_DOMAIN to a registered domain before checking public TLS DNS")

    failed = False
    for host in (domain, f"api.{domain}", f"litellm.{domain}", f"router.{domain}"):
        try:
            records = {item[4][0] for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
        except socket.gaierror:
            print(f"MISSING {host}")
            failed = True
            continue
        if expected_ipv4 and expected_ipv4 not in records:
            print(f"MISMATCH {host}: found {', '.join(sorted(records))}; expected {expected_ipv4}")
            failed = True
        else:
            print(f"OK {host}: {', '.join(sorted(records))}")
    if failed:
        return 1
    print("DNS resolves. Confirm TCP ports 80 and 443 reach this host for Caddy ACME validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

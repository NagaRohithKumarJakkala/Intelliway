#!/usr/bin/env python3
"""Prepare and run a reproducible, deliberately selected MMLU-Pro routing test."""

from __future__ import annotations

import argparse
import asyncio
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import sys
import time
import zipfile
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import pandas as pd


DATA_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
REFERENCE_REVISION = "f418b116db00b065c2aea046518d8fcf74d39872"
DATA_URL = f"https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/{DATA_REVISION}/data/test-00000-of-00001.parquet"
GITHUB = f"https://raw.githubusercontent.com/TIGER-AI-Lab/MMLU-Pro/{REFERENCE_REVISION}/eval_results"
REFERENCES = {
    "economical": [
        f"{GITHUB}/model_outputs_gpt-4o-mini_5shots.zip",
        f"{GITHUB}/model_outputs_gemini-1.5-flash-002_5shots.zip",
    ],
    "strong": [
        f"{GITHUB}/model_outputs_claude-3-5-sonnet-20241022_5shots.json.zip",
        f"{GITHUB}/model_outputs_gemini-3.1-pro_5-shots.zip",
    ],
}
PRESETS = ("auto-cost", "auto-balanced", "auto-quality")
MANIFEST_VERSION = 1
PROMPT_VERSION = "mmlu-pro-zero-shot-v1"
PARSER_VERSION = "final-marker-v2"
MAX_DOWNLOAD = 512 * 1024 * 1024
MAX_UNCOMPRESSED = 1024 * 1024 * 1024
MAX_MEMBER = 128 * 1024 * 1024
TELEMETRY_HEADERS = (
    "x-litellm-response-cost",
    "x-litellm-call-id",
    "x-litellm-model-id",
    "x-litellm-model-group",
    "x-litellm-version",
    "x-litellm-cache-key",
)
# No dedicated response-cache hit/miss header is verified in the upstream API.
# Cost values, cache keys, repeated IDs, and prompt-token caching are not hit indicators.
VERIFIED_CACHE_INDICATORS: dict[str, dict[str, str]] = {}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def answer_letter(value: Any, options: list[str]) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return chr(65 + value) if 0 <= value < len(options) else None
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*(?:\(?\s*)?([A-J])(?:\s*\)?)\s*[.!]?\s*", value, re.I)
    letter = match.group(1).upper() if match else None
    return letter if letter and ord(letter) - 65 < len(options) else None


def prediction_from_text(text: str, options: list[str]) -> tuple[str | None, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    markers = [
        i for i, line in enumerate(lines) if re.match(r"(?i)^Final answer\s*:", line)
    ]
    if len(markers) > 1:
        return None, "ambiguous-final-marker"
    if markers:
        if markers[0] != len(lines) - 1:
            return None, "nonfinal-marker"
        match = re.fullmatch(
            r"Final answer:\s*(?:\(([A-J])\)|([A-J]))[.!]?", lines[-1], re.I
        )
        if not match:
            return None, "unparseable"
        letter = (match.group(1) or match.group(2)).upper()
        return (
            (letter, "final-marker")
            if ord(letter) - 65 < len(options)
            else (None, "invalid-option")
        )
    if re.fullmatch(r"[A-J]", text.strip(), re.I):
        letter = text.strip().upper()
        return (
            (letter, "bare-letter")
            if ord(letter) - 65 < len(options)
            else (None, "invalid-option")
        )
    return None, "unparseable"


def record_id(record: dict[str, Any]) -> str | None:
    for key in ("question_id", "id", "questionId"):
        if key in record:
            value = record[key]
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                return None
            return str(value) if str(value).strip() else None
    return None


def walk_records(value: Any):
    if isinstance(value, list):
        for item in value:
            if (
                isinstance(item, dict)
                and record_id(item) is not None
                and isinstance(item.get("question"), str)
                and isinstance(item.get("options"), list)
                and 2 <= len(item["options"]) <= 10
                and all(isinstance(x, str) for x in item["options"])
                and isinstance(item.get("model_outputs"), (str, list))
            ):
                yield item
    elif isinstance(value, dict):
        # Subject-keyed lists are supported, but never search arbitrary metadata.
        for item in value.values():
            if isinstance(item, list):
                yield from walk_records(item)


def download(url: str, cache: Path) -> tuple[bytes, str]:
    if urlsplit(url).scheme != "https":
        raise ValueError("Source downloads require HTTPS")
    cache.mkdir(parents=True, exist_ok=True)
    name = sha(url.encode())
    path = cache / name
    if path.exists():
        if path.stat().st_size > MAX_DOWNLOAD:
            raise ValueError(f"Cached download exceeds size limit: {url}")
        data = path.read_bytes()
    else:
        with httpx.Client(timeout=httpx.Timeout(30.0), follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                if response.url.scheme != "https":
                    raise ValueError("Source download redirected away from HTTPS")
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_DOWNLOAD:
                        raise ValueError(f"Download exceeds size limit: {url}")
                    chunks.append(chunk)
                data = b"".join(chunks)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return data, sha(data)


def dataset_rows(data: bytes) -> list[dict[str, Any]]:
    frame = pd.read_parquet(__import__("io").BytesIO(data))
    required = {
        "question_id",
        "question",
        "options",
        "answer",
        "answer_index",
        "category",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Pinned dataset schema missing fields: {sorted(missing)}")
    rows = []
    seen = set()
    for raw in frame.to_dict(orient="records"):
        qid = record_id(raw)
        if qid is None:
            raise ValueError("Dataset question_id must be a nonempty string or integer")
        options = raw["options"]
        if hasattr(options, "tolist"):
            options = options.tolist()
        if qid in seen:
            raise ValueError(f"Duplicate dataset question_id: {qid}")
        seen.add(qid)
        if not isinstance(raw["question"], str) or not raw["question"].strip():
            raise ValueError(f"Question {qid} has empty question text")
        if (
            not isinstance(options, (list, tuple))
            or not 2 <= len(options) <= 10
            or any(not isinstance(x, str) or not x.strip() for x in options)
        ):
            raise ValueError(f"Question {qid} must have 2-10 nonempty ordered options")
        category = raw["category"]
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Question {qid} has empty category")
        idx = raw["answer_index"]
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise ValueError(f"Question {qid} answer_index must be an integer")
        gold = raw["answer"]
        if not isinstance(gold, str) or not re.fullmatch(r"[A-J]", gold):
            raise ValueError(
                f"Question {qid} answer must be an uppercase option letter"
            )
        if idx < 0 or idx >= len(options) or gold != chr(65 + idx):
            raise ValueError(f"Question {qid} answer and answer_index disagree")
        rows.append(
            {
                "question_id": qid,
                "question": raw["question"],
                "options": list(options),
                "answer": gold,
                "category": category.strip(),
                **({"src": raw["src"]} if isinstance(raw.get("src"), str) else {}),
            }
        )
    if not rows:
        raise ValueError("Pinned dataset is empty")
    return rows


def read_references(data: bytes, source: str) -> dict[str, list[dict[str, Any]]]:
    if len(data) > MAX_DOWNLOAD:
        raise ValueError(f"Reference archive exceeds size limit: {source}")
    output: dict[str, list[dict[str, Any]]] = {}
    total = 0
    with zipfile.ZipFile(__import__("io").BytesIO(data)) as archive:
        for info in archive.infolist():
            parts = info.filename.replace("\\", "/").split("/")
            if (
                info.filename.startswith(("/", "\\"))
                or ".." in parts
                or ":" in parts[0]
                or (info.external_attr >> 16) & 0o170000 == 0o120000
                or info.flag_bits & 1
            ):
                raise ValueError(
                    f"Unsafe reference archive member: {source}: {info.filename}"
                )
            total += info.file_size
            if info.file_size > MAX_MEMBER or total > MAX_UNCOMPRESSED:
                raise ValueError(
                    f"Reference archive member exceeds size limit: {source}"
                )
            if (
                info.is_dir()
                or not info.filename.lower().endswith(".json")
                or "__macosx/" in info.filename.lower()
                or Path(info.filename).name.startswith("._")
                or Path(info.filename).name.startswith(".")
            ):
                continue
            try:
                payload = json.loads(archive.read(info).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid reference JSON: {source}: {info.filename}"
                ) from exc
            for record in walk_records(payload):
                qid = record_id(record)
                if qid is None:
                    continue
                output.setdefault(qid, []).append(record)
    if not output:
        raise ValueError(f"No validated prediction records found in {source}")
    return output


def stable_hash(seed: int, qid: str, cohort: str) -> str:
    return hashlib.sha256(f"{seed}:{qid}:{cohort}".encode()).hexdigest()


def capacity_quotas(capacities: dict[str, int], target: int) -> dict[str, int]:
    if (
        isinstance(target, bool)
        or not isinstance(target, int)
        or target < 0
        or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0
            for v in capacities.values()
        )
    ):
        raise ValueError("Capacities and target must be nonnegative integers")
    if sum(capacities.values()) < target:
        raise ValueError(
            f"Insufficient cohort capacity: need {target}, have {sum(capacities.values())}"
        )
    names = sorted(capacities)
    quotas = {name: 0 for name in names}
    while sum(quotas.values()) < target:
        available = [name for name in names if quotas[name] < capacities[name]]
        level = min(quotas[name] for name in available)
        chosen = [name for name in available if quotas[name] == level]
        for name in chosen:
            if sum(quotas.values()) == target:
                break
            quotas[name] += 1
    return quotas


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    body = {k: v for k, v in manifest.items() if k != "fingerprint"}
    return sha(canonical(body))


def prepare(args) -> dict[str, Any]:
    cache = Path(args.output_dir) / "cache"
    data, data_hash = download(DATA_URL, cache)
    rows = dataset_rows(data)
    records_by_ref, hashes = {}, {}
    ref_order = [*REFERENCES["economical"], *REFERENCES["strong"]]
    for url in ref_order:
        archive, digest = download(url, cache)
        hashes[url] = digest
        records_by_ref[url] = read_references(archive, url)
    indexed = {r["question_id"]: r for r in rows}
    validated: dict[str, dict[str, str]] = {qid: {} for qid in indexed}
    exclusions: dict[str, dict[str, int]] = {
        url: dict.fromkeys(
            (
                "missing_record",
                "missing_prediction",
                "invalid_prediction",
                "duplicate_or_conflict",
                "source_drift",
                "unknown_question",
            ),
            0,
        )
        for url in ref_order
    }
    for url in ref_order:
        exclusions[url]["missing_record"] = len(
            indexed.keys() - records_by_ref[url].keys()
        )
        for qid, records in records_by_ref[url].items():
            row = indexed.get(qid)
            if row is None:
                exclusions[url]["unknown_question"] += 1
                continue
            if len(records) != 1:
                exclusions[url]["duplicate_or_conflict"] = (
                    exclusions[url].get("duplicate_or_conflict", 0) + 1
                )
                continue
            rec = records[0]
            opts = rec.get("options")
            if (
                normalize_text(rec.get("question", ""))
                != normalize_text(row["question"])
                or not isinstance(opts, list)
                or [normalize_text(x) for x in opts]
                != [normalize_text(x) for x in row["options"]]
            ):
                exclusions[url]["source_drift"] = (
                    exclusions[url].get("source_drift", 0) + 1
                )
                continue
            pred = answer_letter(rec.get("pred"), row["options"])
            if pred is None:
                reason = (
                    "missing_prediction"
                    if rec.get("pred") in (None, "")
                    else "invalid_prediction"
                )
                exclusions[url][reason] += 1
                continue
            validated[qid][url] = pred
    eligible = []
    econ_urls, strong_urls = REFERENCES["economical"], REFERENCES["strong"]
    for row in rows:
        qid = row["question_id"]
        preds = validated[qid]
        if len(preds) != 4:
            continue
        ec = sum(preds[u] == row["answer"] for u in econ_urls) / 2
        sc = sum(preds[u] == row["answer"] for u in strong_urls) / 2
        eligible.append(
            {
                **row,
                "reference_predictions": {Path(u).name: preds[u] for u in ref_order},
                "economical_correct": ec,
                "strong_correct": sc,
                "gap": sc - ec,
            }
        )
    if len(eligible) < 500:
        raise ValueError(
            f"Only {len(eligible)} questions have valid predictions from all four references; 500 required. Exclusions: {json.dumps(exclusions, sort_keys=True)}"
        )
    separating = [q for q in eligible if q["gap"] > 0]
    controls = [
        q for q in eligible if q["economical_correct"] == 1 and q["strong_correct"] == 1
    ]
    categories = sorted({q["category"] for q in rows})
    selected = []
    quotas_out = {}
    for cohort, candidates, target in (
        ("separating", separating, 400),
        ("control", controls, 100),
    ):
        by_cat = {c: [q for q in candidates if q["category"] == c] for c in categories}
        try:
            quota = capacity_quotas({c: len(v) for c, v in by_cat.items()}, target)
        except ValueError as exc:
            raise ValueError(
                f"{cohort}: {exc}. Exclusions: {json.dumps(exclusions, sort_keys=True)}"
            ) from exc
        quotas_out[cohort] = quota
        for category, count in quota.items():
            values = by_cat[category]
            if cohort == "separating":
                values.sort(
                    key=lambda q: (
                        -q["gap"],
                        stable_hash(args.seed, q["question_id"], cohort),
                    )
                )
            else:
                values.sort(
                    key=lambda q: stable_hash(args.seed, q["question_id"], cohort)
                )
            selected.extend([{**q, "cohort": cohort} for q in values[:count]])
    mixed = sorted(
        selected,
        key=lambda q: stable_hash(args.seed, q["question_id"], "evaluation-order"),
    )
    manifest = {
        "version": MANIFEST_VERSION,
        "seed": args.seed,
        "algorithm": "capacity-balanced-gap-hash-v2",
        "source": {
            "dataset_url": DATA_URL,
            "dataset_revision": DATA_REVISION,
            "dataset_sha256": data_hash,
            "reference_revision": REFERENCE_REVISION,
            "reference_urls": ref_order,
            "reference_sha256": hashes,
        },
        "selection": {
            "requested": {"separating": 400, "control": 100},
            "quotas": quotas_out,
            "exclusions": exclusions,
            "eligible_count": len(eligible),
            "dataset_count": len(rows),
            "reference_coverage": {
                u: sum(u in p for p in validated.values()) for u in ref_order
            },
            "eligible_cohort_counts": {
                "separating": len(separating),
                "control": len(controls),
            },
            "eligible_cohorts": {
                cohort: {
                    "count": len(pool),
                    "economical": statistics.mean(q["economical_correct"] for q in pool)
                    if pool
                    else None,
                    "strong": statistics.mean(q["strong_correct"] for q in pool)
                    if pool
                    else None,
                    "gap": statistics.mean(q["gap"] for q in pool) if pool else None,
                    "reference_accuracy": {
                        Path(u).name: statistics.mean(
                            q["reference_predictions"][Path(u).name] == q["answer"]
                            for q in pool
                        )
                        if pool
                        else None
                        for u in ref_order
                    },
                    "categories": {
                        c: sum(q["category"] == c for q in pool) for c in categories
                    },
                }
                for cohort, pool in (
                    ("all", eligible),
                    ("separating", separating),
                    ("control", controls),
                )
            },
            "eligible_accuracy": {
                "gap": statistics.mean(q["gap"] for q in eligible),
                "economical": statistics.mean(
                    q["economical_correct"] for q in eligible
                ),
                "strong": statistics.mean(q["strong_correct"] for q in eligible),
            },
            "selected_accuracy": {
                cohort: {
                    "count": sum(q["cohort"] == cohort for q in selected),
                    "gap": statistics.mean(
                        q["gap"] for q in selected if q["cohort"] == cohort
                    ),
                    "reference_accuracy": {
                        Path(u).name: statistics.mean(
                            q["reference_predictions"][Path(u).name] == q["answer"]
                            for q in selected
                            if q["cohort"] == cohort
                        )
                        for u in ref_order
                    },
                    "economical": statistics.mean(
                        q["economical_correct"]
                        for q in selected
                        if q["cohort"] == cohort
                    ),
                    "strong": statistics.mean(
                        q["strong_correct"] for q in selected if q["cohort"] == cohort
                    ),
                }
                for cohort in ("separating", "control")
            },
        },
        "questions": mixed,
    }
    manifest["fingerprint"] = manifest_fingerprint(manifest)
    out = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"
    if out.exists() and out.read_text() != encoded:
        raise ValueError(f"Refusing to overwrite differing immutable manifest: {out}")
    if not out.exists():
        out.write_text(encoded)
    return manifest


def load_manifest(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if (
        data.get("version") != MANIFEST_VERSION
        or data.get("fingerprint") != manifest_fingerprint(data)
        or len(data.get("questions", [])) != 500
        or len({q["question_id"] for q in data["questions"]}) != 500
    ):
        raise ValueError(
            "Manifest fingerprint or exact 500-question invariant is invalid"
        )
    for q in data["questions"]:
        if (
            q.get("cohort") not in ("separating", "control")
            or not isinstance(q.get("options"), list)
            or not 2 <= len(q["options"]) <= 10
            or answer_letter(q.get("answer"), q["options"]) != q["answer"]
        ):
            raise ValueError("Manifest question/cohort/options are invalid")
    if sum(q["cohort"] == "separating" for q in data["questions"]) != 400:
        raise ValueError(
            "Manifest must contain 400 separating and 100 control questions"
        )
    return data


def make_prompt(question: dict[str, Any]) -> list[dict[str, str]]:
    options = "\n".join(
        f"{chr(65 + i)}. {option}" for i, option in enumerate(question["options"])
    )
    return [
        {
            "role": "user",
            "content": f"Subject: {question['category']}\n\nQuestion: {question['question']}\n\nOptions:\n{options}\n\nReason concisely if useful. End with exactly one line: Final answer: (X), where X is the option letter.",
        }
    ]


def sanitized_url(url: str) -> str:
    p = urlsplit(url)
    if p.username or p.password:
        raise ValueError("Gateway URL must not contain credentials")
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("Gateway URL must be an absolute HTTP(S) URL")
    return urlunsplit((p.scheme, p.netloc, p.path, "", ""))


def atomic_json(path: Path, value: Any):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        stream.write(
            json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def read_jsonl(path: Path, repair: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    records = []
    valid_bytes = 0
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Non-object record in {path} at line {i + 1}")
            records.append(record)
            valid_bytes += len(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1:
                raise ValueError(f"Corrupt JSONL record in {path} at line {i + 1}")
            if repair:
                with path.open("r+b") as stream:
                    stream.truncate(valid_bytes)
            break
    if repair and records and valid_bytes and not data[:valid_bytes].endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return records


@contextmanager
def output_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".benchmark.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another benchmark writer owns {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validated_results(records, manifest, preset, settings_hash=None):
    questions = {q["question_id"]: q for q in manifest["questions"]}
    latest = {}
    identities = set()
    for r in records:
        qid = r.get("question_id")
        if (
            qid not in questions
            or r.get("preset") != preset
            or r.get("manifest_fingerprint") != manifest["fingerprint"]
        ):
            raise ValueError(
                f"Saved {preset} result has incompatible manifest/question identity"
            )
        identities.add(r.get("settings_fingerprint"))
        if not r.get("settings_fingerprint") or (
            settings_hash is not None and r["settings_fingerprint"] != settings_hash
        ):
            raise ValueError(
                f"Saved {preset} result has incompatible settings identity"
            )
        previous = latest.get(qid)
        expected = previous.get("attempt", 1) + 1 if previous else 1
        if r.get("attempt", 1) != expected or (
            previous
            and r.get("previous_attempt_fingerprint") != sha(canonical(previous))
        ):
            raise ValueError(f"Duplicate/conflicting result records for {preset}/{qid}")
        if previous and previous.get("status") not in (
            "transport_error",
            "http_error_retryable",
            "interrupted",
        ):
            raise ValueError(f"Invalid retry of terminal result for {preset}/{qid}")
        if r.get("status") not in (
            "completed",
            "parse_error",
            "truncated",
            "malformed_response",
            "transport_error",
            "http_error",
            "http_error_retryable",
            "interrupted",
        ):
            raise ValueError("Unknown terminal result status")
        latest[qid] = r
    if len(identities) > 1:
        raise ValueError("Conflicting settings in saved results")
    return list(latest.values())


def cost_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


def generation_signature(
    request: dict[str, Any], returned_model: str | None, telemetry=None
) -> str:
    return sha(
        canonical(
            {
                "messages": request["messages"],
                "temperature": request["temperature"],
                "max_tokens": request["max_tokens"],
                "model": returned_model,
                "deployment": {
                    k: (telemetry or {}).get(k)
                    for k in (
                        "x-litellm-model-id",
                        "x-litellm-model-group",
                        "x-litellm-version",
                    )
                },
            }
        )
    )


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict[str, Any]] = {}
        # The derived file is never authoritative; reconstruct even valid-but-incomplete ledgers.

    def rebuild(self, records):
        self.entries = {}
        for r in sorted(records, key=lambda r: r.get("dispatch_sequence", 0)):
            entry = r.get("generation_observation")
            if (
                isinstance(entry, dict)
                and cache_status(r.get("telemetry", {})) == "miss"
                and cost_number(entry.get("cost")) is not None
            ):
                try:
                    signature = generation_signature(
                        r["generation_request"], r["returned_model"], r.get("telemetry")
                    )
                    answer_fp = sha(
                        canonical(
                            {"text": r["text"], "finish_reason": r.get("finish_reason")}
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                expected_key = sha(
                    canonical({"signature": signature, "answer_fingerprint": answer_fp})
                )
                if (
                    entry.get("signature") == signature
                    and entry.get("answer_fingerprint") == answer_fp
                    and entry.get("key") == expected_key
                    and entry.get("cost")
                    == cost_number(
                        r.get("telemetry", {}).get("x-litellm-response-cost")
                    )
                    and entry.get("provenance")
                    == {
                        k: r.get(k)
                        for k in (
                            "preset",
                            "question_id",
                            "manifest_fingerprint",
                            "attempt",
                            "timestamp",
                            "dispatch_sequence",
                        )
                    }
                ):
                    self.entries.setdefault(entry["key"], entry)
        self.save()

    def lookup(self, key: str) -> dict[str, Any] | None:
        return self.entries.get(key)

    def add(self, key: str, entry: dict[str, Any]):
        self.entries.setdefault(key, entry)
        self.save()

    def save(self):
        cache_index: dict[str, list[str]] = {}
        for key, entry in self.entries.items():
            if entry.get("cache_key"):
                cache_index.setdefault(entry["cache_key"], []).append(key)
        atomic_json(
            self.path,
            {
                "version": 2,
                "entries": list(self.entries.values()),
                "cache_key_index": cache_index,
            },
        )


def cache_status(telemetry):
    statuses = {
        values[telemetry[header].lower()]
        for header, values in VERIFIED_CACHE_INDICATORS.items()
        if header in telemetry and telemetry[header].lower() in values
    }
    return statuses.pop() if len(statuses) == 1 else "unknown"


def account_cost(result, request, ledger):
    telemetry = result.get("telemetry", {})
    reported = cost_number(telemetry.get("x-litellm-response-cost"))
    result["reported_response_cost_usd"] = reported
    status = cache_status(telemetry)
    result["cache_status"] = status
    result["estimated_uncached_generation_cost_usd"] = None
    result["cost_estimate_source"] = None
    if not result.get("returned_model") or not isinstance(result.get("text"), str):
        return
    signature = generation_signature(request, result["returned_model"], telemetry)
    answer_fp = sha(
        canonical(
            {"text": result["text"], "finish_reason": result.get("finish_reason")}
        )
    )
    key = sha(canonical({"signature": signature, "answer_fingerprint": answer_fp}))
    result["generation_signature"] = signature
    result["generation_request"] = {
        k: request[k] for k in ("messages", "temperature", "max_tokens")
    }
    result["answer_fingerprint"] = answer_fp
    existing = ledger.lookup(key)
    if status == "miss" and reported is not None:
        entry = {
            "key": key,
            "signature": signature,
            "answer_fingerprint": answer_fp,
            "cache_key": telemetry.get("x-litellm-cache-key"),
            "cost": reported,
            "provenance": {
                "preset": result["preset"],
                "question_id": result["question_id"],
                "manifest_fingerprint": result["manifest_fingerprint"],
                "attempt": result["attempt"],
                "timestamp": result["timestamp"],
                "dispatch_sequence": result.get("dispatch_sequence"),
            },
        }
        result["generation_observation"] = entry
        result["estimated_uncached_generation_cost_usd"] = reported
        result["cost_estimate_source"] = entry["provenance"]
    elif status == "hit" and existing:
        result["estimated_uncached_generation_cost_usd"] = existing["cost"]
        result["cost_estimate_source"] = existing["provenance"]


def sanitized_error(text, secret=None):
    try:
        text = json.dumps(
            redact_credentials(json.loads(text), secret), ensure_ascii=True
        )
    except (ValueError, TypeError):
        pass
    if secret:
        text = text.replace(secret, "[redacted]")
    text = re.sub(r"(?i)bearer\s+[^\s\"<>]+", "Bearer [redacted]", text)
    text = re.sub(r"https?://[^\s\"<>]+", "[url redacted]", text, flags=re.I)
    text = re.sub(
        r"(?i)([\"']?(?:api[_-]?key|authorization|token|password|secret)[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,}]+)",
        r'\1"[redacted]"',
        text,
    )
    return "".join(c if c.isprintable() or c == "\n" else "?" for c in text)[:2000]


def redact_credentials(value: Any, secret=None) -> Any:
    if isinstance(value, dict):
        return {
            k: "[redacted]"
            if re.search(
                r"(?i)authorization|api[_-]?key|password|access[_-]?token|secret|credential",
                k,
            )
            or k.lower() in ("token", "key")
            else redact_credentials(v, secret)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_credentials(v, secret) for v in value]
    if isinstance(value, str):
        if secret:
            value = value.replace(secret, "[redacted]")
        value = re.sub(
            r"(?i)https?://[^/\s\"<>]*@[^\s\"<>]+", "[credential URL redacted]", value
        )
        return re.sub(r"(?i)bearer\s+[^\s\"<>]+", "Bearer [redacted]", value)
    return value


def result_path(root: Path, preset: str) -> Path:
    return root / f"{preset}.jsonl"


def _result_fingerprint(settings: dict[str, Any]) -> str:
    return sha(canonical(settings))


async def run_async(args):
    with output_lock(Path(args.output_dir)):
        await evaluate(args)
        summarize(
            argparse.Namespace(
                manifest=args.manifest, output_dir=args.output_dir, seed=42
            )
        )


def append_result(path, result):
    with path.open("a") as stream:
        stream.write(json.dumps(result, ensure_ascii=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


async def evaluate(args):
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_tokens < 1:
        raise ValueError(
            "timeout must be finite and positive; max-tokens must be positive"
        )
    manifest = load_manifest(args.manifest)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    endpoint = sanitized_url(args.base_url).rstrip("/")
    if not endpoint.endswith("/v1"):
        raise ValueError("Gateway base URL must end in /v1")
    timeout = args.timeout
    settings = {
        "manifest_fingerprint": manifest["fingerprint"],
        "endpoint": endpoint,
        "timeout": timeout,
        "dispatch": "sequential",
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": False,
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
    }
    settings_hash = _result_fingerprint(settings)
    meta_path = root / "run.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("settings_fingerprint") != settings_hash:
            raise ValueError("Resume settings differ from existing run metadata")
        meta["resumed_at"] = now()
    else:
        meta = {
            **settings,
            "settings_fingerprint": settings_hash,
            "started_at": now(),
            "software_version": "2",
            "determinism": "temperature=0 does not guarantee provider/routing determinism",
            "resume_warning": "Interrupted requests may have consumed provider work; ambiguous requests are never automatically resent",
        }
    meta["endpoint"] = endpoint
    meta.pop("ended_at", None)
    meta.pop("stopped_early", None)
    auth_name = args.api_key_env
    if auth_name and (
        auth_name in {"LITELLM_MASTER_KEY", "JEV_API_KEY"}
        or re.search(
            r"(?:OPENAI|ANTHROPIC|GEMINI|GOOGLE|DEEPSEEK|ZHIPU|GROQ|MISTRAL|AWS|AZURE).*KEY",
            auth_name,
            re.I,
        )
    ):
        raise ValueError(
            "Use a dedicated gateway bearer-key environment variable, not a provider/master key"
        )
    key = os.environ.get(auth_name) if auth_name else None
    if auth_name and not key:
        raise ValueError(
            f"Explicit gateway bearer-key environment variable is empty: {auth_name}"
        )
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    paths = {preset: result_path(root, preset) for preset in PRESETS}
    recorded = {p: read_jsonl(path, repair=True) for p, path in paths.items()}
    latest = {
        p: {
            r["question_id"]: r
            for r in validated_results(records, manifest, p, settings_hash)
        }
        for p, records in recorded.items()
    }
    questions_by_id = {q["question_id"]: q for q in manifest["questions"]}
    for preset, records in recorded.items():
        for record in records:
            q = questions_by_id[record["question_id"]]
            expected_request = {
                "model": preset,
                "messages": make_prompt(q),
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "stream": False,
            }
            if (
                record.get("request_fingerprint") != sha(canonical(expected_request))
                or record.get("gold") != q["answer"]
                or record.get("category") != q["category"]
                or record.get("cohort") != q["cohort"]
            ):
                raise ValueError(
                    "Saved result has incompatible request/question fields"
                )
    journal = root / "in-flight.json"
    if journal.exists():
        pending = json.loads(journal.read_text())
        preset, qid = pending.get("preset"), pending.get("question_id")
        if (
            preset not in PRESETS
            or pending.get("settings_fingerprint") != settings_hash
            or pending.get("manifest_fingerprint") != manifest["fingerprint"]
            or qid not in questions_by_id
        ):
            raise ValueError("In-flight journal has incompatible identity")
        saved = latest[preset].get(qid)
        if saved and saved.get("attempt", 1) == pending.get("attempt"):
            if any(
                saved.get(k) != pending.get(k)
                for k in (
                    "request_fingerprint",
                    "timestamp",
                    "previous_attempt_fingerprint",
                )
            ):
                raise ValueError(
                    "In-flight journal conflicts with saved terminal result"
                )
            journal.unlink()
        else:
            pending.update(
                {
                    "status": "interrupted",
                    "correct": None,
                    "error": "Completion unknown after interruption; provider work may already have occurred",
                    "ended_at": now(),
                    "elapsed_seconds": None,
                }
            )
            validated_results(
                [*recorded[preset], pending], manifest, preset, settings_hash
            )
            append_result(paths[preset], pending)
            recorded[preset].append(pending)
            latest[preset] = {
                r["question_id"]: r
                for r in validated_results(
                    recorded[preset], manifest, preset, settings_hash
                )
            }
            journal.unlink()
            print(
                "warning: persisted an ambiguous interrupted request; it will not be resent automatically",
                file=sys.stderr,
            )
    atomic_json(meta_path, meta)
    ledger = Ledger(root / "cost-ledger.json")
    ledger.rebuild([r for records in recorded.values() for r in records])
    dispatch_sequence = max(
        (
            r.get("dispatch_sequence", 0)
            for records in recorded.values()
            for r in records
        ),
        default=0,
    )
    timeout_obj = httpx.Timeout(timeout)
    transport = getattr(args, "transport", None)
    async with httpx.AsyncClient(
        timeout=timeout_obj, headers=headers, transport=transport
    ) as client:
        first = manifest["questions"][0]
        # Score one request per preset before scheduling the remainder.
        scheduled = [(first, preset) for preset in PRESETS]
        scheduled += [
            (q, p)
            for i, q in enumerate(manifest["questions"][1:], 1)
            for p in (PRESETS[(i + j) % 3] for j in range(3))
        ]
        for question, preset in scheduled:
            prior = latest[preset]
            qid = question["question_id"]
            if qid in prior:
                old = prior[qid]
                if args.retry_errors and old.get("status") in (
                    "transport_error",
                    "http_error_retryable",
                ):
                    pass
                elif old.get("status") == "interrupted" and getattr(
                    args, "retry_interrupted", False
                ):
                    pass
                else:
                    if old.get("http_status") in (400, 401, 403, 404, 405, 422, 503):
                        raise RuntimeError(
                            f"Saved configuration/authentication failure for {preset}; correct gateway configuration and explicitly retry eligible HTTP errors with --retry-errors (permanent errors require a new output directory)"
                        )
                    continue
            request = {
                "model": preset,
                "messages": make_prompt(question),
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "stream": False,
            }
            started = time.monotonic()
            dispatch_sequence += 1
            result = {
                "question_id": qid,
                "cohort": question["cohort"],
                "category": question["category"],
                "preset": preset,
                "manifest_fingerprint": manifest["fingerprint"],
                "settings_fingerprint": settings_hash,
                "request_fingerprint": sha(canonical(request)),
                "timestamp": now(),
                "dispatch_sequence": dispatch_sequence,
                "attempt": prior[qid].get("attempt", 1) + 1 if qid in prior else 1,
                "gold": question["answer"],
                "correct": None,
            }
            if qid in prior:
                result["previous_attempt_fingerprint"] = sha(canonical(prior[qid]))
            atomic_json(journal, result)
            try:
                response = await client.post(
                    endpoint + "/chat/completions", json=request
                )
                result["elapsed_seconds"] = time.monotonic() - started
                result["telemetry"] = {
                    k: v for k, v in response.headers.items() if k in TELEMETRY_HEADERS
                }
                result["http_status"] = response.status_code
                if response.status_code != 200:
                    result["status"] = (
                        "http_error_retryable"
                        if response.status_code in (408, 425, 429)
                        or response.status_code >= 500
                        else "http_error"
                    )
                    result["error"] = {
                        "message": sanitized_error(response.text, key),
                        "status": response.status_code,
                    }
                else:
                    try:
                        body = response.json()
                        canonical(body)
                        result["raw_response"] = body
                        if (
                            not isinstance(body, dict)
                            or not isinstance(body.get("choices"), list)
                            or len(body["choices"]) != 1
                        ):
                            raise ValueError("Expected one completion choice")
                        choice = body["choices"][0]
                        text = choice["message"]["content"]
                        finish = choice.get("finish_reason")
                        if not isinstance(text, str):
                            raise ValueError("completion content is not text")
                        if (
                            finish not in ("stop", "length")
                            or (
                                body.get("model") is not None
                                and not isinstance(body["model"], str)
                            )
                            or (
                                body.get("usage") is not None
                                and not isinstance(body["usage"], dict)
                            )
                        ):
                            raise ValueError(
                                "Malformed completion metadata or unsupported finish reason"
                            )
                        pred, parser = prediction_from_text(text, question["options"])
                        result.update(
                            {
                                "status": "truncated"
                                if finish == "length"
                                else (
                                    "completed" if pred is not None else "parse_error"
                                ),
                                "raw_response": body,
                                "response_id": body.get("id"),
                                "returned_model": body.get("model"),
                                "text": text,
                                "prediction": None if finish == "length" else pred,
                                "parser_status": "truncated"
                                if finish == "length"
                                else parser,
                                "gold": question["answer"],
                                "correct": None
                                if finish == "length" or pred is None
                                else pred == question["answer"],
                                "usage": body.get("usage"),
                                "finish_reason": finish,
                            }
                        )
                    except (
                        ValueError,
                        KeyError,
                        IndexError,
                        TypeError,
                        json.JSONDecodeError,
                    ) as exc:
                        result.setdefault(
                            "raw_response", sanitized_error(response.text, key)
                        )
                        result.update(
                            {
                                "status": "malformed_response",
                                "error": type(exc).__name__,
                                "correct": None,
                            }
                        )
            except httpx.TransportError as exc:
                result.update(
                    {
                        "status": "transport_error",
                        "error": type(exc).__name__,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
            result.setdefault("status", "completed")
            result["ended_at"] = now()
            result = redact_credentials(result, key)
            account_cost(result, request, ledger)
            append_result(paths[preset], result)
            if result.get("generation_observation"):
                entry = result["generation_observation"]
                ledger.add(entry["key"], entry)
            journal.unlink()
            recorded[preset].append(result)
            latest[preset][qid] = result
            if result["status"] in (
                "http_error",
                "http_error_retryable",
            ) and result.get("http_status") in (
                401,
                403,
                400,
                404,
                405,
                422,
                503,
            ):
                atomic_json(
                    meta_path, {**meta, "ended_at": now(), "stopped_early": result}
                )
                raise RuntimeError(
                    f"Gateway configuration/authentication failure for {preset}: HTTP {result['http_status']}; saved result and stopped"
                )
    meta["ended_at"] = now()
    meta["logical_results"] = sum(len(v) for v in latest.values())
    meta["extra_calls"] = (
        sum(len(v) for v in recorded.values()) - meta["logical_results"]
    )
    atomic_json(meta_path, meta)


def summary(root: Path, manifest: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    questions = {q["question_id"]: q for q in manifest["questions"]}
    metrics, comparison, pairs = {}, [], {}

    def finite_number(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        )

    def group_metrics(ids, by_id, attempts):
        records = [by_id[qid] for qid in ids if qid in by_id]
        scheduled = len(ids)
        scored = [r for r in records if isinstance(r.get("correct"), bool)]
        correct = sum(r.get("correct") is True for r in scored)
        latencies = [
            r["elapsed_seconds"]
            for r in records
            if finite_number(r.get("elapsed_seconds"))
        ]
        m = {
            "scheduled": scheduled,
            "recorded": len(records),
            "pending": scheduled - len(records),
            "complete": len(records) == scheduled,
            "recorded_coverage": len(records) / scheduled if scheduled else None,
            "correct": correct,
            "accuracy_all_scheduled": correct / scheduled if scheduled else None,
            "scorable": len(scored),
            "scorable_coverage": len(scored) / scheduled if scheduled else None,
            "scorable_accuracy": correct / len(scored) if scored else None,
            "attempts": len(attempts),
            "extra_calls": len(attempts) - len(records),
            "status_counts": {
                s: sum(r.get("status") == s for r in records)
                for s in sorted({r.get("status", "unknown") for r in records})
            },
            "http_errors": sum(
                str(r.get("status", "")).startswith("http_error") for r in records
            ),
            "transport_errors": sum(
                r.get("status") == "transport_error" for r in records
            ),
            "malformed_response_errors": sum(
                r.get("status") == "malformed_response" for r in records
            ),
            "parse_errors": sum(
                r.get("status") == "parse_error"
                or (r.get("status") == "completed" and r.get("correct") is None)
                for r in records
            ),
            "truncation_errors": sum(r.get("status") == "truncated" for r in records),
            "returned_models": {},
            "returned_model_coverage": sum(
                bool(r.get("returned_model")) for r in records
            ),
            "usage": {},
            "usage_coverage": {},
            "usage_records": sum(isinstance(r.get("usage"), dict) for r in records),
            "latency_coverage": len(latencies),
            "median_latency_seconds": statistics.median(latencies)
            if latencies
            else None,
            "p95_latency_seconds": sorted(latencies)[
                min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)
            ]
            if latencies
            else None,
            "cache_counts": {
                s: sum(
                    (
                        r.get("cache_status")
                        if r.get("cache_status") in ("hit", "miss")
                        else "unknown"
                    )
                    == s
                    for r in records
                )
                for s in ("hit", "miss", "unknown")
            },
        }
        for field, prefix in (
            ("reported_response_cost_usd", "reported"),
            ("estimated_uncached_generation_cost_usd", "estimated"),
        ):
            costs = [r[field] for r in records if finite_number(r.get(field))]
            complete = scheduled > 0 and len(costs) == scheduled
            m[field] = sum(costs) if complete else None
            m[f"{prefix}_cost_partial_sum_usd"] = sum(costs)
            m[f"{prefix}_cost_coverage"] = len(costs)
            m[f"{prefix}_cost_coverage_fraction"] = (
                len(costs) / scheduled if scheduled else None
            )
            m[f"{prefix}_cost_complete"] = complete
            m[f"{prefix}_cost_per_correct_usd"] = (
                sum(costs) / correct if complete and correct else None
            )
        for r in records:
            model = r.get("returned_model") or "unknown"
            m["returned_models"][model] = m["returned_models"].get(model, 0) + 1
            usage = r.get("usage")
            for key, value in (usage if isinstance(usage, dict) else {}).items():
                if finite_number(value):
                    m["usage"][key] = m["usage"].get(key, 0) + value
                    m["usage_coverage"][key] = m["usage_coverage"].get(key, 0) + 1
        return m

    cohorts = {
        name: [qid for qid, q in questions.items() if q["cohort"] == name]
        for name in sorted({q["cohort"] for q in questions.values()})
    }
    categories = {
        name: [qid for qid, q in questions.items() if q["category"] == name]
        for name in sorted({q["category"] for q in questions.values()})
    }
    category_cohort = {
        f"{cohort}/{category}": [
            qid for qid in ids if questions[qid]["category"] == category
        ]
        for cohort, ids in cohorts.items()
        for category in categories
        if any(questions[qid]["category"] == category for qid in ids)
    }
    settings_ids = set()
    meta_path = root / "run.json"
    settings_hash = (
        json.loads(meta_path.read_text()).get("settings_fingerprint")
        if meta_path.exists()
        else None
    )
    for preset in PRESETS:
        attempts = read_jsonl(result_path(root, preset))
        records = validated_results(attempts, manifest, preset, settings_hash)
        settings_ids.update(r["settings_fingerprint"] for r in records)
        if len(settings_ids) > 1:
            raise ValueError(
                "Cannot compare results with incompatible settings identity"
            )
        by_id = {r["question_id"]: r for r in records}
        pairs[preset] = by_id
        metrics[preset] = group_metrics(questions, by_id, attempts)
        for field, groups in (
            ("cohorts", cohorts),
            ("categories", categories),
            ("category_cohort", category_cohort),
        ):
            metrics[preset][field] = {
                name: group_metrics(
                    ids, by_id, [r for r in attempts if r["question_id"] in ids]
                )
                for name, ids in groups.items()
            }
    for left, right in itertools.combinations(PRESETS, 2):
        for cohort, ids in {"all": list(questions), **cohorts}.items():
            both = [
                (
                    pairs[left].get(qid, {}).get("correct"),
                    pairs[right].get(qid, {}).get("correct"),
                )
                for qid in ids
            ]
            scorable = [
                (a, b) for a, b in both if isinstance(a, bool) and isinstance(b, bool)
            ]
            recorded = sum(qid in pairs[left] and qid in pairs[right] for qid in ids)
            for population, values in (
                ("all_scheduled", [(a is True, b is True) for a, b in both]),
                ("jointly_scorable", scorable),
            ):
                diff = [int(b) - int(a) for a, b in values]
                comparison.append(
                    {
                        "left": left,
                        "right": right,
                        "cohort": cohort,
                        "population": population,
                        "scheduled": len(ids),
                        "paired": len(values),
                        "both_recorded": recorded,
                        "jointly_scorable": len(scorable),
                        "paired_coverage": len(values) / len(ids) if ids else None,
                        "jointly_scorable_coverage": len(scorable) / len(ids)
                        if ids
                        else None,
                        "complete": recorded == len(ids),
                        "accuracy_delta_pp": 100 * statistics.mean(diff)
                        if diff
                        else None,
                        "left_only_correct": sum(a and not b for a, b in values),
                        "right_only_correct": sum(b and not a for a, b in values),
                        "bootstrap_95pp": bootstrap_mean(diff, seed),
                    }
                )
    return {
        "manifest_fingerprint": manifest["fingerprint"],
        "bootstrap_seed": seed,
        "interpretation": "Deliberately selected routing stress test; not representative or leaderboard score. Independent historical predictions and five-shot protocols may not transfer to today's deployed models or zero-shot prompts. Controls intentionally dilute aggregate separation. One run per preset and cache reuse cannot establish a general routing advantage or benefit over a fixed-model baseline.",
        "caveats": [
            "All-scheduled accuracy counts pending and unscorable questions as not correct; partial-run deltas are provisional. Jointly scorable comparisons exclude operational failures and pending pairs and can select a different population.",
            "Paired bootstrap intervals are descriptive of this selected subset, not the entire dataset or repeated-run uncertainty; resampling uses immutable question pairs and right-minus-left percentage points.",
            "Costs are successful generation estimates, not audited bills or total routing cost; they exclude Jev routing, unobserved failed/fallback attempts, infrastructure and local hardware. Latest logical attempts are counted; extra calls may incur additional unreported cost.",
            "Cost coverage uses all scheduled questions. Incomplete coverage yields only a partial sum, with complete totals and cost-per-correct unknown; missing telemetry is not zero.",
            "Cached answers are not fresh independent generations. Latency is end-to-end service latency, not uncached model latency. Unknown cache status is not inferred from response IDs. Without a verified cache-hit indicator, cache status and original generation costs may remain unknown.",
            "Usage totals include only available numeric fields; per-field coverage counts are reported, not imputed. Routing/provider determinism is not guaranteed. Resume may duplicate interrupted in-flight provider work.",
        ],
        "metrics": metrics,
        "comparison": comparison,
    }


def bootstrap_mean(values: list[int], seed: int, rounds: int = 2000):
    if not values:
        return None
    if rounds < 1:
        raise ValueError("Bootstrap rounds must be positive")
    if min(values) == max(values):
        return [100 * values[0], 100 * values[0]]
    import random

    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(rounds)
    )
    return [
        100 * samples[int(0.025 * rounds)],
        100 * samples[min(rounds - 1, int(0.975 * rounds))],
    ]


def summarize(args):
    manifest = load_manifest(args.manifest)
    root = Path(args.output_dir)
    report = summary(root, manifest, args.seed)
    atomic_json(root / "summary.json", report)
    rows = []
    for preset, metrics in report["metrics"].items():
        groups = [("all", "all", metrics)]
        for scope in ("cohorts", "categories", "category_cohort"):
            groups.extend((scope, name, m) for name, m in metrics[scope].items())
        for scope, group, m in groups:
            row = {"preset": preset, "scope": scope, "group": group}
            for key, value in m.items():
                if key not in ("cohorts", "categories", "category_cohort"):
                    row[key] = (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, dict)
                        else value
                    )
            rows.append(row)
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "comparison.csv").open("w", newline="") as f:
        rows = report["comparison"]
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0]) if rows else ["left", "right"]
        )
        writer.writeheader()
        writer.writerows(
            {**row, "bootstrap_95pp": json.dumps(row["bootstrap_95pp"])} for row in rows
        )
    print(
        "preset          cohort       complete  accuracy/scheduled  scorable  pending  estimated cost"
    )
    for p, m in report["metrics"].items():
        for cohort, group in {"all": m, **m["cohorts"]}.items():
            print(
                f"{p:15} {cohort:12} {str(group['complete']):8} {group['accuracy_all_scheduled']:.3f}/{group['scheduled']:<3}          {group['scorable']:3}       {group['pending']:3}      {group['estimated_uncached_generation_cost_usd']}"
            )
    print(report["interpretation"])
    for caveat in report["caveats"]:
        print(caveat)


def parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="Examples:\n  uv sync --group benchmark\n  uv run --group benchmark python scripts/benchmark_mmlu_pro.py prepare\n  uv run --group benchmark python scripts/benchmark_mmlu_pro.py run # all three presets\n  uv run --group benchmark python scripts/benchmark_mmlu_pro.py run # resume, skip terminal pairs\n  uv run --group benchmark python scripts/benchmark_mmlu_pro.py run --retry-errors # eligible recorded errors only\n  uv run --group benchmark python scripts/benchmark_mmlu_pro.py summarize # offline\n\nGeneration is sequential; a deliberate run sends up to 1,500 logical evaluations.\nCached responses may have unknown original generation cost; no costs are audited spend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser(
        "prepare", help="Download pinned public artifacts and freeze subset"
    )
    q.add_argument("--output-dir", default="benchmark-results")
    q.add_argument("--manifest", default="benchmark-results/subset.json")
    q.add_argument("--seed", type=int, default=42)
    q = sub.add_parser("run", help="Evaluate all three automatic presets")
    q.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    q.add_argument("--output-dir", default="benchmark-results")
    q.add_argument("--manifest", default="benchmark-results/subset.json")
    q.add_argument("--timeout", type=float, default=360)
    q.add_argument("--max-tokens", type=int, default=2048)
    q.add_argument(
        "--api-key-env",
        help="Name of a dedicated gateway bearer-key environment variable; no implicit .env loading",
    )
    q.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry recorded transport/retryable HTTP errors, retaining attempt history",
    )
    q.add_argument(
        "--retry-interrupted",
        action="store_true",
        help="Explicitly resend ambiguous interrupted requests; may duplicate paid provider work",
    )
    q = sub.add_parser("summarize", help="Regenerate reports offline")
    q.add_argument("--output-dir", default="benchmark-results")
    q.add_argument("--manifest", default="benchmark-results/subset.json")
    q.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        with output_lock(Path(args.output_dir)):
            result = prepare(args)
        print(f"Prepared {len(result['questions'])} questions: {result['fingerprint']}")
    elif args.command == "run":
        if not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_tokens < 1:
            raise SystemExit(
                "timeout must be finite and positive; max-tokens must be positive"
            )
        asyncio.run(run_async(args))
    else:
        with output_lock(Path(args.output_dir)):
            summarize(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

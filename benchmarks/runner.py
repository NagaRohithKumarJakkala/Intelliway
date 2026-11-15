"""Run candidate models and score their responses with one fixed judge model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from gateway.upstream import send_to_litellm


ROOT = Path(__file__).resolve().parent
DEFAULT_QUERIES = ROOT / "queries.jsonl"
DEFAULT_PRICING = ROOT / "pricing.json"
DEFAULT_OUTPUT = Path("evidence/benchmark-results.jsonl")
DEFAULT_JUDGE = os.getenv("BENCHMARK_JUDGE_MODEL", "external-general")
DEFAULT_CANDIDATES = tuple(
    model.strip()
    for model in os.getenv(
        "BENCHMARK_CANDIDATE_MODELS",
        "local-general,external-general",
    ).split(",")
    if model.strip()
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            for required in ("id", "prompt", "criteria"):
                if required not in item:
                    raise ValueError(f"{path}:{line_number} is missing {required!r}")
            queries.append(item)
    if not queries:
        raise ValueError(f"{path} does not contain any benchmark queries")
    return queries


def _load_pricing(path: Path) -> dict[str, dict[str, float | None]]:
    with path.open(encoding="utf-8") as file:
        pricing = json.load(file)
    if not isinstance(pricing, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return pricing


def _usage(response: dict[str, Any]) -> dict[str, Any]:
    value = response.get("usage")
    return value if isinstance(value, dict) else {}


def _response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("provider response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("provider response has no text content")
    return message["content"]


def _cost(response: dict[str, Any], model: str, pricing: dict[str, Any]) -> float | None:
    for value in (response.get("cost"), _usage(response).get("cost")):
        if isinstance(value, (int, float)):
            return float(value)
    model_pricing = pricing.get(model)
    usage = _usage(response)
    if not isinstance(model_pricing, dict):
        return None
    input_rate = model_pricing.get("input_per_million_tokens")
    output_rate = model_pricing.get("output_per_million_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not all(isinstance(value, (int, float)) for value in (
        input_rate, output_rate, prompt_tokens, completion_tokens
    )):
        return None
    return (
        float(prompt_tokens) * float(input_rate)
        + float(completion_tokens) * float(output_rate)
    ) / 1_000_000


def _judge_prompt(query: dict[str, Any], answer: str) -> str:
    return json.dumps({
        "task": "Evaluate the candidate answer against the benchmark.",
        "query": query["prompt"],
        "criteria": query["criteria"],
        "answer": answer,
        "output_format": {
            "quality_label": "one of: poor, fair, good, excellent",
            "score": "integer from 1 to 5",
            "reason": "brief explanation",
            "criteria_results": "array of booleans, one per criterion",
        },
        "instruction": "Return JSON only. Do not use markdown.",
    })


def _parse_judgement(text: str) -> dict[str, Any]:
    try:
        judgement = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("judge response was not valid JSON") from exc
    if not isinstance(judgement, dict):
        raise ValueError("judge response must be a JSON object")
    label = judgement.get("quality_label")
    score = judgement.get("score")
    if label not in {"poor", "fair", "good", "excellent"}:
        raise ValueError("judge returned an invalid quality_label")
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError("judge returned an invalid score")
    return judgement


async def _run_one(
    query: dict[str, Any],
    candidate: str,
    judge: str,
    pricing: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "query_id": query["id"],
        "candidate_model": candidate,
        "judge_model": judge,
        "timestamp": time.time(),
    }
    started = time.perf_counter()
    try:
        response = await send_to_litellm(
            model=candidate,
            messages=[{"role": "user", "content": query["prompt"]}],
        )
        result.update({
            "status": "success",
            "answer": _response_text(response),
            "usage": _usage(response),
            "cost": _cost(response, candidate, pricing),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        })
    except Exception as exc:
        result.update({
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error_class": type(exc).__name__,
            "error": str(exc),
        })
        return result

    judge_started = time.perf_counter()
    try:
        judgement_response = await send_to_litellm(
            model=judge,
            messages=[{"role": "user", "content": _judge_prompt(query, result["answer"])}],
            temperature=0,
        )
        result["judgement"] = _parse_judgement(_response_text(judgement_response))
        result["judge_usage"] = _usage(judgement_response)
        result["judge_cost"] = _cost(judgement_response, judge, pricing)
        result["judge_latency_ms"] = round(
            (time.perf_counter() - judge_started) * 1000, 2
        )
        result["label_status"] = "success"
    except Exception as exc:
        result.update({
            "label_status": "error",
            "judge_latency_ms": round((time.perf_counter() - judge_started) * 1000, 2),
            "judge_error_class": type(exc).__name__,
            "judge_error": str(exc),
        })
    return result


async def run_benchmark(
    queries_path: Path,
    output_path: Path,
    pricing_path: Path,
    candidates: tuple[str, ...],
    judge: str,
) -> int:
    queries = _read_jsonl(queries_path)
    pricing = _load_pricing(pricing_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        await _run_one(query, candidate, judge, pricing)
        for query in queries
        for candidate in candidates
    ]
    with output_path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
    successes = sum(record["status"] == "success" for record in records)
    labeled = sum(record.get("label_status") == "success" for record in records)
    print(f"wrote {len(records)} records to {output_path}")
    print(f"candidate successes: {successes}/{len(records)}; quality labels: {labeled}/{len(records)}")
    return 0 if labeled == len(records) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    args = parser.parse_args()
    candidates = tuple(args.candidates or DEFAULT_CANDIDATES)
    if not candidates:
        parser.error("at least one --candidate or BENCHMARK_CANDIDATE_MODELS is required")
    return asyncio.run(run_benchmark(
        args.queries, args.output, args.pricing, candidates, args.judge,
    ))


if __name__ == "__main__":
    raise SystemExit(main())

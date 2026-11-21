import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from scripts import benchmark_mmlu_pro as b


@pytest.fixture
def setup_run(tmp_path):
    manifest = {
        "version": 1,
        "questions": [
            {
                "question_id": str(i),
                "category": "math",
                "cohort": "separating" if i < 400 else "control",
                "question": f"Question {i}",
                "options": ["Alpha", "Beta"],
                "answer": "A",
            }
            for i in range(500)
        ],
    }
    manifest["fingerprint"] = b.manifest_fingerprint(manifest)
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(manifest))
    args = SimpleNamespace(
        manifest=str(path),
        output_dir=str(tmp_path),
        base_url="http://gateway.test/v1",
        timeout=360,
        max_tokens=2048,
        api_key_env=None,
        retry_errors=False,
        retry_interrupted=False,
    )
    return args, manifest, tmp_path


def response(text="Final answer: (A)", finish="stop", **kwargs):
    return httpx.Response(
        200,
        json={
            "id": "same-id",
            "model": "model-a",
            "choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2},
        },
        **kwargs,
    )


def run(args, handler):
    args.transport = httpx.MockTransport(handler)
    asyncio.run(b.run_async(args))


def test_exact_1500_interleaved_requests_schema_prompt_equality_and_resume(setup_run):
    args, manifest, root = setup_run
    calls = []

    def handler(request):
        data = json.loads(request.content)
        assert set(data) == {"model", "messages", "temperature", "max_tokens", "stream"}
        assert data["temperature"] == 0 and data["stream"] is False
        assert request.url.path == "/v1/chat/completions"
        assert request.extensions["timeout"]["read"] == 360
        calls.append(data)
        if len(calls) > 1:
            assert (
                sum(len(b.read_jsonl(root / f"{p}.jsonl")) for p in b.PRESETS)
                == len(calls) - 1
            )
        return response()

    run(args, handler)
    assert len(calls) == 1500
    assert [c["model"] for c in calls[:6]] == [
        *b.PRESETS,
        "auto-balanced",
        "auto-quality",
        "auto-cost",
    ]
    for i in range(0, 1500, 3):
        assert (
            calls[i]["messages"] == calls[i + 1]["messages"] == calls[i + 2]["messages"]
        )
    run(args, lambda _: pytest.fail("resume must not send terminal pairs"))
    meta = json.loads((root / "run.json").read_text())
    assert meta["logical_results"] == 1500 and meta["extra_calls"] == 0
    assert "concurrency" not in b.parser().format_help()
    assert not (root / "in-flight.json").exists()


def test_failures_truncation_no_implicit_retries_and_explicit_retry_history(setup_run):
    args, manifest, root = setup_run
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            raise httpx.ReadTimeout("sensitive detail", request=request)
        if count == 2:
            return httpx.Response(429, text="retry later")
        if count == 3:
            return response(finish="length")
        if count == 4:
            return response("incidental A in reasoning")
        if count == 5:
            return httpx.Response(200, json={"wrong": "schema"})
        return response("Final answer: (B)")

    run(args, handler)
    assert count == 1500
    assert b.read_jsonl(root / "auto-cost.jsonl")[0]["status"] == "transport_error"
    assert b.read_jsonl(root / "auto-quality.jsonl")[0]["status"] == "truncated"
    truncated = b.read_jsonl(root / "auto-quality.jsonl")[0]
    assert truncated["prediction"] is None and truncated["correct"] is None
    assert b.read_jsonl(root / "auto-balanced.jsonl")[1]["status"] == "parse_error"
    malformed = b.read_jsonl(root / "auto-quality.jsonl")[1]
    assert malformed["raw_response"] == {"wrong": "schema"}
    run(args, lambda _: pytest.fail("must skip wrong/unscorable/errors by default"))
    args.retry_errors = True
    retried = []
    run(args, lambda request: (retried.append(request), response())[1])
    assert len(retried) == 2
    records = b.read_jsonl(root / "auto-cost.jsonl")
    assert records[-1]["attempt"] == 2
    assert records[-1]["previous_attempt_fingerprint"] == b.sha(b.canonical(records[0]))
    assert len(b.validated_results(records, manifest, "auto-cost")) == 500
    assert json.loads((root / "run.json").read_text())["extra_calls"] == 2


@pytest.mark.parametrize("status", [401, 403, 503])
def test_configuration_errors_stop_promptly_and_are_saved(setup_run, status):
    args, _, root = setup_run
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="configuration missing")

    with pytest.raises(RuntimeError, match="configuration/authentication"):
        run(args, handler)
    assert len(calls) == 1
    assert b.read_jsonl(root / "auto-cost.jsonl")[0]["http_status"] == status
    with pytest.raises(RuntimeError, match="Saved configuration/authentication"):
        run(
            args,
            lambda _: pytest.fail(
                "saved configuration failure must not be silently skipped"
            ),
        )


def test_interruption_journal_is_not_automatically_resent(setup_run):
    args, manifest, root = setup_run

    def interrupt(_):
        raise RuntimeError("simulate process exit after dispatch")

    with pytest.raises(RuntimeError, match="simulate process exit"):
        run(args, interrupt)
    assert (root / "in-flight.json").exists()
    # Emulate a partially written result as well as the outstanding journal.
    (root / "auto-cost.jsonl").write_text('{"partial":')
    calls = []
    run(args, lambda request: (calls.append(request), response())[1])
    assert len(calls) == 1499
    first = b.read_jsonl(root / "auto-cost.jsonl")[0]
    assert first["status"] == "interrupted" and first["correct"] is None
    args.retry_errors = True
    run(
        args,
        lambda _: pytest.fail("retry-errors must not resend ambiguous in-flight work"),
    )
    args.retry_interrupted = True
    calls.clear()
    run(args, lambda request: (calls.append(request), response())[1])
    assert len(calls) == 1
    assert b.read_jsonl(root / "auto-cost.jsonl")[-1]["attempt"] == 2


def test_resume_rejects_changed_settings_manifest_and_duplicates(setup_run):
    args, manifest, root = setup_run
    run(args, lambda _: response())
    args.max_tokens = 99
    with pytest.raises(ValueError, match="settings differ"):
        run(args, lambda _: pytest.fail("no requests"))
    args.max_tokens = 2048
    manifest["questions"][0]["question"] = "changed question"
    manifest["fingerprint"] = b.manifest_fingerprint(manifest)
    (root / "subset.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="settings differ"):
        run(args, lambda _: pytest.fail("no requests"))


def test_duplicate_attempt_chains_are_rejected(setup_run):
    _, manifest, _ = setup_run
    r = {
        "question_id": "0",
        "preset": "auto-cost",
        "manifest_fingerprint": manifest["fingerprint"],
        "settings_fingerprint": "settings",
        "status": "completed",
        "attempt": 1,
    }
    with pytest.raises(ValueError, match="Duplicate/conflicting"):
        b.validated_results([r, r], manifest, "auto-cost")


def test_output_lock_and_interrupted_final_line_repair(tmp_path):
    with b.output_lock(tmp_path):
        with pytest.raises(ValueError, match="Another benchmark writer"):
            with b.output_lock(tmp_path):
                pass
    path = tmp_path / "a.jsonl"
    path.write_bytes(b'{"a":1}\n{"broken":')
    assert b.read_jsonl(path, repair=True) == [{"a": 1}]
    assert path.read_bytes() == b'{"a":1}\n'
    path.write_bytes(b'{"a":1}')
    b.read_jsonl(path, repair=True)
    assert path.read_bytes().endswith(b"\n")
    path.write_bytes(b'broken\n{"a":1}\n')
    with pytest.raises(ValueError, match="Corrupt JSONL"):
        b.read_jsonl(path)


def test_gateway_credentials_url_sanitization_and_no_implicit_environment(
    setup_run, monkeypatch
):
    args, _, root = setup_run
    args.base_url = "http://name:secret@gateway.test/v1"
    with pytest.raises(ValueError, match="credentials"):
        run(args, lambda _: pytest.fail("no requests"))
    args.base_url = "http://gateway.test/v1?secret=hidden#token"
    args.api_key_env = "BENCH_GATEWAY_KEY"
    monkeypatch.setenv("BENCH_GATEWAY_KEY", "test-secret")

    def handler(request):
        assert request.headers["authorization"] == "Bearer test-secret"
        return httpx.Response(
            401,
            text="Authorization: Bearer test-secret https://a:b@private.test?q=secret",
        )

    with pytest.raises(RuntimeError):
        run(args, handler)
    assert "test-secret" not in (root / "auto-cost.jsonl").read_text()
    assert "secret=hidden" not in (root / "run.json").read_text()
    args.api_key_env = "LITELLM_MASTER_KEY"
    with pytest.raises(ValueError, match="dedicated gateway"):
        run(args, lambda _: pytest.fail("no requests"))


@pytest.fixture
def cache_contract(monkeypatch):
    # Simulate a future verified contract, not an assertion that upstream exposes one.
    monkeypatch.setattr(
        b,
        "VERIFIED_CACHE_INDICATORS",
        {"fixture-cache-hit": {"true": "hit", "false": "miss"}},
    )


def cost_result(hit="false", cost="0.2", **kwargs):
    r = {
        "preset": "auto-cost",
        "question_id": "0",
        "manifest_fingerprint": "manifest",
        "attempt": 1,
        "timestamp": "original",
        "returned_model": "model",
        "text": "Final answer: A",
        "finish_reason": "stop",
        "telemetry": {
            "fixture-cache-hit": hit,
            "x-litellm-response-cost": cost,
            "x-litellm-model-id": "deployment",
            "x-litellm-version": "version",
            "x-litellm-cache-key": "cache-key",
        },
    }
    r.update(kwargs)
    return r


def test_cache_reuse_across_presets_sessions_ledger_recovery(tmp_path, cache_contract):
    request = {
        "model": "auto-cost",
        "messages": [{"role": "user", "content": "Q"}],
        "temperature": 0,
        "max_tokens": 100,
    }
    ledger = b.Ledger(tmp_path / "ledger.json")
    original = cost_result()
    b.account_cost(original, request, ledger)
    ledger.rebuild([original])
    cached = cost_result("true", "0", preset="auto-quality")
    b.account_cost(cached, {**request, "model": "auto-quality"}, ledger)
    assert cached["reported_response_cost_usd"] == 0
    assert cached["estimated_uncached_generation_cost_usd"] == 0.2
    assert cached["cost_estimate_source"]["preset"] == "auto-cost"
    for corrupt in ('{"entries": []}', '{"broken":'):
        ledger.path.write_text(corrupt)
        recovered = b.Ledger(ledger.path)
        recovered.rebuild([original, cached])
        new = cost_result("true", "5")
        b.account_cost(new, request, recovered)
        assert new["estimated_uncached_generation_cost_usd"] == 0.2


@pytest.mark.parametrize(
    "change",
    ["answer", "model", "deployment", "version", "max_tokens", "prompt", "temperature"],
)
def test_cost_reuse_requires_exact_identity(tmp_path, cache_contract, change):
    ledger = b.Ledger(tmp_path / "ledger.json")
    request = {"messages": [], "temperature": 0, "max_tokens": 100}
    original = cost_result()
    b.account_cost(original, request, ledger)
    ledger.rebuild([original])
    cached = cost_result("true", "0")
    changed_request = copy.deepcopy(request)
    if change == "answer":
        cached["text"] = "B"
    elif change == "model":
        cached["returned_model"] = "other"
    elif change == "deployment":
        cached["telemetry"]["x-litellm-model-id"] = "other"
    elif change == "version":
        cached["telemetry"]["x-litellm-version"] = "other"
    elif change == "max_tokens":
        changed_request["max_tokens"] = 101
    elif change == "prompt":
        changed_request["messages"] = [{"role": "user", "content": "other"}]
    elif change == "temperature":
        changed_request["temperature"] = 1
    b.account_cost(cached, changed_request, ledger)
    assert cached["estimated_uncached_generation_cost_usd"] is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0", 0),
        ("bad", None),
        (None, None),
        ("nan", None),
        ("inf", None),
        ("-1", None),
        (True, None),
    ],
)
def test_cost_numbers(value, expected):
    assert b.cost_number(value) == expected


def test_unknown_status_and_preexisting_hits_do_not_invent_original_cost(
    tmp_path, cache_contract
):
    ledger = b.Ledger(tmp_path / "ledger.json")
    for hit, cost in [
        ("true", "0"),
        ("true", "4"),
        ("unknown", "4"),
        ("nonsense", "0"),
    ]:
        r = cost_result(hit, cost)
        b.account_cost(r, {"messages": [], "temperature": 0, "max_tokens": 1}, ledger)
        assert r["estimated_uncached_generation_cost_usd"] is None


def test_multiple_original_observations_keep_first_dispatch_provenance_after_resume(
    tmp_path, cache_contract
):
    ledger = b.Ledger(tmp_path / "ledger.json")
    request = {"messages": [], "temperature": 0, "max_tokens": 1}
    first = cost_result(cost=".2", preset="auto-balanced", dispatch_sequence=1)
    second = cost_result(cost=".4", preset="auto-cost", dispatch_sequence=2)
    for original in (first, second):
        b.account_cost(original, request, ledger)
        entry = original["generation_observation"]
        ledger.add(entry["key"], entry)
    hit = cost_result("true", "0")
    b.account_cost(hit, request, ledger)
    assert hit["estimated_uncached_generation_cost_usd"] == 0.2
    rebuilt = b.Ledger(ledger.path)
    rebuilt.rebuild([second, first])
    resumed_hit = cost_result("true", "0")
    b.account_cost(resumed_hit, request, rebuilt)
    assert resumed_hit["estimated_uncached_generation_cost_usd"] == 0.2
    assert resumed_hit["cost_estimate_source"] == hit["cost_estimate_source"]


def test_corrupt_original_observation_cannot_retarget_cost(tmp_path, cache_contract):
    ledger = b.Ledger(tmp_path / "ledger.json")
    request = {"messages": [], "temperature": 0, "max_tokens": 1}
    original = cost_result()
    b.account_cost(original, request, ledger)
    original["text"] = "Changed answer"
    ledger.rebuild([original])
    hit = cost_result("true", "0")
    b.account_cost(hit, request, ledger)
    assert hit["estimated_uncached_generation_cost_usd"] is None


@pytest.mark.parametrize(
    "body",
    [
        '{"api_key":"provider-secret","password":"password-secret","url":"HTTPS://user:pass@private.test/v1"}',
        '{"error":{"credentials":{"token":"nested-secret"},"message":"Bearer bearer-secret"}}',
        "<html>api_key='quoted-secret' password: plain-secret HTTPS://user:pass@private.test/v1</html>",
    ],
)
def test_quoted_and_nested_http_error_secrets_never_persist(setup_run, body):
    args, _, root = setup_run
    with pytest.raises(RuntimeError):
        run(args, lambda _: httpx.Response(401, text=body))
    saved = (root / "auto-cost.jsonl").read_text() + (root / "run.json").read_text()
    for secret in (
        "provider-secret",
        "password-secret",
        "user:pass",
        "nested-secret",
        "bearer-secret",
        "quoted-secret",
        "plain-secret",
    ):
        assert secret not in saved


@pytest.mark.parametrize("options_count", range(2, 11))
def test_parser_valid_range_and_final_line_only(options_count):
    options = [str(i) for i in range(options_count)]
    letter = chr(64 + options_count)
    assert b.prediction_from_text(
        f"Reasoning\nFinal answer: ({letter})\n", options
    ) == (letter, "final-marker")
    for text in (
        f"Final answer: ({letter})\nMore reasoning",
        "Final answer: (A",
        "Final answer: A)",
        "(A)",
        "A.",
        "Final answer: (Z)\nFinal answer: (A)",
    ):
        assert b.prediction_from_text(text, options)[0] is None


def test_production_does_not_trust_unverified_hit_header_or_response_ids(tmp_path):
    ledger = b.Ledger(tmp_path / "ledger.json")
    result = cost_result()
    result["telemetry"]["x-litellm-cache-hit"] = "false"
    b.account_cost(result, {"messages": [], "temperature": 0, "max_tokens": 1}, ledger)
    assert result["cache_status"] == "unknown"
    assert result["estimated_uncached_generation_cost_usd"] is None


def test_zero_observed_original_is_distinct_from_unknown(tmp_path, cache_contract):
    ledger = b.Ledger(tmp_path / "ledger.json")
    request = {"messages": [], "temperature": 0, "max_tokens": 1}
    original = cost_result(cost="0")
    b.account_cost(original, request, ledger)
    ledger.rebuild([original])
    hit = cost_result("true", "0")
    b.account_cost(hit, request, ledger)
    assert hit["estimated_uncached_generation_cost_usd"] == 0
    assert hit["cost_estimate_source"] is not None
    assert json.loads(ledger.path.read_text())["cache_key_index"]["cache-key"]


@pytest.mark.parametrize(
    "content", [b'{"choices": NaN}', b'{"choices": []}', b"not-json"]
)
def test_malformed_bodies_are_saved_without_aborting_durability(setup_run, content):
    args, _, root = setup_run
    calls = []

    def handler(request):
        calls.append(request)
        return (
            httpx.Response(200, content=content)
            if len(calls) == 1
            else httpx.Response(401, text="stop fixture")
        )

    with pytest.raises(RuntimeError):
        run(args, handler)
    result = b.read_jsonl(root / "auto-cost.jsonl")[0]
    assert result["status"] == "malformed_response" and result["correct"] is None
    assert "raw_response" in result and result["elapsed_seconds"] >= 0

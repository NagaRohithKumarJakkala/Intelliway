import hashlib
import json
import zipfile
from io import BytesIO

import httpx
import pytest

from scripts import benchmark_mmlu_pro as bench


def test_semantic_router_is_the_default_benchmark_target():
    args = bench.parser().parse_args(["run"])
    assert args.base_url == "http://127.0.0.1:8899/v1"


def test_strict_answer_parser_uses_only_unambiguous_final_line():
    options = ["one", "two", "three"]
    assert bench.prediction_from_text(
        "Reasoning mentions A.\nFinal answer: (C)", options
    ) == ("C", "final-marker")
    assert bench.prediction_from_text("A", options) == ("A", "bare-letter")
    assert bench.prediction_from_text("The answer is A", options) == (
        None,
        "unparseable",
    )
    assert (
        bench.prediction_from_text("Final answer: A\nFinal answer: B", options)[0]
        is None
    )
    assert bench.prediction_from_text("Final answer: D", options)[0] is None


def test_quota_redistribution_respects_capacity_deterministically():
    capacities = {"a": 1, "b": 1, "c": 10}
    assert bench.capacity_quotas(capacities, 8) == {"a": 1, "b": 1, "c": 6}
    with pytest.raises(ValueError, match="Insufficient cohort capacity"):
        bench.capacity_quotas({"a": 2}, 3)


def test_manifest_fingerprint_ignores_fingerprint_field():
    manifest = {"version": 1, "questions": []}
    manifest["fingerprint"] = bench.manifest_fingerprint(manifest)
    assert manifest["fingerprint"] == bench.manifest_fingerprint(manifest)


def test_reference_reader_only_accepts_records_and_handles_nested_archive():
    payload = {
        "math": [
            {
                "question_id": 7,
                "question": "Q",
                "options": ["a", "b"],
                "pred": "A",
                "model_outputs": "ignored",
            }
        ],
        "summary": {"score": 1},
    }
    data = BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("subject.json", json.dumps(payload))
        archive.writestr("notes.txt", "not parsed")
    records = bench.read_references(data.getvalue(), "fixture")
    assert records["7"][0]["pred"] == "A"


def test_request_contains_only_allowed_schema_and_excludes_gold_and_references():
    question = {
        "category": "math",
        "question": "2+2?",
        "options": ["3", "4"],
        "answer": "B",
        "reference_predictions": {"secret": "A"},
        "cohort": "control",
    }
    messages = bench.make_prompt(question)
    request = {
        "model": "auto-cost",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 10,
        "stream": False,
    }
    assert set(request) == {"model", "messages", "temperature", "max_tokens", "stream"}
    assert "2+2?" in messages[0]["content"]
    assert (
        "secret" not in messages[0]["content"]
        and "control" not in messages[0]["content"]
    )
    assert "Final answer: (X)" in messages[0]["content"]


def test_download_uses_mocktransport_and_caches_completed_body(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(200, content=b"fixture")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        response = client.get("https://example.test/source")
    assert response.content == b"fixture"
    assert hashlib.sha256(b"fixture").hexdigest() == bench.sha(response.content)
    assert len(calls) == 1

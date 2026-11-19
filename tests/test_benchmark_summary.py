import argparse
import csv
import json

import pytest

from scripts import benchmark_mmlu_pro as bench


@pytest.fixture
def manifest():
    value = {
        "questions": [
            {
                "question_id": str(i),
                "cohort": "separating" if i < 400 else "control",
                "category": "math" if i < 400 else "history",
                "question": f"Question {i}",
                "options": ["one", "two"],
                "answer": "A",
            }
            for i in range(500)
        ]
    }
    value["fingerprint"] = bench.manifest_fingerprint(value)
    return value


def record(manifest, qid, preset="auto-cost", **fields):
    q = manifest["questions"][qid]
    return {
        "question_id": str(qid),
        "preset": preset,
        "manifest_fingerprint": manifest["fingerprint"],
        "settings_fingerprint": "synthetic-settings",
        "cohort": q["cohort"],
        "category": q["category"],
        "attempt": 1,
        "status": "completed",
        "correct": True,
        **fields,
    }


def save(root, preset, records):
    bench.result_path(root, preset).write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )


def comparison(report, population, cohort="all", right="auto-balanced"):
    return next(
        row
        for row in report["comparison"]
        if row["left"] == "auto-cost"
        and row["right"] == right
        and row["population"] == population
        and row["cohort"] == cohort
    )


def test_pending_denominators_errors_and_coverage(tmp_path, manifest):
    records = [
        record(
            manifest,
            0,
            elapsed_seconds=1,
            returned_model="model-a",
            usage={"prompt_tokens": 10, "completion_tokens": 2, "details": {}},
            cache_status="miss",
            reported_response_cost_usd=0,
            estimated_uncached_generation_cost_usd=0.02,
        ),
        record(
            manifest,
            1,
            correct=False,
            elapsed_seconds=9,
            cache_status="hit",
            usage={"prompt_tokens": 4, "completion_tokens": None},
            reported_response_cost_usd=0.01,
        ),
        record(manifest, 2, correct=None, status="http_error_retryable"),
        record(manifest, 3, correct=None, status="parse_error"),
        record(manifest, 4, correct=None, status="truncated"),
        record(manifest, 5, correct=None, status="malformed_response"),
        record(manifest, 6, correct=None, status="transport_error"),
    ]
    save(tmp_path, "auto-cost", records)
    report = bench.summary(tmp_path, manifest)
    m = report["metrics"]["auto-cost"]
    assert (m["scheduled"], m["recorded"], m["pending"]) == (500, 7, 493)
    assert not m["complete"]
    assert m["accuracy_all_scheduled"] == 1 / 500
    assert m["scorable_accuracy"] == 0.5
    assert m["scorable"] == 2
    for key in (
        "http_errors",
        "parse_errors",
        "truncation_errors",
        "transport_errors",
        "malformed_response_errors",
    ):
        assert m[key] == 1
    assert m["cache_counts"] == {"hit": 1, "miss": 1, "unknown": 5}
    assert m["returned_models"] == {"model-a": 1, "unknown": 6}
    assert m["usage"] == {"prompt_tokens": 14, "completion_tokens": 2}
    assert m["usage_coverage"] == {"prompt_tokens": 2, "completion_tokens": 1}
    assert m["median_latency_seconds"] == 5
    assert m["p95_latency_seconds"] == 9
    assert m["latency_coverage"] == 2
    assert m["reported_response_cost_usd"] is None
    assert m["reported_cost_partial_sum_usd"] == 0.01
    assert m["reported_cost_coverage"] == 2
    assert m["estimated_cost_partial_sum_usd"] == 0.02
    assert m["estimated_cost_coverage"] == 1
    assert m["estimated_cost_per_correct_usd"] is None
    assert m["cohorts"]["control"]["pending"] == 100
    assert m["categories"]["history"]["pending"] == 100
    assert m["category_cohort"]["control/history"]["pending"] == 100
    assert report["metrics"]["auto-quality"]["pending"] == 500


def test_all_scheduled_and_jointly_scorable_pairs(tmp_path, manifest):
    save(
        tmp_path,
        "auto-cost",
        [
            record(manifest, 0),
            record(manifest, 1, correct=False),
            record(manifest, 400),
        ],
    )
    save(
        tmp_path,
        "auto-balanced",
        [
            record(manifest, 0, "auto-balanced", correct=False),
            record(manifest, 1, "auto-balanced", correct=True),
            record(manifest, 2, "auto-balanced", correct=True),
            record(manifest, 400, "auto-balanced", correct=None, status="truncated"),
        ],
    )
    report = bench.summary(tmp_path, manifest, seed=17)
    assert len(report["comparison"]) == 18
    all_pairs = comparison(report, "all_scheduled")
    assert all_pairs["paired"] == 500
    assert all_pairs["accuracy_delta_pp"] == 0
    assert (all_pairs["left_only_correct"], all_pairs["right_only_correct"]) == (2, 2)
    joint = comparison(report, "jointly_scorable")
    assert joint["paired"] == 2
    assert joint["paired_coverage"] == 2 / 500
    assert joint["accuracy_delta_pp"] == 0
    assert comparison(report, "all_scheduled", "control")["accuracy_delta_pp"] == -1
    assert comparison(report, "jointly_scorable", "control")["bootstrap_95pp"] is None
    assert (
        comparison(report, "all_scheduled", "separating")["accuracy_delta_pp"] == 0.25
    )
    assert (
        all_pairs["bootstrap_95pp"]
        == bench.summary(tmp_path, manifest, seed=17)["comparison"][0]["bootstrap_95pp"]
    )


def test_latest_attempt_counted_once_and_extra_calls(tmp_path, manifest):
    first = record(manifest, 0, status="transport_error", correct=None)
    latest = record(
        manifest,
        0,
        attempt=2,
        previous_attempt_fingerprint=bench.sha(bench.canonical(first)),
    )
    save(tmp_path, "auto-cost", [first, latest])
    m = bench.summary(tmp_path, manifest)["metrics"]["auto-cost"]
    assert (m["recorded"], m["correct"], m["attempts"], m["extra_calls"]) == (
        1,
        1,
        2,
        1,
    )
    assert m["transport_errors"] == 0
    assert m["cohorts"]["separating"]["extra_calls"] == 1


def test_complete_costs_zero_and_missing_values(tmp_path, manifest):
    records = [
        record(
            manifest,
            i,
            correct=i == 0,
            reported_response_cost_usd=0,
            estimated_uncached_generation_cost_usd=0.01,
        )
        for i in range(500)
    ]
    save(tmp_path, "auto-cost", records)
    m = bench.summary(tmp_path, manifest)["metrics"]["auto-cost"]
    assert m["complete"]
    assert m["reported_response_cost_usd"] == 0
    assert m["reported_cost_per_correct_usd"] == 0
    assert m["estimated_uncached_generation_cost_usd"] == pytest.approx(5)
    assert m["estimated_cost_per_correct_usd"] == pytest.approx(5)
    records[-1]["reported_response_cost_usd"] = -1
    records[-1]["estimated_uncached_generation_cost_usd"] = None
    save(tmp_path, "auto-cost", records)
    m = bench.summary(tmp_path, manifest)["metrics"]["auto-cost"]
    assert m["complete"]
    assert m["reported_response_cost_usd"] is None
    assert m["estimated_uncached_generation_cost_usd"] is None
    assert m["estimated_cost_coverage"] == 499


@pytest.mark.parametrize("change", ["manifest", "settings", "duplicate", "question"])
def test_rejects_conflicting_records(tmp_path, manifest, change):
    r = record(manifest, 0)
    records = [r]
    if change == "manifest":
        r["manifest_fingerprint"] = "other"
    elif change == "settings":
        save(
            tmp_path,
            "auto-balanced",
            [record(manifest, 0, "auto-balanced", settings_fingerprint="other")],
        )
    elif change == "question":
        r["question_id"] = "foreign"
    else:
        records.append(r.copy())
    save(tmp_path, "auto-cost", records)
    with pytest.raises(ValueError):
        bench.summary(tmp_path, manifest)


def test_offline_csv_and_console_include_cohorts(
    tmp_path, manifest, monkeypatch, capsys
):
    monkeypatch.setattr(bench, "load_manifest", lambda path: manifest)
    bench.summarize(argparse.Namespace(manifest="unused", output_dir=tmp_path, seed=42))
    report = json.loads((tmp_path / "summary.json").read_text())
    with (tmp_path / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 21
    assert {r["group"] for r in rows} >= {
        "all",
        "control",
        "separating",
        "control/history",
    }
    assert {
        "cache_counts",
        "usage_coverage",
        "parse_errors",
        "estimated_cost_partial_sum_usd",
        "reported_cost_per_correct_usd",
        "p95_latency_seconds",
    } <= rows[0].keys()
    with (tmp_path / "comparison.csv").open() as f:
        pairs = list(csv.DictReader(f))
    assert len(pairs) == 18
    assert {r["population"] for r in pairs} == {"all_scheduled", "jointly_scorable"}
    console = capsys.readouterr().out
    assert "separating" in console and "control" in console
    assert "not audited bills" in console
    assert "not representative" in report["interpretation"]


def test_bootstrap_empty_constant_and_deterministic():
    assert bench.bootstrap_mean([], 42) is None
    assert bench.bootstrap_mean([-1] * 100, 42) == [-100, -100]
    assert bench.bootstrap_mean([0] * 100, 42) == [0, 0]
    values = [-1, 0, 1, 1]
    assert bench.bootstrap_mean(values, 7) == bench.bootstrap_mean(values, 7)
    low, high = bench.bootstrap_mean(values, 7)
    assert -100 <= low <= high <= 100
    with pytest.raises(ValueError):
        bench.bootstrap_mean(values, 42, rounds=0)

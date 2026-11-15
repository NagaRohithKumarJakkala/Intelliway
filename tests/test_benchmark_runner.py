import json
import asyncio

from benchmarks import runner


def test_run_one_records_metrics_and_judgement(monkeypatch):
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "judge":
            return {
                "choices": [{"message": {"content": json.dumps({
                    "quality_label": "excellent",
                    "score": 5,
                    "reason": "meets all criteria",
                    "criteria_results": [True],
                })}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        return {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3},
        }

    monkeypatch.setattr(runner, "send_to_litellm", fake_send)
    result = asyncio.run(runner._run_one(
        {"id": "q1", "prompt": "Question", "criteria": ["Correct"]},
        "candidate",
        "judge",
        {
            "candidate": {
                "input_per_million_tokens": 1,
                "output_per_million_tokens": 2,
            },
            "judge": {
                "input_per_million_tokens": 3,
                "output_per_million_tokens": 4,
            },
        },
    ))

    assert result["status"] == "success"
    assert result["label_status"] == "success"
    assert result["judgement"]["quality_label"] == "excellent"
    assert result["cost"] == 0.00001
    assert result["judge_cost"] == 0.00005
    assert [call["model"] for call in calls] == ["candidate", "judge"]


def test_run_one_keeps_provider_failure_unlabeled(monkeypatch):
    async def fail_send(**kwargs):
        raise TimeoutError("provider unavailable")

    monkeypatch.setattr(runner, "send_to_litellm", fail_send)
    result = asyncio.run(runner._run_one(
        {"id": "q1", "prompt": "Question", "criteria": ["Correct"]},
        "candidate",
        "judge",
        {},
    ))

    assert result["status"] == "error"
    assert "label_status" not in result
    assert result["error_class"] == "TimeoutError"

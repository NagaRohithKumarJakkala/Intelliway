import copy
import json
import zipfile
from collections import Counter
from io import BytesIO
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from scripts import benchmark_mmlu_pro as bench


def parquet(rows):
    stream = BytesIO()
    pd.DataFrame(rows).to_parquet(stream, index=False)
    return stream.getvalue()


def zipped(payload, name="results.json", extra=None):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, json.dumps(payload))
        for member, content in (extra or {}).items():
            archive.writestr(member, content)
    return stream.getvalue()


@pytest.fixture
def sources(tmp_path):
    rows, references = [], [[] for _ in range(4)]
    for category, separating, controls in (
        ("a", 5, 2),
        ("b", 200, 100),
        ("c", 400, 200),
    ):
        for cohort, count in (("separating", separating), ("control", controls)):
            for i in range(count):
                qid = len(rows)
                row = dict(
                    question_id=qid,
                    question=f"Question {qid}",
                    options=["first", "second"],
                    answer="A",
                    answer_index=0,
                    category=category,
                    src="synthetic",
                    cot_content="SECRET",
                )
                rows.append(row)
                for ref in range(4):
                    references[ref].append(
                        {
                            **copy.deepcopy(row),
                            "pred": "B" if cohort == "separating" and ref < 2 else "A",
                            "model_outputs": "SECRET reference reasoning",
                        }
                    )
    return rows, references


def prepare(tmp_path, sources, seed=42, reverse=False):
    rows, references = sources
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    (cache / bench.sha(bench.DATA_URL.encode())).write_bytes(
        parquet(rows[::-1] if reverse else rows)
    )
    for i, url in enumerate(sum(bench.REFERENCES.values(), [])):
        records = references[i][::-1] if reverse else references[i]
        # Both actual layouts: flat lists and subject-keyed lists with summary metadata.
        payload = records if i < 3 else {"subject": records, "summary": {"acc": 1}}
        (cache / bench.sha(url.encode())).write_bytes(zipped(payload))
    return bench.prepare(
        SimpleNamespace(
            output_dir=str(tmp_path), manifest=str(tmp_path / "subset.json"), seed=seed
        )
    )


def test_exact_500_deterministic_waterfill_ranking_order_and_manifest(
    tmp_path, sources
):
    manifest = prepare(tmp_path, sources)
    assert manifest == prepare(tmp_path, sources)
    questions = manifest["questions"]
    assert len(questions) == len({q["question_id"] for q in questions}) == 500
    assert Counter(q["cohort"] for q in questions) == {
        "separating": 400,
        "control": 100,
    }
    assert manifest["selection"]["quotas"] == {
        "separating": {"a": 5, "b": 198, "c": 197},
        "control": {"a": 2, "b": 49, "c": 49},
    }
    assert all(
        "cot_content" not in q and "model_outputs" not in q and q["src"] == "synthetic"
        for q in questions
    )
    hashes = [
        bench.stable_hash(42, q["question_id"], "evaluation-order") for q in questions
    ]
    assert hashes == sorted(hashes)
    candidates = [str(r["question_id"]) for r in sources[0] if r["category"] == "b"][
        :200
    ]
    expected = sorted(candidates, key=lambda q: bench.stable_hash(42, q, "separating"))[
        :198
    ]
    assert {
        q["question_id"]
        for q in questions
        if q["category"] == "b" and q["cohort"] == "separating"
    } == set(expected)
    other = tmp_path / "reordered"
    other.mkdir()
    reordered = prepare(other, sources, reverse=True)
    assert reordered["questions"] == questions
    assert reordered["selection"] == manifest["selection"]
    assert manifest["fingerprint"] == bench.manifest_fingerprint(manifest)
    with pytest.raises(ValueError, match="immutable manifest"):
        prepare(tmp_path, sources, seed=43)


def test_drift_duplicates_missing_and_updated_gold(tmp_path, sources):
    rows, refs = sources
    refs[0][0]["question"] += " changed"
    refs[0][1]["options"].reverse()
    refs[0].append(copy.deepcopy(refs[0][2]))
    refs[0].append({**refs[0][3], "pred": "A"})
    del refs[0][4]["pred"]
    refs[0][5]["pred"] = "Z"
    refs[0] = [r for r in refs[0] if r["question_id"] != 6]
    refs[0].append({**refs[0][0], "question_id": 99999})
    # Historical gold/correctness must not be used. This control becomes separating.
    rows[5]["answer"], rows[5]["answer_index"] = "B", 1
    for ref in refs:
        for rec in ref:
            if rec["question_id"] == 5:
                rec["pred"] = "B" if ref is refs[2] or ref is refs[3] else "A"
                rec["correct"] = False
    manifest = prepare(tmp_path, sources)
    diagnostics = manifest["selection"]
    counts = diagnostics["exclusions"][bench.REFERENCES["economical"][0]]
    assert counts == dict(
        source_drift=2,
        duplicate_or_conflict=2,
        missing_prediction=1,
        invalid_prediction=0,
        missing_record=1,
        unknown_question=1,
    )
    assert diagnostics["eligible_count"] == len(rows) - 6
    assert diagnostics["eligible_cohorts"]["separating"]["gap"] == 1
    assert diagnostics["selected_accuracy"]["control"]["gap"] == 0
    assert (
        diagnostics["reference_coverage"][bench.REFERENCES["economical"][0]]
        == len(rows) - 6
    )
    assert not {"0", "1", "2", "3", "4", "6"} & {
        q["question_id"] for q in manifest["questions"]
    }
    changed = next(q for q in manifest["questions"] if q["question_id"] == "5")
    assert changed["answer"] == "B" and changed["gap"] == 1


@pytest.mark.parametrize("cohort", ["separating", "control", "all"])
def test_insufficient_coverage_never_changes_targets(tmp_path, sources, cohort):
    rows, refs = sources
    if cohort == "all":
        refs[0][:] = refs[0][:100]
    else:
        for ref in refs:
            for rec in ref:
                if (
                    cohort == "separating" and rec["pred"] == "B"
                ) or cohort == "control":
                    rec["pred"] = "A" if cohort == "separating" else "B"
    with pytest.raises(ValueError, match="Exclusions"):
        prepare(tmp_path, sources)
    assert not (tmp_path / "subset.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("question_id", None),
        ("question_id", True),
        ("question", " "),
        ("category", ""),
        ("options", ["one"]),
        ("options", ["", "two"]),
        ("answer_index", 0.5),
        ("answer_index", True),
        ("answer_index", "0"),
        ("answer", "a"),
        ("answer", "B"),
    ],
)
def test_strict_dataset_validation(sources, field, value):
    row = copy.deepcopy(sources[0][0])
    row[field] = value
    with pytest.raises(ValueError):
        bench.dataset_rows(parquet([row]))


def test_dataset_duplicate_empty_missing_schema_and_option_order(sources):
    row = sources[0][0]
    with pytest.raises(ValueError, match="Duplicate"):
        bench.dataset_rows(parquet([row, row]))
    with pytest.raises(ValueError, match="empty"):
        stream = BytesIO()
        pd.DataFrame([row]).iloc[:0].to_parquet(stream, index=False)
        bench.dataset_rows(stream.getvalue())
    with pytest.raises(ValueError, match="missing fields"):
        bench.dataset_rows(parquet([{"question_id": 1}]))
    assert bench.dataset_rows(parquet([row]))[0]["options"] == row["options"]


def test_gap_precedes_hash_and_conservative_source_normalization(tmp_path, sources):
    rows, refs = sources
    # Category b has exactly two more separators than its quota.
    low_gap = [r for r in refs[0] if r["category"] == "b"][:2]
    for record in low_gap:
        record["pred"] = "A"
    for ref in refs:
        ref[0]["question"] = " \r\n" + ref[0]["question"] + " \r"
        ref[0]["options"] = [" " + option + "\r\n" for option in ref[0]["options"]]
    manifest = prepare(tmp_path, sources)
    assert not {str(r["question_id"]) for r in low_gap} & {
        q["question_id"] for q in manifest["questions"]
    }
    assert manifest["selection"]["eligible_count"] == len(rows)
    assert all(
        c["source_drift"] == 0 for c in manifest["selection"]["exclusions"].values()
    )


def test_invalid_prediction_diagnostics(tmp_path, sources):
    sources[1][0][0]["pred"] = "Z"
    manifest = prepare(tmp_path, sources)
    counts = manifest["selection"]["exclusions"][bench.REFERENCES["economical"][0]]
    assert counts["invalid_prediction"] == 1
    assert manifest["selection"]["eligible_count"] == len(sources[0]) - 1


def test_waterfill_edge_cases():
    assert bench.capacity_quotas({"c": 100, "b": 100, "a": 1}, 8) == {
        "a": 1,
        "b": 4,
        "c": 3,
    }
    assert bench.capacity_quotas({"a": 0, "b": 100, "c": 100}, 7) == {
        "a": 0,
        "b": 4,
        "c": 3,
    }
    assert bench.capacity_quotas({}, 0) == {}
    for capacities, target in [
        ({}, 1),
        ({"a": -1}, 0),
        ({"a": 1}, -1),
        ({"a": True}, 0),
    ]:
        with pytest.raises(ValueError):
            bench.capacity_quotas(capacities, target)


@pytest.mark.parametrize(
    "name",
    ["../results.json", "/results.json", "C:/results.json", "x\\..\\results.json"],
)
def test_reject_unsafe_archive_members(name, sources):
    with pytest.raises(ValueError, match="Unsafe"):
        bench.read_references(zipped(sources[1][0], name), "fixture")


def test_archive_schema_metadata_and_size_limits(monkeypatch, sources):
    record = sources[1][0][0]
    data = zipped(
        [record],
        extra={
            "summary.json": json.dumps({"metadata": record}),
            "__MACOSX/._x.json": "garbage",
        },
    )
    assert list(bench.read_references(data, "fixture")) == ["0"]
    with pytest.raises(ValueError, match="No validated"):
        bench.read_references(zipped({"summary": record}), "fixture")
    with pytest.raises(ValueError, match="Invalid reference JSON"):
        bench.read_references(zipped([record], extra={"broken.json": "{"}), "fixture")
    monkeypatch.setattr(bench, "MAX_MEMBER", 2)
    with pytest.raises(ValueError, match="size limit"):
        bench.read_references(data, "fixture")
    monkeypatch.setattr(bench, "MAX_MEMBER", 1000000)
    monkeypatch.setattr(bench, "MAX_UNCOMPRESSED", 2)
    with pytest.raises(ValueError, match="size limit"):
        bench.read_references(data, "fixture")


def test_download_cache_and_limits(tmp_path, monkeypatch):
    real_client = httpx.Client
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=b"complete body")

    monkeypatch.setattr(
        bench.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    url = "https://example.test/artifact"
    assert bench.download(url, tmp_path) == (
        b"complete body",
        bench.sha(b"complete body"),
    )
    assert bench.download(url, tmp_path)[0] == b"complete body"
    assert len(calls) == 1
    monkeypatch.setattr(bench, "MAX_DOWNLOAD", 3)
    with pytest.raises(ValueError, match="Cached download"):
        bench.download(url, tmp_path)
    with pytest.raises(ValueError, match="Download exceeds"):
        bench.download(url + "2", tmp_path)
    assert not (tmp_path / bench.sha((url + "2").encode())).exists()
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="HTTPS"):
        bench.download("http://example.test/artifact", tmp_path)

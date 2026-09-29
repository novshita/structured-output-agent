import json

from agent.evaluate import render_markdown, run_eval
from agent.llm import FakeLLM

from .conftest import VALID, as_json


def test_eval_computes_metrics_and_renders(tmp_path):
    (tmp_path / "labels").mkdir()
    (tmp_path / "01.txt").write_text("posting one")
    (tmp_path / "02.txt").write_text("posting two")
    (tmp_path / "labels" / "01.json").write_text(json.dumps(VALID))

    # Order of calls: 01 baseline, 01 full, 02 baseline, 02 full (+ retry).
    llm = FakeLLM([
        f"```json\n{as_json()}\n```",
        as_json(),
        "Sorry, no JSON",
        as_json(work_mode="remote-ish"), as_json(title="Other"),
    ])
    reports = run_eval(tmp_path, llm)
    base, full = reports["baseline"], reports["full"]

    assert base.final_rate() == 0.5
    assert full.first_try_rate() == 0.5
    assert full.final_rate() == 1.0
    assert full.mean_attempts_per_success() == 1.5
    assert full.field_accuracy("company") == 1.0
    assert full.failure_breakdown()["schema_enum"] == 1

    md = render_markdown(reports)
    assert "first-try validity 50% → final validity 100%" in md
    assert "| Final validity rate | 50% | 100% | +50 pp |" in md

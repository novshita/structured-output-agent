"""Evaluation harness: baseline vs. full agent over a directory of postings."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .extractor import Mode, extract
from .llm import LLM
from .logger import AttemptLogger
from .schemas import ExtractionResult, JobPosting

FIELDS = ["title", "company", "work_mode", "salary"]


@dataclass
class ModeReport:
    mode: Mode
    results: dict[str, ExtractionResult] = field(default_factory=dict)
    field_hits: Counter = field(default_factory=Counter)
    labeled: int = 0

    @property
    def n(self) -> int:
        return len(self.results)

    def first_try_rate(self) -> float:
        return _rate(sum(r.ok and r.calls[0].success for r in self.results.values()), self.n)

    def final_rate(self) -> float:
        return _rate(sum(r.ok for r in self.results.values()), self.n)

    def mean_attempts_per_success(self) -> Optional[float]:
        ok = [r.attempts for r in self.results.values() if r.ok]
        return sum(ok) / len(ok) if ok else None

    def avg_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results.values()) / max(self.n, 1)

    def avg_tokens(self) -> float:
        return sum(r.tokens_in + r.tokens_out for r in self.results.values()) / max(self.n, 1)

    def failure_breakdown(self) -> Counter:
        return Counter(c.error_type for r in self.results.values() for c in r.calls if c.error_type)

    def field_accuracy(self, name: str) -> Optional[float]:
        return self.field_hits[name] / self.labeled if self.labeled else None


def _rate(k: int, n: int) -> float:
    return k / n if n else 0.0


def _norm(s: Optional[str]) -> str:
    return re.sub(r"[^\w]+", " ", (s or "").casefold()).strip()


def field_matches(name: str, predicted: JobPosting, label: dict) -> bool:
    if name in ("title", "company"):
        return _norm(getattr(predicted, name)) == _norm(label.get(name))
    if name == "work_mode":
        return predicted.work_mode == label.get("work_mode")
    if name == "salary":
        want = label.get("salary")
        got = predicted.salary.model_dump() if predicted.salary else None
        if not want or not got:
            return not want and not got
        return all(got.get(k) == want.get(k) for k in ("min", "max", "currency", "period"))
    raise ValueError(name)


def run_eval(
    sample_dir: Path,
    llm: LLM,
    *,
    modes: tuple[Mode, ...] = ("baseline", "full"),
    logger: Optional[AttemptLogger] = None,
    max_retries: Optional[int] = None,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[Mode, ModeReport]:
    docs = sorted(sample_dir.glob("*.txt"))
    if not docs:
        raise FileNotFoundError(f"no .txt samples in {sample_dir}")
    labels_dir = sample_dir / "labels"
    reports = {m: ModeReport(m) for m in modes}

    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        label_path = labels_dir / f"{doc.stem}.json"
        label = json.loads(label_path.read_text(encoding="utf-8")) if label_path.exists() else None
        for mode in modes:
            res = extract(text, llm, mode=mode, max_retries=max_retries, logger=logger or False)
            rep = reports[mode]
            rep.results[doc.name] = res
            if label is not None:
                rep.labeled += 1
                if res.ok:
                    for f in FIELDS:
                        rep.field_hits[f] += field_matches(f, res.posting, label)
            progress(f"{doc.name} [{mode}] {'ok' if res.ok else res.last_error} "
                     f"(attempts={res.attempts})")
    return reports


# --- Markdown report -------------------------------------------------------

ERROR_TYPES = ["json_parse", "schema_missing_field", "schema_type", "schema_enum", "cross_field",
               "tool_response_invalid", "llm_error", "max_retries_exceeded"]


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _delta_pct(a: Optional[float], b: Optional[float]) -> str:
    return "n/a" if a is None or b is None else f"{(b - a) * 100:+.0f} pp"


def render_markdown(reports: dict[Mode, ModeReport], model: str = "") -> str:
    base, full = reports.get("baseline"), reports.get("full")
    main = full or base
    lines = [
        "# Evaluation Results",
        "",
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}"
        + (f" with `{model}`" if model else "")
        + f" on {main.n} documents ({main.labeled} labelled)._",
        "",
    ]
    if full:
        lines += [f"**Headline:** first-try validity {_pct(full.first_try_rate())} → "
                  f"final validity {_pct(full.final_rate())}.", ""]

    def val(rep, fn):
        return fn(rep) if rep else None

    rows = [
        ("First-try validity rate", lambda r: r.first_try_rate(), _pct, True),
        ("Final validity rate", lambda r: r.final_rate(), _pct, True),
        ("Mean attempts per success", lambda r: r.mean_attempts_per_success(),
         lambda x: "n/a" if x is None else f"{x:.2f}", False),
        ("Avg latency per doc", lambda r: r.avg_latency_ms(), lambda x: f"{x / 1000:.2f} s", False),
        ("Avg tokens per doc", lambda r: r.avg_tokens(), lambda x: f"{x:,.0f}", False),
    ]
    lines += ["## Baseline vs. full agent", "",
              "| Metric | Baseline (no schema, no retries) | Full agent | Δ |",
              "|---|---|---|---|"]
    for name, fn, fmt, is_rate in rows:
        a, b = val(base, fn), val(full, fn)
        if is_rate:
            delta = _delta_pct(a, b)
        elif a is not None and b is not None and name != "Mean attempts per success":
            delta = f"{(b - a) / a:+.0%}" if a else "n/a"
        else:
            delta = "n/a"
        lines.append(f"| {name} | {fmt(a) if a is not None else 'n/a'} | "
                     f"{fmt(b) if b is not None else 'n/a'} | {delta} |")

    lines += ["", "## Field-level accuracy (vs. labels)", "",
              "_Failed extractions count as wrong._", "",
              "| Field | Baseline | Full agent |", "|---|---|---|"]
    for f in FIELDS:
        lines.append(f"| `{f}` | {_pct(val(base, lambda r: r.field_accuracy(f)))} | "
                     f"{_pct(val(full, lambda r: r.field_accuracy(f)))} |")

    lines += ["", "## Failure breakdown", ""]
    for rep in (base, full):
        if not rep:
            continue
        counts = rep.failure_breakdown()
        total = sum(counts.values())
        lines += [f"**{rep.mode}** ({total} failed calls)", "",
                  "| `error_type` | Count | % of failed calls |", "|---|---|---|"]
        for et in ERROR_TYPES:
            lines.append(f"| `{et}` | {counts[et]} | {_pct(_rate(counts[et], total)) if total else 'n/a'} |")
        lines.append("")

    lines += ["## Per-document results", "", "| Document | " +
              " | ".join(f"{m}" for m in reports) + " |", "|---|" + "---|" * len(reports)]
    for name in main.results:
        cells = []
        for rep in reports.values():
            r = rep.results[name]
            cells.append(f"✅ {r.attempts}" if r.ok else f"❌ {r.calls[-2].error_type if len(r.calls) > 1 else r.error_type}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"

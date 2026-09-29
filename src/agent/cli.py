"""CLI: `extract <file>` and `eval <dir>` commands."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .evaluate import render_markdown, run_eval
from .extractor import extract
from .llm import LLMError, get_llm
from .logger import AttemptLogger


def cmd_extract(args: argparse.Namespace) -> int:
    text = Path(args.file).read_text(encoding="utf-8")
    res = extract(text, get_llm(), mode="baseline" if args.baseline else "full",
                  max_retries=args.max_retries, logger=AttemptLogger(args.db))
    if res.ok:
        print(res.posting.model_dump_json(indent=2))
        print(f"# ok after {res.attempts} call(s), run_id={res.run_id}", file=sys.stderr)
        return 0
    print(f"FAILED after {res.attempts} call(s), run_id={res.run_id}: {res.last_error}",
          file=sys.stderr)
    return 1


def cmd_eval(args: argparse.Namespace) -> int:
    modes = ("full",) if args.skip_baseline else ("baseline", "full")
    reports = run_eval(Path(args.dir), get_llm(), modes=modes, logger=AttemptLogger(args.db),
                       max_retries=args.max_retries, progress=lambda m: print(m, file=sys.stderr))
    md = render_markdown(reports, model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"# wrote {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="agent", description=__doc__)
    p.add_argument("--db", default=None, help="SQLite path (default: $AGENT_DB or agent.db)")
    p.add_argument("--max-retries", type=int, default=None, help="override MAX_RETRIES")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extract one posting to JSON")
    e.add_argument("file")
    e.add_argument("--baseline", action="store_true", help="no schema prompt, no retries")
    e.set_defaults(fn=cmd_extract)

    v = sub.add_parser("eval", help="run baseline vs. full agent over a directory")
    v.add_argument("dir")
    v.add_argument("--out", default="eval/results.md")
    v.add_argument("--skip-baseline", action="store_true")
    v.set_defaults(fn=cmd_eval)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (LLMError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

"""report.html: one offline file. The page is template.html with results.json injected as data."""
from __future__ import annotations

import json
import os
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "template.html"
PLACEHOLDER = "__CRB_DATA__"


def data_block(results: dict) -> str:
    """results as JSON that is safe inside <script type="application/json">: no "</", no "<!--"."""
    text = json.dumps(results, ensure_ascii=False, separators=(",", ":"), default=str)
    # "<" is the only character that can end or confuse the script element; \u003c is the same string in JSON.
    # U+2028/2029 are valid JSON but older parsers treat them as line ends.
    return text.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def render(results: dict) -> str:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    if tpl.count(PLACEHOLDER) != 1:
        raise RuntimeError("template.html must contain the data placeholder exactly once")
    return tpl.replace(PLACEHOLDER, data_block(results))


def write(run_dir: Path, results: dict) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "report.html"
    tmp = run_dir / "report.html.tmp"
    tmp.write_text(render(results), encoding="utf-8")
    os.replace(tmp, out)
    return out

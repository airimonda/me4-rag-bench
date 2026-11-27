"""Tests for crb.data: dev packs, read API, verify, sampling, CRAG parsing, ViDoRe build, downloads.
All offline (local files and a localhost HTTP server only)."""
from __future__ import annotations

import bz2
import collections
import json
import re
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from crb.data import crag, devpack, fetch, packs, sample


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("CRB_DATA", str(tmp_path))
    return tmp_path


@pytest.fixture()
def dev(data):
    devpack.generate(data)
    return data


# ------------------------------------------------------------------ dev pack + read API

def test_data_dir_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CRB_DATA", str(tmp_path / "x"))
    assert packs.data_dir() == tmp_path / "x"
    monkeypatch.delenv("CRB_DATA")
    assert packs.data_dir() == Path.home() / "crb-data"


def test_dev_vidore_shape(dev):
    qs = packs.load_questions("dev_vidore")
    assert len(qs) == 8
    assert {"qid", "pack", "question", "answer", "alt_answers", "qtype", "content_type",
            "query_time", "gold_pages"} <= set(qs[0])
    pngs = sorted((dev / "packs/dev_vidore/pages").glob("*.png"))
    mds = sorted((dev / "packs/dev_vidore/pages").glob("*.md"))
    assert len(pngs) == 12 and len(mds) == 12
    for p in pngs:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    for q in qs:
        assert q["gold_pages"]
        for g in q["gold_pages"]:
            png, md = packs.page_paths("dev_vidore", g)
            assert png.is_file() and md.is_file() and png.is_absolute()
            assert md.read_text().startswith("# ")
    qrels = packs.load_qrels("dev_vidore")
    assert set(qrels) == {q["qid"] for q in qs}
    for q in qs:
        assert set(qrels[q["qid"]]) == set(q["gold_pages"])
        assert all(s in (1, 2) for s in qrels[q["qid"]].values())


def test_png_is_decodable():
    png = devpack.make_png(5, 4, 3)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # walk the chunks, check CRCs, inflate the pixel data
    pos, idat, tags = 8, b"", []
    while pos < len(png):
        n = int.from_bytes(png[pos:pos + 4], "big")
        tag, body = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + n]
        crc = int.from_bytes(png[pos + 8 + n:pos + 12 + n], "big")
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF
        tags.append(tag)
        if tag == b"IDAT":
            idat += body
        pos += 12 + n
    assert tags == [b"IHDR", b"IDAT", b"IEND"]
    assert len(zlib.decompress(idat)) == 4 * (1 + 5 * 3)


def test_dev_crag_shape_and_pages(dev):
    qs = packs.load_questions("dev_crag")
    assert len(qs) == 6
    assert all(q["query_time"] and q["gold_pages"] == [] for q in qs)
    assert packs.load_qrels("dev_crag") == {}
    for q in qs:
        pages = packs.crag_pages("dev_crag", q["qid"])
        assert len(pages) == 5
        for p in pages:
            assert Path(p["path"]).is_file() and Path(p["path"]).is_absolute()
            assert set(p) == {"path", "page_name", "page_url", "page_snippet", "page_last_modified"}
            assert p["page_url"].startswith("https://")
    with pytest.raises(packs.DataError):
        packs.crag_pages("dev_crag", "crag-nope")


def test_dev_sample_lists_everything(dev):
    s = packs.load_sample("dev")
    assert s["seed"] == 7
    assert s["packs"]["dev_vidore"]["retrieval"] == sorted(q["qid"] for q in packs.load_questions("dev_vidore"))
    assert len(s["packs"]["dev_crag"]["answer"]) == 6
    with pytest.raises(packs.DataError, match="crb fetch"):
        packs.load_sample("class-v1")


def test_missing_pack_message(data):
    with pytest.raises(packs.DataError, match="crb fetch"):
        packs.load_questions("vidore_v3_hr")
    problems = packs.verify("vidore_v3_hr")
    assert len(problems) == 1 and "crb fetch" in problems[0]


def test_dev_generation_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    devpack.generate(a)
    devpack.generate(b)
    devpack.generate(b)  # idempotent
    fa = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert fa == fb and len(fa) > 50
    for rel in fa:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


def test_verify_ok_and_detects_damage(dev):
    for pk in packs.DEV_PACK_IDS:
        assert packs.verify(pk) == []
    root = dev / "packs/dev_vidore"
    # flip a byte in a page image
    png = next((root / "pages").glob("*.png"))
    raw = bytearray(png.read_bytes())
    raw[-20] ^= 0xFF
    png.write_bytes(bytes(raw))
    probs = packs.verify("dev_vidore")
    assert any("damaged" in p and png.name in p for p in probs)
    assert packs.verify("dev_vidore", quick=True) == []  # quick mode does not hash
    # delete a file
    png.unlink()
    probs = packs.verify("dev_vidore", quick=True)
    assert any("missing" in p and png.name in p for p in probs)


def test_verify_detects_bad_gold_page_and_crag_gap(dev):
    root = dev / "packs/dev_vidore"
    qs = packs.load_questions("dev_vidore")
    qs[0]["gold_pages"] = ["ghost__p9"]
    (root / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in qs) + "\n")
    probs = packs.verify("dev_vidore", quick=True)
    assert any("unknown page ghost__p9" in p for p in probs)
    (dev / "packs/dev_crag/web/dc-2/0.html").unlink()
    assert any("dc-2" in p for p in packs.verify("dev_crag", quick=True))


def test_verify_without_manifest_counts(dev):
    (dev / "packs/dev_crag/manifest.json").unlink()
    assert "not downloaded" in packs.verify("dev_crag")[0]


def test_fetch_main_dev_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CRB_DATA", "/nonexistent")
    rc = fetch.main(SimpleNamespace(dev=True, pack=None, data_dir=str(tmp_path / "d")))
    assert rc == 0
    assert "Dev packs ready" in capsys.readouterr().out
    assert packs.verify("dev_vidore") == [] and packs.verify("dev_crag") == []


def test_fetch_unknown_pack(tmp_path, capsys):
    assert fetch.main(SimpleNamespace(dev=False, pack="nope", data_dir=str(tmp_path))) == 2
    assert "Unknown pack" in capsys.readouterr().out


def test_disk_check_message(tmp_path, monkeypatch):
    msgs = []
    monkeypatch.setattr(fetch, "free_gb", lambda p: 0.2)
    assert fetch.check_disk(tmp_path, ["vidore_v3_hr"], msgs.append) is False
    assert "Not enough free disk space" in msgs[0] and "CRB_DATA" in msgs[0]
    monkeypatch.setattr(fetch, "free_gb", lambda p: 50.0)
    assert fetch.check_disk(tmp_path, list(packs.PACK_IDS), msgs.append) is True


def test_fetch_stops_when_disk_full(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch, "free_gb", lambda p: 0.1)
    assert fetch.main(SimpleNamespace(dev=False, pack="crag_task1", data_dir=str(tmp_path))) == 2
    assert "Not enough free disk space" in capsys.readouterr().out


# ------------------------------------------------------------------ sampling

def _synthetic(n_by_type: dict[str, int], prefix="hr") -> list[tuple[str, str]]:
    items, i = [], 0
    for t, n in n_by_type.items():
        for _ in range(n):
            items.append((f"{prefix}-{i}", t))
            i += 1
    return items


def test_sample_size_and_proportional_stratification():
    items = _synthetic({"extractive": 120, "open-ended": 90, "boolean": 60, "multi-hop": 33, "numerical": 15})
    out = sample.stratified_sample(items, 100)
    assert len(out) == 100 == len(set(out))
    assert out == sorted(out)
    stratum = dict(items)
    got = collections.Counter(stratum[q] for q in out)
    total = len(items)
    for t, size in collections.Counter(s for _, s in items).items():
        exact = 100 * size / total
        assert int(exact) <= got[t] <= int(exact) + 1, (t, exact, got[t])


def test_sample_largest_remainder_exact():
    # sizes 5,3,2 of 10, n=4: exact 2.0, 1.2, 0.8 -> floors 2,1,0 + leftover 1 goes to biggest remainder (0.8)
    items = _synthetic({"a": 5, "b": 3, "c": 2})
    got = collections.Counter(dict(items)[q] for q in sample.stratified_sample(items, 4))
    assert got == {"a": 2, "b": 1, "c": 1}
    # ties on remainder are broken by stratum name: 1/1/1 of n=2 -> a and b get the seats
    items = _synthetic({"a": 1, "b": 1, "c": 1})
    got = sample.stratified_sample(items, 2)
    assert {dict(items)[q] for q in got} == {"a", "b"}


def test_sample_deterministic_and_order_independent():
    items = _synthetic({"x": 200, "y": 70, "z": 31})
    a = sample.stratified_sample(items, 100)
    b = sample.stratified_sample(list(reversed(items)), 100)
    assert a == b == sample.stratified_sample(items, 100)
    assert a != sample.stratified_sample(items, 100, seed=8)
    # the module rule is random.Random(7)
    assert sample.SEED == 7


def test_sample_small_pool_returns_everything():
    items = _synthetic({"a": 3, "b": 2})
    assert sample.stratified_sample(items, 100) == sorted(q for q, _ in items)


def test_build_class_v1_from_synthetic_packs(data):
    def write_pack(pack, items):
        d = data / "packs" / pack
        d.mkdir(parents=True)
        (d / "questions.jsonl").write_text(
            "\n".join(json.dumps({"qid": q, "qtype": t}) for q, t in items) + "\n")
    hr = _synthetic({"extractive": 150, "open-ended": 100, "multi-hop": 68}, "hr")
    crag_items = _synthetic({"simple": 700, "comparison": 300, "set": 200, "false_premise": 135}, "crag")
    write_pack("vidore_v3_hr", hr)
    write_pack("crag_task1", crag_items)
    out = sample.build(data)
    s = json.loads(out.read_text())
    assert s["seed"] == 7
    assert set(s["packs"]) == {"vidore_v3_hr", "crag_task1"}   # the CS pack is not on disk, so not listed
    assert s["packs"]["vidore_v3_hr"]["retrieval"] == sorted(q for q, _ in hr)
    assert len(s["packs"]["vidore_v3_hr"]["answer"]) == 100
    assert "retrieval" not in s["packs"]["crag_task1"] and len(s["packs"]["crag_task1"]["answer"]) == 400
    # adding the CS pack later keeps the others and does not change them
    write_pack("vidore_v3_computer_science", _synthetic({"a": 100, "b": 115}, "cs"))
    s2 = json.loads(sample.build(data).read_text())
    assert s2["packs"]["vidore_v3_hr"] == s["packs"]["vidore_v3_hr"]
    assert len(s2["packs"]["vidore_v3_computer_science"]["answer"]) == 100
    assert packs.load_sample("class-v1")["packs"]["crag_task1"] == s["packs"]["crag_task1"]


# ------------------------------------------------------------------ CRAG html stripper

HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>My   Page</title>
<style>body{color:red}</style><script>var x = "<p>not text</p>";</script></head>
<body class="a" style="color:blue" onload="evil()">
<!-- a comment with <b>tags</b> -->
<noscript><img src="x.gif">enable js</noscript>
<svg width="10"><path d="M0 0"/><text>svgtext</text></svg>
<div id="main" data-x="1" style="display:none"><p style="margin:0">Hello <b>big</b>   world &amp; friends</p>
<span><a href="http://example.com/x?y=1">link text</a></span>
<div> </div><p></p><div><span></span></div>
<ul><li>one</li><li>two</li></ul>
<table><tr><td colspan="2" class="c">cell &lt;1&gt;</td></tr></table>
<script type="application/ld+json">{"a": "</div>"}</script>
<form><input type="text"><button>Go</button>form text kept</form>
</div><br><br>tail</body></html>"""


def test_strip_html_removes_noise_keeps_text():
    out = crag.strip_html(HTML)
    for gone in ("<script", "<style", "<noscript", "<svg", "<!--", "style=", "class=", "id=", "onload",
                 "not text", "color:red", "comment", "svgtext", "enable js", "evil", "ld+json", "href",
                 "<span", "<a ", "<form", "<input", "<button", "<meta"):
        assert gone not in out, gone
    for kept in ("<title>My Page</title>", "Hello <b>big</b> world &amp; friends", "link text",
                 "<li>one</li><li>two</li>", '<td colspan="2">cell &lt;1&gt;</td>', "form text kept", "tail"):
        assert kept in out, (kept, out)
    assert "<p></p>" not in out and "<div></div>" not in out and "<div> </div>" not in out
    assert "  " not in out and "\n" not in out
    assert out.count("<br>") == 1  # consecutive br collapsed


def test_strip_html_edge_cases():
    assert crag.strip_html("") == "" and crag.strip_html(None) == ""
    assert crag.strip_html("<script>x</script><style>y</style>") == ""
    assert crag.strip_html("plain text only") == "plain text only"
    # unbalanced / malformed input still gives balanced output
    out = crag.strip_html("<div><p>one<div>two</p>three")
    assert out.count("<div>") == out.count("</div>") and out.count("<p>") == out.count("</p>")
    assert "one" in out and "two" in out and "three" in out
    # words around removed inline tags do not glue together
    assert "alpha beta" in re.sub(r"<[^>]+>", "", crag.strip_html("<p>alpha <span>beta</span></p>"))
    # nested svg inside svg is dropped as a whole, text after it survives
    assert crag.strip_html("<p>a</p><svg><svg><g/></svg>x</svg><p>b</p>") == "<p>a</p><p>b</p>"


# ------------------------------------------------------------------ CRAG parser on a synthetic .jsonl.bz2

def _crag_records(n=12):
    recs = []
    for i in range(n):
        recs.append({
            "interaction_id": f"id-{i:02d}",
            "query_time": "03/05/2024, 23:18:31 PT",
            "domain": "sports",
            "question_type": ["simple", "comparison", "set"][i % 3],
            "static_or_dynamic": "static",
            "query": f"question {i}?",
            "answer": f"answer {i}" if i % 5 else 7,
            "alt_ans": ["alt a"] if i % 2 else [],
            "split": i % 2,            # odd i -> 1 (public test), even -> 0
            "search_results": [
                {"page_name": f"n{i}-{k}", "page_url": f"http://e.org/{i}/{k}",
                 "page_snippet": f"snippet {k}", "page_last_modified": "Tue, 1 Jan 2024",
                 "page_result": f"<html><body><script>junk()</script><p>text {i}-{k}</p></body></html>"}
                for k in range(5)],
        })
    return recs


def _write_bz2(path: Path, recs):
    with bz2.open(path, "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_crag_build_from_synthetic_bz2(data, tmp_path):
    src = tmp_path / "crag.jsonl.bz2"
    recs = _crag_records()
    _write_bz2(src, recs)
    msgs = []
    m = crag.build_from_file(src, data, say=msgs.append)
    assert m["counts"] == {"questions": 6, "pages": 30}
    assert m["license"] == "CC BY-NC 4.0" and "facebookresearch/CRAG" in m["source"]
    assert packs.verify("crag_task1") == []
    qs = packs.load_questions("crag_task1")
    assert [q["qid"] for q in qs] == [f"crag-id-{i:02d}" for i in (1, 3, 5, 7, 9, 11)]
    q = qs[0]
    assert q["pack"] == "crag_task1" and q["qtype"] == "comparison" and q["alt_answers"] == ["alt a"]
    assert q["query_time"] == "03/05/2024, 23:18:31 PT" and q["gold_pages"] == [] and q["answer"] == "answer 1"
    pages = packs.crag_pages("crag_task1", "crag-id-05")
    assert len(pages) == 5 and pages[3]["page_name"] == "n5-3" and pages[3]["page_url"] == "http://e.org/5/3"
    html0 = Path(pages[0]["path"]).read_text()
    assert html0 == "<html><body><p>text 5-0</p></body></html>"
    assert "junk" not in html0
    # a split == 0 question is not in the pack
    assert not (data / "packs/crag_task1/web/crag-id-00").exists()
    # int answer is stringified
    src2 = tmp_path / "c2.jsonl.bz2"
    _write_bz2(src2, [dict(_crag_records(2)[1], answer=7)])
    crag.build_from_file(src2, data, say=msgs.append)
    assert packs.load_questions("crag_task1")[0]["answer"] == "7"


def test_crag_split_as_string_and_empty_file(data, tmp_path):
    recs = _crag_records(2)
    recs[1]["split"] = "1"
    src = tmp_path / "s.jsonl.bz2"
    _write_bz2(src, recs)
    assert crag.build_from_file(src, data, say=lambda m: None)["counts"]["questions"] == 1
    empty = tmp_path / "e.jsonl.bz2"
    _write_bz2(empty, [_crag_records(1)[0]])  # only split 0
    with pytest.raises(ValueError, match="split==1"):
        crag.build_from_file(empty, data, say=lambda m: None)


def test_crag_truncated_download_is_reported(data, tmp_path):
    src = tmp_path / "t.jsonl.bz2"
    _write_bz2(src, _crag_records(40))
    raw = src.read_bytes()
    src.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(fetch.DownloadError, match="incomplete"):
        crag.build_from_file(src, data, say=lambda m: None)


def test_crag_sampling_end_to_end(data, tmp_path):
    recs = _crag_records(40)
    for r in recs:
        r["split"] = 1
    src = tmp_path / "big.jsonl.bz2"
    _write_bz2(src, recs)
    crag.build_from_file(src, data, say=lambda m: None)
    out = json.loads(sample.build(data).read_text())["packs"]["crag_task1"]["answer"]
    assert len(out) == 40  # fewer questions than the 400 asked for: everything is in


# ------------------------------------------------------------------ ViDoRe build from synthetic parquet

pa = pytest.importorskip("pyarrow")


def _write_vidore_parquets(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    langs = ["english", "french"]
    qrows = []
    qid = 0
    for lang in langs:
        for i in range(6):
            qrows.append({"query_id": qid, "query": f"{lang} q{i}", "language": lang,
                          "query_types": ["multi-hop", "numerical"] if i % 2 else ["extractive"],
                          "content_type": ["Text", "Table"], "answer": f"{lang} a{i}"})
            qid += 1
    pq.write_table(pa.Table.from_pylist(qrows), tmp_path / "queries.parquet")
    # corpus: 3 docs x 4 pages, in two files; page numbers deliberately not contiguous
    corpus = []
    for cid in range(12):
        doc = f"Doc One/{cid // 4}"   # unsafe characters on purpose
        corpus.append({"corpus_id": cid, "image": {"bytes": devpack.make_png(8, 8, cid), "path": f"{cid}.png"},
                       "doc_id": doc, "markdown": f"markdown {cid}" if cid != 5 else None,
                       "page_number_in_doc": (cid % 4) * 3})
    pq.write_table(pa.Table.from_pylist(corpus[:7]), tmp_path / "c0.parquet", row_group_size=3)
    pq.write_table(pa.Table.from_pylist(corpus[7:]), tmp_path / "c1.parquet")
    qrels = []
    for q in qrows:
        qid_ = qrows.index(q)
        qrels.append({"query_id": qid_, "corpus_id": (qid_ * 2) % 12, "score": 1,
                      "content_type": ["Text"], "bounding_boxes": []})
        qrels.append({"query_id": qid_, "corpus_id": (qid_ * 2 + 3) % 12, "score": 2,
                      "content_type": ["Text"], "bounding_boxes": []})
    qrels.append({"query_id": 0, "corpus_id": 999, "score": 1, "content_type": [], "bounding_boxes": []})
    pq.write_table(pa.Table.from_pylist(qrels), tmp_path / "qrels.parquet")
    return tmp_path / "queries.parquet", tmp_path / "qrels.parquet", [tmp_path / "c0.parquet", tmp_path / "c1.parquet"]


def test_vidore_build_from_synthetic_parquet(data, tmp_path):
    from crb.data import vidore
    qp, rp, cps = _write_vidore_parquets(tmp_path)
    msgs = []
    m = vidore.build_from_files("vidore_v3_hr", data, qp, rp, cps, say=msgs.append)
    assert m["counts"] == {"pages": 12, "questions": 6, "qrels": 12}
    assert m["license"] == "CC-BY-4.0" and "vidore_v3_hr" in m["source"]
    assert packs.verify("vidore_v3_hr") == []
    assert any("not in the corpus" in x for x in msgs)  # the corpus_id 999 row was skipped, with a warning
    qs = packs.load_questions("vidore_v3_hr")
    assert [q["qid"] for q in qs] == [f"hr-{i}" for i in range(6)]       # English only, French ignored
    assert qs[1]["qtype"] == "multi-hop" and qs[0]["qtype"] == "extractive"  # first query type
    assert qs[0]["content_type"] == ["Text", "Table"] and qs[0]["alt_answers"] == [] and qs[0]["query_time"] is None
    pid = packs.page_id_for("Doc One/0", 3)
    assert re.fullmatch(r"[A-Za-z0-9._-]+", pid) and pid == "Doc_One_0__p3"
    png, md = packs.page_paths("vidore_v3_hr", pid)
    assert png.read_bytes()[:4] == b"\x89PNG" and md.read_text() == "markdown 1"
    # empty markdown is written as an empty file
    assert packs.page_paths("vidore_v3_hr", packs.page_id_for("Doc One/1", 3))[1].read_text() == ""
    qrels = packs.load_qrels("vidore_v3_hr")
    assert set(qrels) == {f"hr-{i}" for i in range(6)}                     # French questions' rows dropped
    # gold pages: highest relevance first
    q0 = qs[0]
    assert len(q0["gold_pages"]) == 2 and qrels["hr-0"][q0["gold_pages"][0]] == 2
    # the generated sample covers all of them (retrieval = every English question)
    s = json.loads(sample.build(data).read_text())["packs"]["vidore_v3_hr"]
    assert s["retrieval"] == sorted(q["qid"] for q in qs) and len(s["answer"]) == 6


def test_vidore_duplicate_page_id_is_an_error(data, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from crb.data import vidore
    qp, rp, _ = _write_vidore_parquets(tmp_path)
    row = {"corpus_id": 0, "image": {"bytes": devpack.make_png(4, 4, 1), "path": "x.png"},
           "doc_id": "d", "markdown": "m", "page_number_in_doc": 1}
    pq.write_table(pa.Table.from_pylist([row, dict(row, corpus_id=1)]), tmp_path / "dup.parquet")
    with pytest.raises(ValueError, match="same page id"):
        vidore.build_from_files("vidore_v3_hr", data, qp, rp, [tmp_path / "dup.parquet"], say=lambda m: None)


# ------------------------------------------------------------------ downloader (local server)

class _Handler(BaseHTTPRequestHandler):
    payload = bytes(range(256)) * 4000   # ~1 MB
    fail_first = 0
    seen_ranges: list = []
    no_range = False

    def log_message(self, *a):
        pass

    def do_GET(self):
        cls = type(self)
        if self.path == "/missing":
            self.send_response(404); self.end_headers(); return
        rng = self.headers.get("Range")
        cls.seen_ranges.append(rng)
        data = cls.payload
        start = 0
        if rng and not cls.no_range:
            start = int(re.match(r"bytes=(\d+)-", rng).group(1))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if cls.fail_first > 0 and start == 0:
            cls.fail_first -= 1
            self.wfile.write(body[: len(body) // 3])   # then drop the connection mid-body
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)


@pytest.fixture()
def server():
    _Handler.fail_first = 0
    _Handler.seen_ranges = []
    _Handler.no_range = False
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_download_plain_and_skip_existing(server, tmp_path):
    dest = tmp_path / "sub" / "f.bin"
    msgs = []
    fetch.download(server + "/f", dest, expected_size=len(_Handler.payload), say=msgs.append)
    assert dest.read_bytes() == _Handler.payload and not dest.with_name("f.bin.part").exists()
    n = len(_Handler.seen_ranges)
    fetch.download(server + "/f", dest, expected_size=len(_Handler.payload), say=msgs.append)
    assert len(_Handler.seen_ranges) == n and any("already downloaded" in m for m in msgs)


def test_download_resumes_from_part_file(server, tmp_path):
    dest = tmp_path / "f.bin"
    part = tmp_path / "f.bin.part"
    part.write_bytes(_Handler.payload[:300_000])
    fetch.download(server + "/f", dest, expected_size=len(_Handler.payload), say=lambda m: None)
    assert dest.read_bytes() == _Handler.payload
    assert _Handler.seen_ranges[-1] == "bytes=300000-"


def test_download_retries_after_dropped_connection(server, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    _Handler.fail_first = 2
    msgs = []
    dest = tmp_path / "g.bin"
    fetch.download(server + "/f", dest, say=msgs.append)  # no expected size: uses Content-Length
    assert dest.read_bytes() == _Handler.payload
    assert sum("retrying" in m for m in msgs) >= 1


def test_download_server_ignores_range(server, tmp_path):
    _Handler.no_range = True
    dest = tmp_path / "h.bin"
    (tmp_path / "h.bin.part").write_bytes(b"garbage" * 100)
    fetch.download(server + "/f", dest, expected_size=len(_Handler.payload), say=lambda m: None)
    assert dest.read_bytes() == _Handler.payload


def test_download_404_is_a_plain_error(server, tmp_path):
    with pytest.raises(fetch.DownloadError, match="refused"):
        fetch.download(server + "/missing", tmp_path / "x", say=lambda m: None)

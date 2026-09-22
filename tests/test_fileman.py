"""FileManager / CorpusStore tests: incremental regeneration memory (network-free).

Tiny doc trees are built directly under tmp_path (Manual/ + ScriptReference/);
FileManager root is tmp_path itself.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from rag.store import CorpusStore, FileManager, ManifestDiff
from rag.store.corpus_store import CorruptCorpus, corpus_path

DIRS = ["Manual", "ScriptReference"]

GEN_PARTS = {
    "prompt_version": "p1",
    "model": "m1",
    "extractor_version": "e1",
    "schema_version": "s1",
}


# --------------------------------------------------------------------------------------
# helpers / fixtures
# --------------------------------------------------------------------------------------
def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _manager(tmp_path: Path, **kwargs) -> FileManager:
    return FileManager(root=tmp_path, dirs=list(DIRS), corpus_dir=tmp_path / "corpus", **kwargs)


@pytest.fixture()
def site(tmp_path: Path) -> Path:
    """Tiny doc mirror: 4 html pages across Manual/ and ScriptReference/."""
    _write(tmp_path, "Manual/index.html", "<html>index</html>")
    _write(tmp_path, "Manual/install.html", "<html>install</html>")
    _write(tmp_path, "ScriptReference/Rigidbody.html", "<html>rb</html>")
    _write(tmp_path, "ScriptReference/Collider.html", "<html>col</html>")
    return tmp_path


ALL_FOUR = [
    "Manual/index.html",
    "Manual/install.html",
    "ScriptReference/Collider.html",
    "ScriptReference/Rigidbody.html",
]


# --------------------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------------------
def test_scan_counts_html_files(site):
    fm = _manager(site)
    scanned = fm.scan()
    assert len(scanned) == 4
    assert sorted(scanned) == ALL_FOUR
    assert list(scanned) == sorted(scanned)  # keys sorted
    assert scanned["Manual/index.html"] == _md5(site / "Manual/index.html")
    assert scanned["ScriptReference/Rigidbody.html"] == _md5(site / "ScriptReference/Rigidbody.html")


def test_scan_ignores_non_html(site):
    _write(site, "Manual/notes.txt", "not html")
    _write(site, "ScriptReference/image.png", "PNG")
    assert len(_manager(site).scan()) == 4


def test_scan_skips_missing_dirs(site):
    fm = FileManager(root=site, dirs=["NoSuchDir"], corpus_dir=site / "corpus")
    assert fm.scan() == {}


# --------------------------------------------------------------------------------------
# diff
# --------------------------------------------------------------------------------------
def test_diff_without_manifest_everything_added(site):
    diff = _manager(site).diff()
    assert isinstance(diff, ManifestDiff)
    assert diff.added == ALL_FOUR
    assert diff.changed == diff.removed == diff.unchanged == []


def test_diff_detects_added_changed_removed_unchanged(site):
    fm = _manager(site)
    fm.save_manifest(fm.scan(), gen_key="k0", gen_parts={})
    # mutate the mirror: change one, remove one, add one, leave two alone
    _write(site, "Manual/index.html", "<html>index v2</html>")
    (site / "Manual/install.html").unlink()
    _write(site, "Manual/newpage.html", "<html>new page</html>")
    diff = fm.diff()
    assert diff.changed == ["Manual/index.html"]
    assert diff.removed == ["Manual/install.html"]
    assert diff.added == ["Manual/newpage.html"]
    assert diff.unchanged == ["ScriptReference/Collider.html", "ScriptReference/Rigidbody.html"]


def test_rename_is_remove_plus_add(site):
    fm = _manager(site)
    fm.save_manifest(fm.scan(), gen_key="k0", gen_parts={})
    (site / "ScriptReference/Collider.html").rename(site / "ScriptReference/BoxCollider.html")
    diff = fm.diff()
    assert diff.removed == ["ScriptReference/Collider.html"]
    assert diff.added == ["ScriptReference/BoxCollider.html"]
    assert diff.changed == []


def test_second_scan_is_all_unchanged(site):
    fm = _manager(site)
    fm.save_manifest(fm.scan(), gen_key="k0", gen_parts={})
    diff = fm.diff()
    assert diff.added == diff.changed == diff.removed == []
    assert diff.unchanged == ALL_FOUR


def test_diff_honours_explicit_scanned(site):
    fm = _manager(site)
    scanned = fm.scan()
    diff = fm.diff(scanned)
    assert diff.added == ALL_FOUR  # no manifest yet; explicit dict is used as-is


# --------------------------------------------------------------------------------------
# gen_key / current_gen_key
# --------------------------------------------------------------------------------------
def test_gen_key_is_16_hex_chars_and_stable():
    k1 = FileManager.gen_key("p1", "e1", "s1")
    k2 = FileManager.gen_key("p1", "e1", "s1")
    assert k1 == k2
    assert re.fullmatch(r"[0-9a-f]{16}", k1)


def test_gen_key_changes_when_any_component_changes():
    """The key covers prompt/extractor/schema versions only — deliberately
    NOT the model (a flash -> flashx rename once requeued a 21k-page build).
    Version bumps change the key; the model never does."""
    base = FileManager.gen_key("p1", "e1", "s1")
    assert FileManager.gen_key("p1-changed", "e1", "s1") != base
    assert FileManager.gen_key("p1", "e1-changed", "s1") != base
    assert FileManager.gen_key("p1", "e1", "s1-changed") != base


def test_current_gen_key_recomputed_from_manifest_parts(site):
    fm = _manager(site)
    assert fm.current_gen_key() == ""  # no manifest -> no key
    key = FileManager.gen_key("p1", "e1", "s1")
    # GEN_PARTS still carries a "model" (informational): recomputation ignores it
    fm.save_manifest({}, gen_key=key, gen_parts=GEN_PARTS)
    assert fm.current_gen_key() == key
    # the stored key string itself is not trusted; parts are recomputed
    fm.save_manifest({}, gen_key="bogus", gen_parts=GEN_PARTS)
    assert fm.current_gen_key() == key


# --------------------------------------------------------------------------------------
# load_manifest / save_manifest
# --------------------------------------------------------------------------------------
def test_load_manifest_missing_returns_empty(site):
    m = _manager(site).load_manifest()
    assert m["files"] == {}
    assert m["gen_key"] == ""
    assert m["version"] == 1


def test_load_manifest_tolerates_corrupt_json(site):
    fm = _manager(site)
    fm.corpus_dir.mkdir(parents=True)
    fm.manifest_path.write_bytes(b"\x00\xff definitely not json {{{")
    m = fm.load_manifest()
    assert m["files"] == {}
    assert m["version"] == 1


def test_save_manifest_round_trip(site):
    fm = _manager(site)
    files = fm.scan()
    key = FileManager.gen_key("p1", "e1", "s1")
    fm.save_manifest(files, gen_key=key, gen_parts=GEN_PARTS)
    m = fm.load_manifest()
    assert m["files"] == files
    assert m["gen_key"] == key
    assert m["gen_parts"] == GEN_PARTS
    assert m["version"] == 1
    # paths in the manifest are posix, forward slashes
    assert "Manual/index.html" in m["files"]
    assert all("\\" not in rel for rel in m["files"])


def test_save_manifest_creates_corpus_dir(site):
    fm = _manager(site)
    assert not fm.corpus_dir.exists()
    fm.save_manifest(fm.scan(), gen_key="k", gen_parts={})
    assert fm.manifest_path.is_file()


def test_save_manifest_leaves_no_tmp_and_replaces_stale_tmp(site):
    fm = _manager(site)
    fm.corpus_dir.mkdir(parents=True)
    stale_tmp = fm.manifest_path.parent / (fm.manifest_path.name + ".tmp")
    stale_tmp.write_text("partial write from a crashed run", encoding="utf-8")
    fm.save_manifest(fm.scan(), gen_key="k1", gen_parts={})
    # tmp sibling consumed by os.replace; stale partial write gone
    assert not stale_tmp.exists()
    assert not list(fm.corpus_dir.glob("*.tmp"))
    assert fm.load_manifest()["gen_key"] == "k1"


# --------------------------------------------------------------------------------------
# prune
# --------------------------------------------------------------------------------------
def test_prune_deletes_only_listed_corpus_files(site):
    fm = _manager(site)
    store = CorpusStore(fm.corpus_dir)
    store.save("Manual/index.html", {"page": "index"})
    store.save("ScriptReference/Rigidbody.html", {"page": "rb"})
    store.save("Manual/keep.html", {"page": "keep"})
    # missing entries in `removed` are tolerated
    fm.prune(["Manual/index.html", "ScriptReference/Rigidbody.html", "Manual/not-there.html"])
    assert not corpus_path(fm.corpus_dir, "Manual/index.html").exists()
    assert not corpus_path(fm.corpus_dir, "ScriptReference/Rigidbody.html").exists()
    assert corpus_path(fm.corpus_dir, "Manual/keep.html").exists()


# --------------------------------------------------------------------------------------
# CorpusStore
# --------------------------------------------------------------------------------------
def test_corpus_store_save_load_round_trip_unicode(site):
    store = CorpusStore(site / "corpus")
    data = {"title": "物理引擎 Rigidbody 刚体", "symbols": ["速度", "angularVelocity"]}
    target = store.save("ScriptReference/Rigidbody.html", data)
    assert target == site / "corpus" / "ScriptReference" / "Rigidbody.html.rag.json"
    assert target.is_file()
    assert store.load("ScriptReference/Rigidbody.html") == data
    assert store.exists("ScriptReference/Rigidbody.html")
    # ensure_ascii=False: raw file holds utf-8, not \u escapes
    assert "物理引擎".encode("utf-8") in target.read_bytes()
    # and it parses back as valid JSON
    assert json.loads(target.read_text(encoding="utf-8")) == data


def test_corpus_store_save_creates_parents_and_is_atomic(site):
    store = CorpusStore(site / "corpus")
    target = store.save("Manual/Sub/Section/deep.html", {"x": 1})
    assert target.is_file()
    assert not list((site / "corpus").glob("*.tmp"))


def test_corpus_store_missing_loads_as_none(site):
    store = CorpusStore(site / "corpus")
    assert store.load("Manual/nope.html") is None
    assert not store.exists("Manual/nope.html")
    assert store.missing_or_corrupt("Manual/nope.html")


def test_corpus_store_corrupt_load_raises(site):
    store = CorpusStore(site / "corpus")
    bad = corpus_path(store.path, "Manual/bad.html")
    bad.parent.mkdir(parents=True)
    bad.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(CorruptCorpus) as exc_info:
        store.load("Manual/bad.html")
    assert isinstance(exc_info.value, ValueError)
    assert exc_info.value.path == bad
    assert store.missing_or_corrupt("Manual/bad.html")
    # a healthy file right next to it is fine
    store.save("Manual/good.html", {"ok": True})
    assert not store.missing_or_corrupt("Manual/good.html")


def test_corpus_store_iterate_all(site):
    store = CorpusStore(site / "corpus")
    store.save("Manual/index.html", {"p": 1})
    store.save("ScriptReference/Rigidbody.html", {"p": 2})
    store.save("Manual/Sub/Section/deep.html", {"p": 3})
    # corrupt file is skipped, not fatal
    bad = corpus_path(store.path, "ScriptReference/bad.html")
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("]]] not json", encoding="utf-8")
    entries = list(store.iterate_all())
    assert entries == [
        ("Manual/Sub/Section/deep.html", {"p": 3}),
        ("Manual/index.html", {"p": 1}),
        ("ScriptReference/Rigidbody.html", {"p": 2}),
    ]
    for rel, _data in entries:  # posix rel keys, .rag.json stripped
        assert "\\" not in rel
        assert not rel.endswith(".rag.json")


def test_corpus_store_iterate_all_empty_dir(site):
    store = CorpusStore(site / "corpus")
    assert list(store.iterate_all()) == []

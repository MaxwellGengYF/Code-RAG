"""md5-based incremental regeneration memory.

Compile is expensive (one LLM call per page), so the pipeline remembers what it
already generated: :class:`FileManager` scans the doc mirror, keeps a manifest of
``rel-path -> content-md5`` at ``corpus_dir/manifest.json``, and diffs a fresh scan
against it to find pages that were added / changed / removed / left alone. Only
``added + changed`` pages are regenerated; ``removed`` pages have their corpus
files pruned; ``unchanged`` pages are reused as-is.

A "generation key" (sha1 over prompt/model/extractor/schema versions, first 16
hex chars) is stored with the manifest — together with the parts it was built
from — so a configuration change can be detected and force a full rebuild.

All paths inside the manifest are posix paths relative to *root* (forward slashes).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.store.corpus_store import corpus_path

log = logging.getLogger(__name__)

#: manifest schema version written by save_manifest
MANIFEST_VERSION = 1


@dataclass
class ManifestDiff:
    """Result of comparing a fresh scan against the stored manifest.

    Every field holds ROOT-relative posix source paths (``Manual/...`` / ``ScriptReference/...``),
    sorted. A renamed page shows up as one ``removed`` + one ``added`` entry
    (content md5 is not compared across different paths).
    """

    added: list[str]      # in the scan but not in the manifest
    changed: list[str]    # in both, content md5 differs
    removed: list[str]    # in the manifest but no longer on disk
    unchanged: list[str]  # in both, content md5 identical

    def __post_init__(self) -> None:
        self.added = sorted(self.added)
        self.changed = sorted(self.changed)
        self.removed = sorted(self.removed)
        self.unchanged = sorted(self.unchanged)


def _md5_hex(path: Path) -> str:
    """Content md5 of one file (chunked reads)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _empty_manifest() -> dict[str, Any]:
    """The manifest equivalent to "nothing generated yet"."""
    return {"files": {}, "gen_key": "", "version": MANIFEST_VERSION}


class FileManager:
    """Scan + manifest bookkeeping for the doc mirror under *root* / *dirs*.

    Args:
        root: doc-mirror root; every path in the manifest is posix-relative to it.
        dirs: subdirectories to scan, e.g. ["Manual", "ScriptReference"].
        corpus_dir: directory holding corpus files and the manifest.
        manifest_name: manifest file name inside *corpus_dir*.
    """

    def __init__(
        self,
        root: str | Path,
        dirs: list[str],
        corpus_dir: str | Path,
        manifest_name: str = "manifest.json",
    ):
        self.root = Path(root).resolve()
        self.dirs = list(dirs)
        self.corpus_dir = Path(corpus_dir)
        self.manifest_name = manifest_name

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        """Where the manifest JSON lives (corpus_dir / manifest_name)."""
        return self.corpus_dir / self.manifest_name

    # ------------------------------------------------------------------
    # scanning
    # ------------------------------------------------------------------
    def scan(self) -> dict[str, str]:
        """{rel_posix_path: content_md5} for every *.html under root/dirs, sorted.

        Missing dirs are skipped silently so the same config works on partial
        mirrors; non-.html files are ignored.
        """
        scanned: dict[str, str] = {}
        for d in self.dirs:
            base = self.root / d
            if not base.is_dir():
                continue
            for dirpath, _dirnames, filenames in os.walk(base):
                for name in filenames:
                    if not name.endswith(".html"):
                        continue
                    full = Path(dirpath) / name
                    scanned[full.relative_to(self.root).as_posix()] = _md5_hex(full)
        return dict(sorted(scanned.items()))

    # ------------------------------------------------------------------
    # manifest
    # ------------------------------------------------------------------
    def load_manifest(self) -> dict:
        """Parsed manifest; a missing or corrupt file reads as empty.

        The result always contains a ``files`` dict (rel -> md5). A corrupt manifest
        is logged and treated as "nothing generated yet" so it can never block the
        pipeline; a stored manifest is returned as-is (plus the ``files`` guarantee).
        """
        path = self.manifest_path
        if not path.exists():
            return _empty_manifest()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            log.warning("corrupt manifest %s treated as empty: %s", path, exc)
            return _empty_manifest()
        if not isinstance(data, dict):
            log.warning("manifest %s is not a JSON object; treated as empty", path)
            return _empty_manifest()
        if not isinstance(data.get("files"), dict):
            data["files"] = {}
        return data

    def diff(self, scanned: dict[str, str] | None = None) -> ManifestDiff:
        """Compare *scanned* (default: scan()) against the stored manifest."""
        if scanned is None:
            scanned = self.scan()
        manifest_files = self.load_manifest().get("files", {})
        added = [rel for rel in scanned if rel not in manifest_files]
        changed = [rel for rel in scanned
                   if rel in manifest_files and manifest_files[rel] != scanned[rel]]
        unchanged = [rel for rel in scanned
                     if rel in manifest_files and manifest_files[rel] == scanned[rel]]
        removed = [rel for rel in manifest_files if rel not in scanned]
        return ManifestDiff(added=added, changed=changed, removed=removed, unchanged=unchanged)

    # ------------------------------------------------------------------
    # generation key
    # ------------------------------------------------------------------
    @staticmethod
    def gen_key(prompt_version: str, model: str, extractor_version: str, schema_version: str) -> str:
        """Fingerprint of the generation configuration: first 16 hex chars of the
        sha1 of "prompt_version|model|extractor_version|schema_version"."""
        raw = f"{prompt_version}|{model}|{extractor_version}|{schema_version}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def current_gen_key(self) -> str:
        """gen_key recomputed from the parts stored in the manifest; "" when absent."""
        parts = self.load_manifest().get("gen_parts")
        if not isinstance(parts, dict):
            return ""
        try:
            return self.gen_key(
                str(parts["prompt_version"]),
                str(parts["model"]),
                str(parts["extractor_version"]),
                str(parts["schema_version"]),
            )
        except KeyError:
            return ""

    def save_manifest(self, files: dict[str, str], gen_key: str, gen_parts: dict) -> None:
        """Atomically persist the manifest.

        JSON is written to ``manifest.json.tmp`` (fsynced) and then os.replace-d over
        ``manifest.json``; *corpus_dir* is created when missing. The generation parts
        are stored alongside the key so :meth:`current_gen_key` can recompute it.
        """
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": MANIFEST_VERSION,
            "gen_key": gen_key,
            "gen_parts": dict(gen_parts),
            "files": dict(files),
        }
        tmp = self.manifest_path.parent / (self.manifest_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.manifest_path)

    # ------------------------------------------------------------------
    # pruning
    # ------------------------------------------------------------------
    def prune(self, removed: list[str]) -> None:
        """Delete the corpus files for *removed* rel paths; missing files tolerated.

        Empty parent directories are left behind on purpose.
        """
        for rel in removed:
            corpus_path(self.corpus_dir, rel).unlink(missing_ok=True)

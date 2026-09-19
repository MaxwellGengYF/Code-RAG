"""Per-page corpus file I/O.

The compile step turns every source page (``Manual/*.html`` / ``ScriptReference/*.html``)
into one JSON "corpus file"::

    <corpus_dir>/<rel>.rag.json        # rel = posix path relative to ROOT

e.g. ``ScriptReference/Rigidbody.html`` ->
``<corpus_dir>/ScriptReference/Rigidbody.html.rag.json``.

Writes are atomic (tmp sibling + os.replace, with fsync) so a crashed compile never
leaves a half-written JSON file behind. Readers treat a corrupt file as "needs
regeneration" (:class:`CorruptCorpus`) instead of crashing the pipeline.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

#: suffix every corpus file carries (appended to the source rel path)
CORPUS_SUFFIX = ".rag.json"


class CorruptCorpus(ValueError):
    """A corpus file exists but does not contain a parseable JSON object."""

    def __init__(self, path: str | Path, detail: str):
        self.path = Path(path)
        super().__init__(f"corrupt corpus file {self.path}: {detail}")


def corpus_path(corpus_dir: str | Path, rel: str) -> Path:
    """The corpus file for source page *rel* (posix, ROOT-relative).

    Pure path computation; parent directories are only created on write
    (:meth:`CorpusStore.save`).
    """
    return Path(corpus_dir) / f"{rel}{CORPUS_SUFFIX}"


def _write_json_atomic(path: Path, data: dict) -> None:
    """json.dump *data* to *path* via a tmp sibling + os.replace (fsynced)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class CorpusStore:
    """JSON read/write access to the per-page corpus files under one directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, rel: str, data: dict) -> Path:
        """Write *data* as the corpus file for *rel*; returns the file path.

        Atomic: a tmp sibling is written, fsynced, then os.replace-d into place.
        """
        target = corpus_path(self.path, rel)
        _write_json_atomic(target, data)
        return target

    def load(self, rel: str) -> dict | None:
        """Parsed corpus for *rel*; None when absent, CorruptCorpus when unparseable."""
        target = corpus_path(self.path, rel)
        if not target.exists():
            return None
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CorruptCorpus(target, str(exc)) from exc
        if not isinstance(data, dict):
            raise CorruptCorpus(target, f"expected a JSON object, got {type(data).__name__}")
        return data

    def exists(self, rel: str) -> bool:
        """Whether the corpus file for *rel* is present."""
        return corpus_path(self.path, rel).is_file()

    def missing_or_corrupt(self, rel: str) -> bool:
        """True when *rel* has no usable corpus file (missing or unparseable)."""
        if not self.exists(rel):
            return True
        try:
            self.load(rel)
        except CorruptCorpus:
            return True
        return False

    def iterate_all(self) -> Iterator[tuple[str, dict]]:
        """Yield (rel, data) for every corpus file, sorted by rel.

        *rel* is the path below the corpus dir with the .rag.json suffix stripped,
        in posix form. Corrupt or non-object files are skipped with a logged warning.
        """
        if not self.path.is_dir():
            return
        files = sorted(
            (Path(dirpath) / name
             for dirpath, _dirnames, filenames in os.walk(self.path)
             for name in filenames if name.endswith(CORPUS_SUFFIX)),
            key=lambda p: p.relative_to(self.path).as_posix(),
        )
        for full in files:
            rel = full.relative_to(self.path).as_posix().removesuffix(CORPUS_SUFFIX)
            try:
                data = json.loads(full.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                log.warning("skipping corrupt corpus file %s: %s", full, exc)
                continue
            if not isinstance(data, dict):
                log.warning("skipping non-object corpus file %s", full)
                continue
            yield rel, data

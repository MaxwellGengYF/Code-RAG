"""Shared pytest fixtures for the RAG test suite (all network-free)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def repo_root() -> Path:
    return ROOT


@pytest.fixture()
def tmp_site(tmp_path: Path) -> Path:
    """A tiny fake doc mirror: <tmp>/Manual/*.html + <tmp>/ScriptReference/*.html."""
    manual = tmp_path / "Manual"
    sr = tmp_path / "ScriptReference"
    manual.mkdir()
    sr.mkdir()
    (manual / "index.html").write_text(
        "<html><head><title>Unity - Manual: Test Page</title></head>"
        "<body><div id='content-wrap'><div class='section'>"
        "<h1>Test Page</h1><p>Hello world body.</p>"
        "</div></div></body></html>", encoding="utf-8")
    (sr / "Rigidbody.html").write_text(
        "<html><head><title>Unity - Scripting API: Rigidbody</title></head>"
        "<body><div id='content-wrap'><div class='section'>"
        "<h1>Rigidbody</h1>"
        "<div class='sig-block'>public Vector3 velocity;</div>"
        "<p>Control the velocity of the rigidbody.</p>"
        "</div></div></body></html>", encoding="utf-8")
    return tmp_path

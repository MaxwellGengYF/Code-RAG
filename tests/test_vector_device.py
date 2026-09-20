"""Embed-device selection tests (network-free: fake SentenceTransformer).

The BGE-M3 embedder must follow the hardware: cuda -> fp16 (fast enough to
embed the full corpus in minutes), cpu -> fp32. Both the batch path
(``embed_texts``/resumable build) and the query path (``embed_query``) load
the model through ``ensure_embed_model``, so device/dtype selection belongs
there and is what these tests pin down.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

import rag.index.vector_index as vi


class FakeST:
    """Records the ctor args SentenceTransformer was called with."""

    instances: list["FakeST"] = []

    def __init__(self, model, device=None, model_kwargs=None, **kw):
        self.model = model
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.encoded: list[list[str]] = []
        FakeST.instances.append(self)

    def encode(self, texts, **kw):
        texts = list(texts)
        self.encoded.append(texts)
        return np.ones((len(texts), 4), dtype=np.float32)


@pytest.fixture()
def fake_st(monkeypatch):
    """Swap in a fake SentenceTransformer and restore the model singleton after."""
    FakeST.instances.clear()
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    saved_model, saved_name = vi._MODEL, vi._MODEL_NAME
    saved_device = vi._MODEL_DEVICE
    vi._MODEL, vi._MODEL_NAME, vi._MODEL_DEVICE = None, None, None
    yield FakeST
    vi._MODEL, vi._MODEL_NAME = saved_model, saved_name
    vi._MODEL_DEVICE = saved_device


def test_cuda_selects_fp16(fake_st, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert vi.select_embed_device() == "cuda"

    vi.ensure_embed_model("fake-model")
    inst = fake_st.instances[-1]
    assert inst.device == "cuda"
    assert inst.model_kwargs.get("dtype") == torch.float16
    assert vi.embed_device() == "cuda"


def test_cpu_falls_back_to_fp32(fake_st, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert vi.select_embed_device() == "cpu"

    vi.ensure_embed_model("fake-model")
    inst = fake_st.instances[-1]
    assert inst.device == "cpu"
    assert inst.model_kwargs.get("dtype") == torch.float32
    assert vi.embed_device() == "cpu"


def test_model_singleton_not_reloaded_per_dtype(fake_st, monkeypatch):
    """Both embed paths share one model load; a second call must not reload."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    vi.ensure_embed_model("fake-model")
    vi.embed_query("Rigidbody.AddForce", model="fake-model")
    vi.embed_texts(["some passage"], model="fake-model")
    assert len(fake_st.instances) == 1, "each embed call must reuse the singleton"


def test_embed_query_uses_instruction_prefix(fake_st, monkeypatch):
    """Query path keeps the BGE convention: prefix on the query, none on passages."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    vi.embed_query("Rigidbody.AddForce")
    (texts,) = fake_st.instances[-1].encoded
    assert texts[0] == vi.BGE_QUERY_INSTRUCTION + "Rigidbody.AddForce"

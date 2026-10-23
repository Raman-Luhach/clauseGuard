"""LLM explanations and the zero-shot comparison, with a fake Anthropic client (no API calls)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from clauseguard.explain.llm import Clause, Explainer, LLMSettings, acceptable, request_params
from clauseguard.pipeline import analyze, get_analyzer, get_explainer
from tests.test_pipeline import TOS_TEXT

ON = LLMSettings(provider="anthropic")


class FakeClient:
    """Mimics ``client.beta.messages.create``; ``reply(params)`` builds the JSON answer."""

    def __init__(self, reply, stop_reason="end_turn", error=None):
        self.reply, self.stop_reason, self.error = reply, stop_reason, error
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **params):
        self.calls.append(params)
        if self.error:
            raise self.error
        text = json.dumps(self.reply(params))
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


def _clauses(params) -> list[dict]:
    return json.loads(params["messages"][0]["content"].split("\n", 1)[1])


def explain_all(params):
    """A valid answer for every clause; ids become letters so no number is invented."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    return {
        "explanations": [
            {"id": c["id"], "explanation": f"In plain words: clause {letters[c['id'] % 26]}."}
            for c in _clauses(params)
        ]
    }


def test_request_uses_schema_effort_and_fallback():
    params = request_params(ON, [Clause("We may end your account.", "Termination", "def")])
    assert params["model"] == "claude-opus-5-5"
    assert params["output_config"]["effort"] == "low"
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["fallbacks"] == "default"
    assert params["betas"] == ["server-side-fallback-2026-07-01"]
    haiku = request_params(LLMSettings(provider="anthropic", model="claude-haiku-5-5"), [])
    assert "fallbacks" not in haiku and "betas" not in haiku  # Haiku has no server-side fallback


def test_acceptable():
    clause = "Fees of $5,000 are due within 30 days."
    assert acceptable("You must pay $5,000 within 30 days.", clause)
    assert not acceptable("You must pay $6,000 within 30 days.", clause)  # invented number
    assert not acceptable("Pay.", clause)
    assert not acceptable("word " * 80, clause)


def test_explainer_caches_and_validates():
    def reply(params):
        out = explain_all(params)
        out["explanations"][-1]["explanation"] = "Costs 999 dollars."  # fails validation
        return out

    client = FakeClient(reply)
    explainer = Explainer(ON, client)
    clauses = [Clause(f"Clause number {i} text here.", "Cat", "def") for i in range(3)]
    first = explainer.explain(clauses)
    assert first[:2] == ["In plain words: clause a.", "In plain words: clause b."]
    assert first[2] is None  # invented number -> template
    assert explainer.explain(clauses[:2]) == first[:2] and len(client.calls) == 1  # cached


@pytest.mark.parametrize(
    "client",
    [FakeClient(explain_all, stop_reason="refusal"), FakeClient(explain_all, error=TimeoutError())],
)
def test_failures_fall_back_to_templates(client):
    assert Explainer(ON, client).explain([Clause("Some clause text.", "Cat", "def")]) == [None]


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CLAUSEGUARD_LLM", raising=False)
    client = FakeClient(explain_all)
    assert Explainer(LLMSettings.from_env(), client).explain([Clause("x y z", "c", "d")]) == [None]
    assert client.calls == []


def test_analysis_uses_llm_explanations(trained_tos, monkeypatch):
    client = FakeClient(explain_all)
    monkeypatch.setattr(get_explainer(), "settings", ON)
    monkeypatch.setattr(get_explainer(), "_client", client)
    get_analyzer.cache_clear()
    result = analyze("tos", text=TOS_TEXT)
    assert result.findings and all(f.explanation_source == "llm" for f in result.findings)
    assert all(f.explanation.startswith("In plain words") for f in result.findings)
    sent = _clauses(client.calls[0])
    assert {c["clause"] for c in sent} == {f.text for f in result.findings}  # only flagged clauses
    assert "ACME TERMS OF SERVICE" not in json.dumps(sent)


def test_llm_compare_with_fake_client(trained_tos):
    from clauseguard.config import load_module_config
    from clauseguard.evaluation.llm_compare import llm_compare

    gold = {}
    from clauseguard.data.units import load_units

    units = load_units("tos")
    for text, labels in zip(units["text"], units["labels"], strict=True):
        gold[text] = list(labels)

    def oracle(params):
        units_sent = json.loads(params["messages"][0]["content"])["units"]
        return {"items": [{"id": u["id"], "labels": gold[u["text"]]} for u in units_sent]}

    report = llm_compare("tos", n=40, settings=ON, client=FakeClient(oracle), log=lambda *_: None)
    assert report["llm"]["macro_f1"] == [1.0, 1.0]
    assert report["llm"]["consistency_identical_label_sets"] == 1.0
    assert report["llm"]["input_tokens"] == 2 * 2 * 100  # 2 batches of 20, asked twice
    assert set(report["classifier"]["per_label_f1"]) == set(load_module_config("tos").label_ids)
    with pytest.raises(ValueError, match="--final"):
        llm_compare("tos", split="test", settings=ON, client=FakeClient(oracle))

"""Optional plain-language explanations from Claude, with the templates as the fallback.

Only the flagged clauses are sent (never the whole document), together with the category the
classifier assigned and its definition. The model never decides whether a clause is risky; it
only rewords what the classifier found. One request covers a whole document, and its JSON
output is constrained by a schema. Every explanation is checked before use (length, and no
numbers that are not in the clause); anything that fails, and any API error, timeout or
refusal, falls back to the template explanation.

Configuration (environment variables):
  CLAUSEGUARD_LLM          "anthropic" to enable; anything else (default "none") disables it
  CLAUSEGUARD_LLM_MODEL    default "claude-opus-5-5"
  CLAUSEGUARD_LLM_EFFORT   default "low" (one or two plain sentences need little reasoning)
  CLAUSEGUARD_LLM_TIMEOUT  seconds per request, default 30
  CLAUSEGUARD_LLM_MAX      most clauses explained per document, default 30 (most severe first)
Credentials come from the usual Anthropic sources (ANTHROPIC_API_KEY, `ant auth login`, ...).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
# Models that accept Anthropic's server-side refusal fallback (`fallbacks: "default"`).
SERVER_FALLBACK_MODELS = {
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-fable-5-1",
    "claude-sonnet-5-5",
}
MAX_WORDS = 60
CACHE_SIZE = 2048

SYSTEM = """You explain clauses from legal documents (terms of service, privacy policies and \
contracts) to people without legal training.

For each clause you receive its id, its text, the category a classifier assigned, and that \
category's definition. Write one or two short sentences (at most 45 words) that say what the \
clause means for the reader in everyday English.

Rules:
- Use only what the clause says. Do not add facts, numbers, names, places or time periods that \
are not in the clause.
- Do not give legal advice and do not say whether the clause is legal or enforceable.
- Do not argue with the category; if the clause only loosely matches it, explain what the \
clause actually says.
- Address the reader as "you" and the company or other party as "the company" or "the other \
party", unless the clause names them.
- Return an explanation for every id you were given."""

SCHEMA = {
    "type": "object",
    "properties": {
        "explanations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "explanation": {"type": "string"},
                },
                "required": ["id", "explanation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["explanations"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "none"
    model: str = DEFAULT_MODEL
    effort: str = "low"
    timeout: float = 30.0
    max_clauses: int = 30

    @classmethod
    def from_env(cls) -> LLMSettings:
        return cls(
            provider=os.environ.get("CLAUSEGUARD_LLM", "none").strip().lower(),
            model=os.environ.get("CLAUSEGUARD_LLM_MODEL", DEFAULT_MODEL),
            effort=os.environ.get("CLAUSEGUARD_LLM_EFFORT", "low"),
            timeout=float(os.environ.get("CLAUSEGUARD_LLM_TIMEOUT", "30")),
            max_clauses=int(os.environ.get("CLAUSEGUARD_LLM_MAX", "30")),
        )

    @property
    def enabled(self) -> bool:
        return self.provider == "anthropic"


@dataclass(frozen=True)
class Clause:
    text: str
    category: str
    definition: str


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def acceptable(explanation: str, clause: str) -> bool:
    """Short enough, and every number it mentions appears in the clause."""
    words = explanation.split()
    if not 3 <= len(words) <= MAX_WORDS:
        return False
    clause_numbers = {n.replace(",", "") for n in _NUMBER.findall(clause)}
    return all(n.replace(",", "") in clause_numbers for n in _NUMBER.findall(explanation))


def request_params(settings: LLMSettings, clauses: list[Clause]) -> dict:
    """Arguments for ``client.beta.messages.create`` (kept separate so tests can check them)."""
    payload = [
        {"id": i, "clause": c.text, "category": c.category, "definition": c.definition}
        for i, c in enumerate(clauses)
    ]
    params = {
        "model": settings.model,
        "max_tokens": 16000,
        "system": SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": "Explain these clauses:\n" + json.dumps(payload, ensure_ascii=False),
            }
        ],
        "output_config": {
            "effort": settings.effort,
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
    }
    if settings.model in SERVER_FALLBACK_MODELS:
        # Anthropic's recommended fallback model takes over if a safety classifier declines.
        params["betas"] = ["server-side-fallback-2026-07-01"]
        params["fallbacks"] = "default"
    return params


class Explainer:
    def __init__(self, settings: LLMSettings | None = None, client=None) -> None:
        self.settings = settings or LLMSettings.from_env()
        self._client = client
        self._cache: OrderedDict[str, str] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(timeout=self.settings.timeout, max_retries=1)
        return self._client

    def _key(self, clause: Clause) -> str:
        raw = f"{self.settings.model}\x00{clause.category}\x00{clause.text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def explain(self, clauses: list[Clause]) -> list[str | None]:
        """One explanation per clause, or None where the template should be used."""
        results: list[str | None] = [None] * len(clauses)
        if not self.enabled or not clauses:
            return results
        todo = []
        for i, clause in enumerate(clauses[: self.settings.max_clauses]):
            cached = self._cache.get(self._key(clause))
            if cached is not None:
                results[i] = cached
            else:
                todo.append(i)
        if not todo:
            return results
        try:
            answers = self._call([clauses[i] for i in todo])
        except Exception as exc:  # network, auth, rate limit, timeout, bad JSON: use templates
            log.warning(
                "LLM explanations unavailable (%s: %s); using templates", type(exc).__name__, exc
            )
            return results
        for local_id, explanation in answers.items():
            if not 0 <= local_id < len(todo):
                continue
            i = todo[local_id]
            text = " ".join(explanation.split())
            if acceptable(text, clauses[i].text):
                results[i] = text
                self._cache[self._key(clauses[i])] = text
                while len(self._cache) > CACHE_SIZE:
                    self._cache.popitem(last=False)
        return results

    def _call(self, clauses: list[Clause]) -> dict[int, str]:
        response = self._get_client().beta.messages.create(**request_params(self.settings, clauses))
        if response.stop_reason != "end_turn":  # refusal, max_tokens, ...
            raise RuntimeError(f"stop_reason={response.stop_reason}")
        text = next(block.text for block in response.content if block.type == "text")
        return {
            int(item["id"]): str(item["explanation"]) for item in json.loads(text)["explanations"]
        }

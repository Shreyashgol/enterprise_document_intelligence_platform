"""Phase 14 — Answer generators for RAG.

The generator is the "G" in RAG: given a question and retrieved context, produce
an answer. Two implementations behind one interface (``generate(question,
contexts) -> str``):

* `ExtractiveGenerator` — no LLM. Selects the most relevant sentence(s) from the
  retrieved context by lexical overlap with the question. Deterministic, free,
  fully testable offline — the honest fallback and the default in tests.
* `GroqGenerator` — calls the Groq API (`llama-3.3-70b-versatile`) to synthesize
  a grounded answer over the context. Activated when a ``GROQ_API_KEY`` is
  available; the client is injectable so the prompt construction is unit-tested
  without a network call.

Both are *grounded*: the answer must come from the provided context, not the
model's parametric memory — that is the whole point of RAG (reduce hallucination,
cite sources).
"""

from __future__ import annotations

import re
from typing import Optional, Protocol, Sequence, runtime_checkable

_WORD = re.compile(r"[A-Za-z0-9']+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

NO_ANSWER = "I don't know based on the provided documents."

# Common function words carry no topical signal; counting them as "overlap"
# produces false matches (e.g. a query and a doc sharing only "in"/"the").
STOPWORDS: frozenset[str] = frozenset(
    "a an the of to in on at for with and or but is are was were be been being "
    "do did does this that these those it its as by from into out i you he she "
    "we they who whom which what when where why how their our your his her".split()
)


def _tokens(text: str) -> set[str]:
    """Content-word token set: lowercased, stopwords removed."""
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in STOPWORDS}


@runtime_checkable
class Generator(Protocol):
    name: str

    def generate(self, question: str, contexts: Sequence[str]) -> str: ...


class ExtractiveGenerator:
    """Pick the best-matching sentence(s) from the retrieved context."""

    name = "extractive"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        q = _tokens(question)
        if not q:
            return NO_ANSWER
        scored: list[tuple[float, str]] = []
        for ctx in contexts:
            for sent in _SENT_SPLIT.split(ctx.strip()):
                sent = sent.strip()
                if not sent:
                    continue
                overlap = len(q & _tokens(sent))
                if overlap:
                    scored.append((overlap / len(q), sent))
        if not scored:
            return NO_ANSWER
        scored.sort(key=lambda x: -x[0])
        top = [s for _, s in scored[: self.max_sentences]]
        return " ".join(top)


class GroqGenerator:
    """Synthesize a grounded answer with the Groq API (Llama 3.3 70B).

    Groq serves an OpenAI-compatible chat-completions API; the answer is read
    from ``response.choices[0].message.content``.

    Args:
        model:      Groq model id (default ``llama-3.3-70b-versatile``).
        client:     a ``groq.Groq``-like client. If None, one is created lazily
                    (reads ``GROQ_API_KEY`` from the env).
        max_tokens: response cap.
        temperature: low by default for faithful, grounded answers.
    """

    name = "groq"

    SYSTEM = (
        "You are a precise enterprise document assistant. Answer the user's "
        "question using ONLY the provided context passages. If the answer is not "
        "contained in the context, say you don't know — do not use outside "
        "knowledge. Cite the passages you used as [n]."
    )

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        client: Optional[object] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self._client = client
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _get_client(self):
        if self._client is None:
            from groq import Groq  # lazy: only needed for live calls

            self._client = Groq()
        return self._client

    @staticmethod
    def _format_context(contexts: Sequence[str]) -> str:
        return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        if not contexts:
            return NO_ANSWER
        client = self._get_client()
        user = (
            f"Context passages:\n{self._format_context(contexts)}\n\n"
            f"Question: {question}"
        )
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip() or NO_ANSWER

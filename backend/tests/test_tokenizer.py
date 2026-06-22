"""Unit tests for Phase 3 — tokenizer.

Run with::

    cd backend && python -m pytest tests/test_tokenizer.py -v
"""

from __future__ import annotations

import pytest

from app.tokenizer.tokenizer import Tokenizer, WhitespaceTokenizer, Token


@pytest.fixture
def tok():
    return Tokenizer()


def _texts(tokens):
    return [t.text for t in tokens]


# ---------------------------------------------------------------------------
# V1 — Whitespace
# ---------------------------------------------------------------------------
class TestWhitespace:
    def test_split_matches_stdlib(self):
        wt = WhitespaceTokenizer()
        assert wt.tokens("a b  c") == "a b  c".split()

    def test_offsets_recovered(self):
        wt = WhitespaceTokenizer()
        text = "John  works"
        toks = wt.tokenize(text)
        assert _texts(toks) == ["John", "works"]
        for t in toks:
            assert text[t.start : t.end] == t.text

    def test_glues_punctuation(self):
        # demonstrates V1's weakness that V2 fixes
        assert WhitespaceTokenizer().tokens("Hello, world!") == ["Hello,", "world!"]


# ---------------------------------------------------------------------------
# V2 — offset invariant (the most important property)
# ---------------------------------------------------------------------------
class TestOffsetInvariant:
    @pytest.mark.parametrize(
        "text",
        [
            "John Smith works at OpenAI.",
            "Email a.b+x@mail.co.uk or call +1 (555) 123-4567 by Jan. 5, 2024.",
            "Café paid €50 for CloudSync Pro — that's 1,000.00 USD.",
            "Dr. Lee (U.S.A.) said e.g. don't worry, etc.",
            "",
            "   ",
        ],
    )
    def test_invariant_holds(self, tok, text):
        for t in tok.tokenize(text):
            assert text[t.start : t.end] == (
                t.text if not tok.lowercase else text[t.start : t.end]
            )

    def test_tokens_are_sorted_and_non_overlapping(self, tok):
        text = "Email a@b.com or call 555-123-4567 today."
        toks = tok.tokenize(text)
        starts = [t.start for t in toks]
        assert starts == sorted(starts)
        for prev, nxt in zip(toks, toks[1:]):
            assert prev.end <= nxt.start


# ---------------------------------------------------------------------------
# V2 — punctuation
# ---------------------------------------------------------------------------
class TestPunctuation:
    def test_comma_and_bang_split(self, tok):
        assert _texts(tok.tokenize("Hello, world!")) == ["Hello", ",", "world", "!"]

    def test_parentheses(self, tok):
        assert _texts(tok.tokenize("(hi)")) == ["(", "hi", ")"]

    def test_currency_symbol_separated_from_number(self, tok):
        assert _texts(tok.tokenize("$2.5")) == ["$", "2.5"]


# ---------------------------------------------------------------------------
# V2 — emails kept atomic
# ---------------------------------------------------------------------------
class TestEmail:
    def test_email_is_single_token(self, tok):
        toks = tok.tokenize("write to john.doe+x@mail.corp.co.uk now")
        emails = [t for t in toks if t.kind == "EMAIL"]
        assert len(emails) == 1
        assert emails[0].text == "john.doe+x@mail.corp.co.uk"

    def test_email_not_shattered_by_dots(self, tok):
        assert "a@b.com" in _texts(tok.tokenize("a@b.com."))


# ---------------------------------------------------------------------------
# V2 — phones kept atomic
# ---------------------------------------------------------------------------
class TestPhone:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("call 555-123-4567 now", "555-123-4567"),
            ("call +1 (555) 123-4567 now", "+1 (555) 123-4567"),
            ("call +44 20 7946 0958 now", "+44 20 7946 0958"),
        ],
    )
    def test_phone_single_token(self, tok, text, expected):
        phones = [t.text for t in tok.tokenize(text) if t.kind == "PHONE"]
        assert phones == [expected]


# ---------------------------------------------------------------------------
# V2 — abbreviations kept atomic
# ---------------------------------------------------------------------------
class TestAbbreviation:
    def test_title_abbrev(self, tok):
        assert "Dr." in _texts(tok.tokenize("Dr. Lee arrived"))

    def test_dotted_acronym(self, tok):
        assert "U.S.A." in _texts(tok.tokenize("from U.S.A. today"))

    def test_eg_ie(self, tok):
        toks = _texts(tok.tokenize("e.g. this and i.e. that"))
        assert "e.g." in toks and "i.e." in toks

    def test_company_suffix(self, tok):
        assert "Inc." in _texts(tok.tokenize("Acme Inc. signed"))

    def test_without_period_is_plain_word(self, tok):
        # "Dr" with no period should NOT be treated as abbreviation glue
        toks = tok.tokenize("Dr Lee")
        assert _texts(toks) == ["Dr", "Lee"]


# ---------------------------------------------------------------------------
# V2 — numbers, words, options
# ---------------------------------------------------------------------------
class TestNumbersWordsOptions:
    def test_number_with_separators(self, tok):
        assert "1,000.00" in _texts(tok.tokenize("paid 1,000.00 total"))

    def test_contraction_kept(self, tok):
        assert "don't" in _texts(tok.tokenize("I don't know"))

    def test_hyphenated_word(self, tok):
        assert "state-of-the-art" in _texts(tok.tokenize("a state-of-the-art model"))

    def test_alnum_word(self, tok):
        assert "B2B" in _texts(tok.tokenize("a B2B deal"))

    def test_lowercase_option(self):
        t = Tokenizer(lowercase=True)
        assert t.tokens("Hello WORLD") == ["hello", "world"]

    def test_lowercase_preserves_offsets(self):
        t = Tokenizer(lowercase=True)
        text = "Hello"
        tok0 = t.tokenize(text)[0]
        assert text[tok0.start : tok0.end] == "Hello"  # source slice unchanged
        assert tok0.text == "hello"

    def test_empty_string(self, tok):
        assert tok.tokenize("") == []

    def test_unicode_word(self, tok):
        # accented letters: ensure we don't crash and offsets stay valid
        toks = tok.tokenize("Café €50")
        for t in toks:
            assert "Café €50"[t.start : t.end] == t.text


# ---------------------------------------------------------------------------
# Integration: tokenizer feeds Phase 2 spans (preview of Phase 4 alignment)
# ---------------------------------------------------------------------------
class TestSpanAlignmentReadiness:
    def test_every_char_of_an_entity_is_covered_by_tokens(self, tok):
        text = "Reach john@acme.com now"
        # entity span for the email
        e_start, e_end = text.index("john@acme.com"), text.index("john@acme.com") + len("john@acme.com")
        covering = [t for t in tok.tokenize(text) if t.start >= e_start and t.end <= e_end]
        assert covering and covering[0].text == "john@acme.com"

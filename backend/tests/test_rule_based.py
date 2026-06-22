"""Unit tests for Phase 1 rule-based extraction.

Run with::

    cd backend && python -m pytest tests/test_rule_based.py -v
"""

from __future__ import annotations

import pytest

from app.ner.rule_based import (
    extract_email,
    extract_phone,
    extract_date,
    extract_money,
    extract_entities,
    extract_all,
)


def _texts(entities):
    return [e.text for e in entities]


# ---------------------------------------------------------------------------
# EMAIL
# ---------------------------------------------------------------------------
class TestEmail:
    def test_simple(self):
        ents = extract_email("Contact john.doe@example.com please")
        assert _texts(ents) == ["john.doe@example.com"]
        assert ents[0].normalized == "john.doe@example.com"

    def test_plus_and_subdomain(self):
        ents = extract_email("Reach me at a.b+tag@mail.corp.co.uk now")
        assert _texts(ents) == ["a.b+tag@mail.corp.co.uk"]

    def test_uppercase_is_normalized(self):
        ents = extract_email("JOHN@EXAMPLE.COM")
        assert ents[0].normalized == "john@example.com"

    def test_span_is_exact(self):
        text = "x foo@bar.com y"
        e = extract_email(text)[0]
        assert text[e.start : e.end] == "foo@bar.com"

    def test_multiple(self):
        ents = extract_email("a@x.com, b@y.org")
        assert _texts(ents) == ["a@x.com", "b@y.org"]

    @pytest.mark.parametrize("bad", ["plainword", "no@tld", "@nolocal.com", "a@b"])
    def test_negatives(self, bad):
        assert extract_email(bad) == []

    def test_not_grabbed_mid_token(self):
        # should not match inside a URL-ish blob with no real domain TLD
        assert extract_email("user@localhost") == []


# ---------------------------------------------------------------------------
# PHONE
# ---------------------------------------------------------------------------
class TestPhone:
    @pytest.mark.parametrize(
        "text,expected_digits",
        [
            ("Call +1 (555) 123-4567 today", "+15551234567"),
            ("555-123-4567", "5551234567"),
            ("555.123.4567", "5551234567"),
            ("+44 20 7946 0958", "+442079460958"),
            ("(020) 7946 0958", "02079460958"),
        ],
    )
    def test_formats(self, text, expected_digits):
        ents = extract_phone(text)
        assert len(ents) == 1
        assert ents[0].normalized == expected_digits

    def test_too_few_digits_rejected(self):
        assert extract_phone("12-34") == []

    def test_too_many_digits_rejected(self):
        assert extract_phone("1234567890123456789") == []

    def test_span_exact(self):
        text = "ph: 555-123-4567."
        e = extract_phone(text)[0]
        assert text[e.start : e.end] == "555-123-4567"


# ---------------------------------------------------------------------------
# DATE
# ---------------------------------------------------------------------------
class TestDate:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Signed on 2024-01-15.", "2024-01-15"),
            ("Due 01/15/2024 sharp", "01/15/2024"),
            ("on 15-01-2024", "15-01-2024"),
            ("dated January 15, 2024 here", "January 15, 2024"),
            ("by 15 January 2024", "15 January 2024"),
            ("in Dec 2023", "Dec 2023"),
            ("the 3rd of nonsense", None),  # no year -> no match
        ],
    )
    def test_formats(self, text, expected):
        ents = extract_date(text)
        if expected is None:
            assert ents == []
        else:
            assert _texts(ents) == [expected]

    def test_multiple_dates(self):
        ents = extract_date("from 2024-01-01 to 2024-12-31")
        assert _texts(ents) == ["2024-01-01", "2024-12-31"]


# ---------------------------------------------------------------------------
# MONEY
# ---------------------------------------------------------------------------
class TestMoney:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Paid $1,000.00 total", "$1,000.00"),
            ("worth $1.5M", "$1.5M"),
            ("USD 1,000 received", "USD 1,000"),
            ("about 1000 dollars", "1000 dollars"),
            ("costs €50", "€50"),
            ("£2.3 billion deal", "£2.3 billion"),
        ],
    )
    def test_formats(self, text, expected):
        ents = extract_money(text)
        assert _texts(ents) == [expected]

    def test_plain_number_not_money(self):
        assert extract_money("there were 42 boxes") == []


# ---------------------------------------------------------------------------
# AGGREGATION
# ---------------------------------------------------------------------------
class TestExtractEntities:
    def test_contract_shape(self):
        result = extract_entities("hi")
        assert set(result.keys()) == {"emails", "phones", "dates", "money"}
        assert all(isinstance(v, list) for v in result.values())

    def test_realistic_document(self):
        text = (
            "On 2024-01-15, ACME signed a deal worth $2.5M. "
            "Email contracts@acme.com or call +1 (555) 987-6543."
        )
        result = extract_entities(text)
        assert result["emails"][0]["text"] == "contracts@acme.com"
        assert result["dates"][0]["text"] == "2024-01-15"
        assert result["money"][0]["text"] == "$2.5M"
        assert result["phones"][0]["normalized"] == "+15559876543"

    def test_records_are_dicts_with_spans(self):
        result = extract_entities("mail me a@b.com")
        rec = result["emails"][0]
        assert {"text", "label", "start", "end", "normalized", "source"} <= rec.keys()

    def test_extract_all_is_non_overlapping_sorted(self):
        text = "On 2024-01-15 pay $5,000 to a@b.com at 555-123-4567."
        ents = extract_all(text)
        starts = [e.start for e in ents]
        assert starts == sorted(starts)
        for prev, nxt in zip(ents, ents[1:]):
            assert prev.end <= nxt.start


class TestEdgeCases:
    def test_empty_string(self):
        assert extract_entities("") == {
            "emails": [],
            "phones": [],
            "dates": [],
            "money": [],
        }

    def test_unicode_text(self):
        result = extract_entities("Café paid €50 on 2024-06-01")
        assert result["money"][0]["text"] == "€50"
        assert result["dates"][0]["text"] == "2024-06-01"

    def test_no_false_positive_on_prose(self):
        result = extract_entities("The quick brown fox jumps over the lazy dog.")
        assert all(v == [] for v in result.values())

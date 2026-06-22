"""Unit tests for Phase 5 — vocabulary builder.

Run with::

    cd backend && python -m pytest tests/test_vocabulary.py -v
"""

from __future__ import annotations

import pytest

from app.datasets.vocabulary import (
    Vocabulary,
    VocabularyBuilder,
    build_tag_vocabulary,
    PAD_TOKEN,
    UNK_TOKEN,
)


# ---------------------------------------------------------------------------
# Specials & basic mapping
# ---------------------------------------------------------------------------
class TestSpecials:
    def test_pad_is_zero_unk_is_one(self):
        vocab = VocabularyBuilder().fit([["a", "b", "c"]]).build()
        assert vocab.pad_id == 0
        assert vocab.unk_id == 1
        assert vocab.decode(0) == PAD_TOKEN
        assert vocab.decode(1) == UNK_TOKEN

    def test_word2idx_and_idx2word_are_inverse(self):
        vocab = VocabularyBuilder().fit([["x", "y"]]).build()
        for tok, idx in vocab.word2idx.items():
            assert vocab.idx2word[idx] == tok

    def test_len_counts_specials(self):
        vocab = VocabularyBuilder().fit([["a", "b"]]).build()
        assert len(vocab) == 4  # PAD, UNK, a, b


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------
class TestEncodeDecode:
    def test_known_token(self):
        vocab = VocabularyBuilder().fit([["hello", "world"]]).build()
        assert vocab.decode(vocab.encode("hello")) == "hello"

    def test_unknown_maps_to_unk(self):
        vocab = VocabularyBuilder().fit([["hello"]]).build()
        assert vocab.encode("zzz") == vocab.unk_id

    def test_no_unk_raises_on_unknown(self):
        tagv = build_tag_vocabulary()  # no UNK
        with pytest.raises(KeyError):
            tagv.encode("B-NOTREAL")

    def test_decode_out_of_range(self):
        vocab = VocabularyBuilder().fit([["a"]]).build()
        with pytest.raises(IndexError):
            vocab.decode(999)

    def test_contains(self):
        vocab = VocabularyBuilder().fit([["apple"]]).build()
        assert "apple" in vocab and "banana" not in vocab


# ---------------------------------------------------------------------------
# Sequence encode/decode + padding
# ---------------------------------------------------------------------------
class TestSequences:
    def test_pad_to_max_len(self):
        vocab = VocabularyBuilder().fit([["a", "b"]]).build()
        ids = vocab.encode_sequence(["a"], max_len=4)
        assert len(ids) == 4
        assert ids[1:] == [vocab.pad_id] * 3

    def test_truncate_to_max_len(self):
        vocab = VocabularyBuilder().fit([["a", "b", "c"]]).build()
        ids = vocab.encode_sequence(["a", "b", "c"], max_len=2)
        assert len(ids) == 2

    def test_decode_strip_pad(self):
        vocab = VocabularyBuilder().fit([["a"]]).build()
        ids = vocab.encode_sequence(["a"], max_len=3)
        assert vocab.decode_sequence(ids, strip_pad=True) == ["a"]


# ---------------------------------------------------------------------------
# Frequency cutoff & ordering determinism
# ---------------------------------------------------------------------------
class TestBuildOptions:
    def test_min_freq_filters_rare(self):
        vb = VocabularyBuilder().fit([["a", "a", "b"]])  # a:2, b:1
        vocab = vb.build(min_freq=2)
        assert "a" in vocab and "b" not in vocab

    def test_ordering_by_frequency_then_alpha(self):
        vb = VocabularyBuilder().fit([["b", "a", "a", "c", "c"]])  # a:2,c:2,b:1
        vocab = vb.build()
        # specials first, then a,c (freq2, alpha) then b
        assert vocab.decode(2) == "a"
        assert vocab.decode(3) == "c"
        assert vocab.decode(4) == "b"

    def test_max_size_keeps_most_frequent(self):
        vb = VocabularyBuilder().fit([["a", "a", "a", "b", "b", "c"]])
        vocab = vb.build(max_size=3)  # PAD, UNK, a
        assert len(vocab) == 3
        assert "a" in vocab and "b" not in vocab

    def test_lowercase_folds_case(self):
        vb = VocabularyBuilder(lowercase=True).fit([["Apple", "apple"]])
        vocab = vb.build()
        assert vocab.encode("APPLE") == vocab.encode("apple")
        assert len(vocab) == 3  # PAD, UNK, apple

    def test_update_accumulates(self):
        vb = VocabularyBuilder()
        vb.update(["a", "b"]).update(["a"])
        vocab = vb.build()
        assert "a" in vocab and "b" in vocab


# ---------------------------------------------------------------------------
# Tag vocabulary
# ---------------------------------------------------------------------------
class TestTagVocabulary:
    def test_pad_then_tags(self):
        tagv = build_tag_vocabulary()
        assert tagv.pad_id == 0
        assert tagv.decode(0) == PAD_TOKEN
        assert tagv.decode(1) == "O"
        assert len(tagv) == 1 + 17  # PAD + 17 BIO tags

    def test_no_unk(self):
        assert build_tag_vocabulary().unk_id is None

    def test_roundtrip_known_tags(self):
        tagv = build_tag_vocabulary()
        for tag in ["O", "B-PERSON", "I-ORG", "B-MONEY"]:
            assert tagv.decode(tagv.encode(tag)) == tag


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
class TestSerialization:
    def test_save_load_roundtrip(self, tmp_path):
        vocab = VocabularyBuilder(lowercase=True).fit([["a", "b", "b"]]).build()
        p = tmp_path / "vocab.json"
        vocab.save(p)
        loaded = Vocabulary.load(p)
        assert loaded == vocab
        assert loaded.encode("b") == vocab.encode("b")

    def test_to_from_dict(self):
        vocab = VocabularyBuilder().fit([["x"]]).build()
        assert Vocabulary.from_dict(vocab.to_dict()) == vocab

    def test_indices_stable_across_load(self, tmp_path):
        vocab = VocabularyBuilder().fit([["a", "a", "b"]]).build()
        p = tmp_path / "v.json"
        vocab.save(p)
        loaded = Vocabulary.load(p)
        for i in range(len(vocab)):
            assert loaded.decode(i) == vocab.decode(i)

    def test_from_tokens_explicit_order(self):
        vocab = Vocabulary.from_tokens(["cat", "dog", "cat"])
        assert vocab.decode(0) == PAD_TOKEN
        assert vocab.decode(1) == UNK_TOKEN
        assert vocab.decode(2) == "cat"
        assert vocab.decode(3) == "dog"
        assert len(vocab) == 4  # dedup


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_duplicate_tokens_rejected(self):
        with pytest.raises(ValueError):
            Vocabulary([PAD_TOKEN, UNK_TOKEN, "a", "a"])

    def test_missing_pad_rejected(self):
        with pytest.raises(ValueError):
            Vocabulary(["a", "b"], pad_token=PAD_TOKEN)

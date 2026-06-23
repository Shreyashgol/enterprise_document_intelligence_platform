"""Unit tests for Phase 7D — encoder + CRF.

Confirms the capstone model reuses the Phase 7B ``CRF`` verbatim, gathers
first-subword emissions into a clean word-level chain, and decodes to word level.

Uses the tiny random BERT from the HF test hub; skips gracefully offline.

Run with::

    cd backend && python -m pytest tests/test_bert_crf.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from app.ner.crf import CRF
from app.ner.bert_ner import BertNERConfig, load_tokenizer

TINY_MODEL = "hf-internal-testing/tiny-random-BertModel"


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return load_tokenizer(TINY_MODEL)
    except Exception as exc:
        pytest.skip(f"could not load {TINY_MODEL}: {exc}")


@pytest.fixture(scope="module")
def model():
    from app.ner.bert_crf import BertCRF

    torch.manual_seed(0)
    try:
        return BertCRF(BertNERConfig(num_tags=7, encoder_name=TINY_MODEL))
    except Exception as exc:
        pytest.skip(f"could not build BertCRF from {TINY_MODEL}: {exc}")


def _encode(tokenizer, sentences):
    enc = tokenizer(
        sentences, is_split_into_words=True, return_tensors="pt", padding=True
    )
    word_ids = [enc.word_ids(i) for i in range(len(sentences))]
    return enc, word_ids


# ---------------------------------------------------------------------------
# The reuse claim: same CRF class as 7B
# ---------------------------------------------------------------------------
class TestReuse:
    def test_uses_the_phase7b_crf(self, model):
        from app.ner.bilstm_crf import BiLSTMCRF
        from app.ner.model import NERModelConfig

        assert isinstance(model.crf, CRF)
        bilstm = BiLSTMCRF(NERModelConfig(vocab_size=10, num_tags=7))
        # identical class & API — only the emission source differs.
        assert type(model.crf) is type(bilstm.crf)


# ---------------------------------------------------------------------------
# Forward / loss / decode
# ---------------------------------------------------------------------------
class TestForward:
    def test_loss_is_scalar_nonneg(self, model, tokenizer):
        sents = [["OpenAI", "rocks"], ["hello", "big", "world"]]
        enc, word_ids = _encode(tokenizer, sents)
        word_tags = [[5, 0], [0, 0, 6]]
        loss = model(enc["input_ids"], enc["attention_mask"], word_ids, word_tags)
        assert loss.dim() == 0
        assert float(loss.detach()) >= -1e-4

    def test_decode_returns_word_level_paths(self, model, tokenizer):
        sents = [["OpenAI", "rocks"], ["hello", "big", "world"]]
        enc, word_ids = _encode(tokenizer, sents)
        paths = model.decode(enc["input_ids"], enc["attention_mask"], word_ids)
        # one tag per WORD (not per subword), regardless of subword splitting.
        assert [len(p) for p in paths] == [2, 3]
        assert all(0 <= t < 7 for p in paths for t in p)

    def test_word_tag_count_mismatch_raises(self, model, tokenizer):
        enc, word_ids = _encode(tokenizer, [["OpenAI", "rocks"]])
        with pytest.raises(ValueError):
            model(enc["input_ids"], enc["attention_mask"], word_ids, [[1, 2, 3]])

    def test_subword_count_exceeds_word_count(self, model, tokenizer):
        # sanity: the encoder really does split into more subwords than words,
        # so the gather step is doing non-trivial work.
        enc, word_ids = _encode(tokenizer, [["OpenAI", "rocks"]])
        n_sub = int(enc["attention_mask"][0].sum())
        n_words = len({w for w in word_ids[0] if w is not None})
        assert n_sub > n_words


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------
class TestGradients:
    def test_grads_flow_to_encoder_and_transitions(self, model, tokenizer):
        enc, word_ids = _encode(tokenizer, [["OpenAI", "rocks"]])
        loss = model(enc["input_ids"], enc["attention_mask"], word_ids, [[5, 0]])
        loss.backward()
        assert model.crf.transitions.grad is not None
        assert torch.any(model.crf.transitions.grad != 0)
        enc_grads = [p.grad for p in model.bert.encoder.parameters() if p.grad is not None]
        assert enc_grads and any(torch.any(g != 0) for g in enc_grads)


# ---------------------------------------------------------------------------
# End-to-end: overfit → valid BIO at word level
# ---------------------------------------------------------------------------
class TestOverfit:
    def test_overfits_to_valid_bio(self, model, tokenizer):
        # tag ids: 0=O, 5=B-ORG, 6=I-ORG (arbitrary but consistent here)
        sents = [["OpenAI", "Corp", "rocks"], ["hello", "world"]]
        word_tags = [[5, 6, 0], [0, 0]]
        enc, word_ids = _encode(tokenizer, sents)

        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(60):
            opt.zero_grad()
            loss = model(enc["input_ids"], enc["attention_mask"], word_ids, word_tags)
            loss.backward()
            opt.step()

        paths = model.decode(enc["input_ids"], enc["attention_mask"], word_ids)
        assert paths == word_tags
        # I-ORG(6) only ever follows B-ORG(5)/I-ORG(6): a valid BIO span.
        for seq in paths:
            prev = 0
            for t in seq:
                if t == 6:
                    assert prev in (5, 6)
                prev = t


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
class TestCheckpoint:
    def test_roundtrip(self, model, tokenizer, tmp_path):
        from app.ner.bert_crf import BertCRF

        enc, word_ids = _encode(tokenizer, [["OpenAI", "rocks"]])
        before = model.decode(enc["input_ids"], enc["attention_mask"], word_ids)

        p = tmp_path / "bert_crf.pt"
        model.save_checkpoint(p, extra={"epoch": 4})
        loaded, extra = BertCRF.load_checkpoint(p)
        after = loaded.decode(enc["input_ids"], enc["attention_mask"], word_ids)

        assert extra == {"epoch": 4}
        assert before == after
        assert loaded.config.to_dict() == model.config.to_dict()

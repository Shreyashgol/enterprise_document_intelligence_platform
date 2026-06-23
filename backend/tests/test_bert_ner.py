"""Unit tests for Phase 7C — encoder + linear NER + subword alignment.

The alignment tests are pure Python (no heavy deps). The model tests use a tiny
random BERT from the HuggingFace test hub; they ``importorskip`` transformers and
skip gracefully if the model can't be fetched (offline first run).

Run with::

    cd backend && python -m pytest tests/test_bert_ner.py -v
"""

from __future__ import annotations

import pytest

from app.ner.decode import (
    align_labels_to_subwords,
    first_subword_mask,
    gather_word_predictions,
    IGNORE_INDEX,
)

torch = pytest.importorskip("torch")

TINY_MODEL = "hf-internal-testing/tiny-random-BertModel"


# ---------------------------------------------------------------------------
# Subword↔word alignment  (the real subtlety — no transformers needed)
# ---------------------------------------------------------------------------
class TestAlignment:
    # "[CLS] open ##ai works [SEP]"  →  words: 0=OpenAI, 1=works
    word_ids = [None, 0, 0, 1, None]
    word_labels = [5, 0]  # e.g. B-ORG, O

    def test_first_subword_gets_label_rest_ignored(self):
        out = align_labels_to_subwords(self.word_labels, self.word_ids)
        assert out == [IGNORE_INDEX, 5, IGNORE_INDEX, 0, IGNORE_INDEX]

    def test_specials_are_ignore_index(self):
        out = align_labels_to_subwords(self.word_labels, self.word_ids)
        assert out[0] == IGNORE_INDEX and out[-1] == IGNORE_INDEX

    def test_custom_ignore_index(self):
        out = align_labels_to_subwords(self.word_labels, self.word_ids, ignore_index=-1)
        assert out == [-1, 5, -1, 0, -1]

    def test_first_subword_mask(self):
        assert first_subword_mask(self.word_ids) == [False, True, False, True, False]

    def test_roundtrip_is_identity(self):
        # word labels → subword labels → gather first-subword back to word level
        subword = align_labels_to_subwords(self.word_labels, self.word_ids)
        recovered = gather_word_predictions(subword, self.word_ids)
        assert recovered == self.word_labels

    def test_gather_picks_first_subword_prediction(self):
        # predictions differ across subwords of the same word; we keep the first.
        subword_preds = [9, 5, 7, 0, 9]
        assert gather_word_predictions(subword_preds, self.word_ids) == [5, 0]

    def test_out_of_range_word_id_raises(self):
        with pytest.raises(IndexError):
            align_labels_to_subwords([0], [None, 0, 1])  # label list too short


# ---------------------------------------------------------------------------
# Model tests — require transformers + the tiny test model
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bert_cls():
    pytest.importorskip("transformers")
    from app.ner.bert_ner import BertNER, BertNERConfig

    return BertNER, BertNERConfig


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")
    from app.ner.bert_ner import load_tokenizer

    try:
        return load_tokenizer(TINY_MODEL)
    except Exception as exc:  # offline / fetch failure
        pytest.skip(f"could not load {TINY_MODEL}: {exc}")


@pytest.fixture(scope="module")
def model(bert_cls):
    BertNER, BertNERConfig = bert_cls
    torch.manual_seed(0)
    try:
        return BertNER(BertNERConfig(num_tags=7, encoder_name=TINY_MODEL))
    except Exception as exc:
        pytest.skip(f"could not build BertNER from {TINY_MODEL}: {exc}")


def _encode(tokenizer, sentences):
    return tokenizer(
        sentences, is_split_into_words=True, return_tensors="pt", padding=True
    )


class TestBertNER:
    def test_emission_shape(self, model, tokenizer):
        enc = _encode(tokenizer, [["OpenAI", "rocks"], ["hello", "world", "again"]])
        out = model(enc["input_ids"], attention_mask=enc["attention_mask"])
        assert out.shape[0] == 2
        assert out.shape[1] == enc["input_ids"].shape[1]
        assert out.shape[2] == 7

    def test_predict_range(self, model, tokenizer):
        enc = _encode(tokenizer, [["OpenAI", "rocks"]])
        preds = model.predict(enc["input_ids"], attention_mask=enc["attention_mask"])
        assert preds.shape == enc["input_ids"].shape
        assert int(preds.max()) < 7 and int(preds.min()) >= 0

    def test_gradients_flow_when_unfrozen(self, model, tokenizer):
        enc = _encode(tokenizer, [["OpenAI", "rocks"]])
        out = model(enc["input_ids"], attention_mask=enc["attention_mask"])
        labels = torch.zeros(out.shape[:2], dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(
            out.reshape(-1, 7), labels.reshape(-1)
        )
        loss.backward()
        enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
        assert enc_grads and any(torch.any(g != 0) for g in enc_grads)
        assert model.classifier.weight.grad is not None

    def test_frozen_encoder_has_no_grad(self, bert_cls, tokenizer):
        BertNER, BertNERConfig = bert_cls
        torch.manual_seed(0)
        try:
            frozen = BertNER(
                BertNERConfig(num_tags=7, encoder_name=TINY_MODEL, freeze_encoder=True)
            )
        except Exception as exc:
            pytest.skip(f"could not build frozen BertNER: {exc}")

        enc = _encode(tokenizer, [["OpenAI", "rocks"]])
        out = frozen(enc["input_ids"], attention_mask=enc["attention_mask"])
        loss = out.sum()
        loss.backward()
        assert all(p.grad is None for p in frozen.encoder.parameters())
        assert frozen.classifier.weight.grad is not None
        # trainable-param count excludes the frozen encoder
        assert frozen.num_parameters(trainable_only=True) < frozen.num_parameters(
            trainable_only=False
        )

    def test_ignore_index_excluded_from_loss(self, model, tokenizer):
        # -100 positions must not contribute: masking all but one position must
        # give exactly the plain cross-entropy of that single real position.
        enc = _encode(tokenizer, [["OpenAI", "rocks"]])
        out = model(enc["input_ids"], attention_mask=enc["attention_mask"]).detach()
        n = out.shape[1]

        labels = torch.full((1, n), IGNORE_INDEX, dtype=torch.long)
        labels[0, 1] = 3  # the only real position

        ce = torch.nn.functional.cross_entropy
        masked = ce(out.reshape(-1, 7), labels.reshape(-1), ignore_index=IGNORE_INDEX)
        only_real = ce(out[0, 1:2], torch.tensor([3]))
        assert torch.allclose(masked, only_real, atol=1e-6)

    def test_checkpoint_roundtrip(self, model, bert_cls, tokenizer, tmp_path):
        BertNER, _ = bert_cls
        enc = _encode(tokenizer, [["OpenAI", "rocks"]])
        model.eval()
        before = model(enc["input_ids"], attention_mask=enc["attention_mask"])

        p = tmp_path / "bert_ner.pt"
        model.save_checkpoint(p, extra={"epoch": 1})
        loaded, extra = BertNER.load_checkpoint(p)
        loaded.eval()
        after = loaded(enc["input_ids"], attention_mask=enc["attention_mask"])

        assert extra == {"epoch": 1}
        assert torch.allclose(before, after, atol=1e-5)
        assert loaded.config.to_dict() == model.config.to_dict()

"""Unit tests for Phase 7B — CRF layer and BiLSTM-CRF model.

Run with::

    cd backend && python -m pytest tests/test_crf.py -v
"""

from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")

from app.ner.crf import CRF
from app.ner.bilstm_crf import BiLSTMCRF, build_bilstm_crf_from_vocabs
from app.ner.model import NERModelConfig
from app.datasets.vocabulary import VocabularyBuilder, build_tag_vocabulary
from app.evaluation.metrics import precision_recall_f1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _emissions(b=3, t=5, c=4, seed=0):
    torch.manual_seed(seed)
    return torch.randn(b, t, c)


def _mask(lengths, t):
    return torch.arange(t)[None, :] < torch.tensor(lengths)[:, None]


def _path_score(crf: CRF, emissions_tc: torch.Tensor, tags: list[int]) -> torch.Tensor:
    """Score of one full (unmasked) path — independent re-implementation used
    to check the CRF's own scoring/decoding against first principles."""
    s = crf.start_transitions[tags[0]] + emissions_tc[0, tags[0]]
    for i in range(1, len(tags)):
        s = s + crf.transitions[tags[i - 1], tags[i]] + emissions_tc[i, tags[i]]
    s = s + crf.end_transitions[tags[-1]]
    return s


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_param_shapes(self):
        crf = CRF(5)
        assert crf.start_transitions.shape == (5,)
        assert crf.end_transitions.shape == (5,)
        assert crf.transitions.shape == (5, 5)

    def test_rejects_nonpositive_tags(self):
        with pytest.raises(ValueError):
            CRF(0)

    def test_rejects_wrong_emission_dim(self):
        crf = CRF(4)
        with pytest.raises(ValueError):
            crf.decode(torch.randn(2, 3, 5))  # last dim != num_tags

    def test_rejects_masked_first_step(self):
        crf = CRF(4)
        em = _emissions(2, 3, 4)
        mask = torch.tensor([[True, True, True], [False, True, True]])
        with pytest.raises(ValueError):
            crf.decode(em, mask)


# ---------------------------------------------------------------------------
# Loss: the partition function dominates any single path
# ---------------------------------------------------------------------------
class TestLoss:
    def test_nll_nonnegative(self):
        # partition = logΣexp over ALL paths ≥ gold path score  ⇒  nll ≥ 0.
        crf = CRF(4)
        em = _emissions(3, 5, 4)
        tags = torch.randint(0, 4, (3, 5))
        nll = crf(em, tags, reduction="none")
        assert nll.shape == (3,)
        assert torch.all(nll >= -1e-5)

    def test_reductions_agree(self):
        crf = CRF(4)
        em = _emissions(3, 5, 4)
        tags = torch.randint(0, 4, (3, 5))
        none = crf(em, tags, reduction="none")
        assert torch.allclose(crf(em, tags, reduction="sum"), none.sum())
        assert torch.allclose(crf(em, tags, reduction="mean"), none.mean())

    def test_mean_loss_is_scalar(self):
        crf = CRF(4)
        loss = crf(_emissions(2, 4, 4), torch.randint(0, 4, (2, 4)))
        assert loss.dim() == 0

    def test_padding_does_not_change_loss(self):
        # A masked-out tail must contribute nothing: the per-sequence NLL of a
        # length-3 sequence is identical whether or not we pad it to length 6.
        crf = CRF(4)
        torch.manual_seed(2)
        core = torch.randn(1, 3, 4)
        core_tags = torch.tensor([[1, 2, 0]])
        short = crf(core, core_tags, mask=torch.ones(1, 3, dtype=torch.bool),
                    reduction="none")

        pad_em = torch.cat([core, torch.randn(1, 3, 4)], dim=1)   # garbage tail
        pad_tags = torch.cat([core_tags, torch.randint(0, 4, (1, 3))], dim=1)
        pad_mask = torch.tensor([[True, True, True, False, False, False]])
        padded = crf(pad_em, pad_tags, mask=pad_mask, reduction="none")
        assert torch.allclose(short, padded, atol=1e-5)


# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------
class TestViterbi:
    def test_decode_lengths_match_mask(self):
        crf = CRF(4)
        em = _emissions(3, 5, 4)
        mask = _mask([5, 4, 2], 5)
        paths = crf.decode(em, mask)
        assert [len(p) for p in paths] == [5, 4, 2]
        assert all(0 <= t < 4 for p in paths for t in p)

    def test_viterbi_finds_global_max(self):
        # Brute-force every possible path on a tiny problem and confirm Viterbi
        # returns one whose score equals the true maximum.
        torch.manual_seed(3)
        crf = CRF(3)
        T, C = 4, 3
        em = torch.randn(1, T, C)
        best = crf.decode(em)[0]
        best_score = _path_score(crf, em[0], best)

        brute_max = max(
            _path_score(crf, em[0], list(p))
            for p in itertools.product(range(C), repeat=T)
        )
        assert torch.allclose(best_score, brute_max, atol=1e-5)

    def test_decoded_score_matches_partition_bound(self):
        # The best single path can never out-score the log-partition.
        crf = CRF(4)
        em = _emissions(1, 6, 4, seed=5)
        path = crf.decode(em)[0]
        with torch.no_grad():
            score = _path_score(crf, em[0], path)
            partition = crf._compute_normalizer(
                em.transpose(0, 1), torch.ones(6, 1, dtype=torch.bool)
            )
        assert float(score) <= float(partition) + 1e-4


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------
class TestGradients:
    def test_grads_flow_to_transitions(self):
        crf = CRF(4)
        em = _emissions(2, 4, 4)
        em.requires_grad_(True)
        loss = crf(em, torch.randint(0, 4, (2, 4)))
        loss.backward()
        assert crf.transitions.grad is not None
        assert torch.any(crf.transitions.grad != 0)
        assert em.grad is not None and torch.any(em.grad != 0)

    def test_grads_flow_through_bilstm(self):
        cfg = NERModelConfig(vocab_size=20, num_tags=5, embed_dim=8, hidden_dim=8)
        torch.manual_seed(0)
        model = BiLSTMCRF(cfg)
        input_ids = torch.randint(0, 20, (3, 6))
        tags = torch.randint(0, 5, (3, 6))
        mask = _mask([6, 5, 4], 6)
        loss = model(input_ids, tags, mask=mask)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and any(torch.any(g != 0) for g in grads)
        assert model.crf.transitions.grad is not None


# ---------------------------------------------------------------------------
# BiLSTMCRF wiring + checkpointing
# ---------------------------------------------------------------------------
class TestBiLSTMCRF:
    def test_emissions_shape(self):
        cfg = NERModelConfig(vocab_size=20, num_tags=5, embed_dim=8, hidden_dim=8)
        model = BiLSTMCRF(cfg)
        out = model.emissions(torch.randint(0, 20, (3, 6)))
        assert out.shape == (3, 6, 5)

    def test_decode_shapes(self):
        cfg = NERModelConfig(vocab_size=20, num_tags=5, embed_dim=8, hidden_dim=8)
        model = BiLSTMCRF(cfg)
        input_ids = torch.randint(0, 20, (3, 6))
        mask = _mask([6, 5, 4], 6)
        paths = model.decode(input_ids, mask=mask)
        assert [len(p) for p in paths] == [6, 5, 4]

    def test_checkpoint_roundtrip(self, tmp_path):
        cfg = NERModelConfig(vocab_size=20, num_tags=5, embed_dim=8, hidden_dim=8)
        torch.manual_seed(0)
        model = BiLSTMCRF(cfg)
        input_ids = torch.randint(0, 20, (2, 5))
        mask = _mask([5, 4], 5)
        before = model.decode(input_ids, mask=mask)

        p = tmp_path / "bilstm_crf.pt"
        model.save_checkpoint(p, extra={"epoch": 2})
        loaded, extra = BiLSTMCRF.load_checkpoint(p)
        after = loaded.decode(input_ids, mask=mask)

        assert extra == {"epoch": 2}
        assert before == after
        assert loaded.config.to_dict() == cfg.to_dict()


# ---------------------------------------------------------------------------
# End-to-end: overfit a toy corpus to F1 = 1.0 (parity with the 7A test)
# ---------------------------------------------------------------------------
class TestOverfit:
    def test_overfits_tiny_corpus(self):
        sentences = [
            ["John", "Smith", "works", "at", "OpenAI"],
            ["Mary", "lives", "in", "Paris"],
            ["Acme", "Corp", "hired", "Bob"],
        ]
        tag_seqs = [
            ["B-PERSON", "I-PERSON", "O", "O", "B-ORG"],
            ["B-PERSON", "O", "O", "B-LOCATION"],
            ["B-ORG", "I-ORG", "O", "B-PERSON"],
        ]

        word_vocab = VocabularyBuilder().fit(sentences).build()
        tag_vocab = build_tag_vocabulary()

        max_t = max(len(s) for s in sentences)
        input_ids = torch.zeros(len(sentences), max_t, dtype=torch.long)
        tags = torch.zeros(len(sentences), max_t, dtype=torch.long)
        lengths = [len(s) for s in sentences]
        for i, (sent, tg) in enumerate(zip(sentences, tag_seqs)):
            for j, (w, t) in enumerate(zip(sent, tg)):
                input_ids[i, j] = word_vocab.encode(w)
                tags[i, j] = tag_vocab.encode(t)
        mask = _mask(lengths, max_t)

        torch.manual_seed(0)
        model = build_bilstm_crf_from_vocabs(
            word_vocab, tag_vocab, embed_dim=32, hidden_dim=32, dropout=0.0
        )
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        model.train()
        for _ in range(200):
            opt.zero_grad()
            loss = model(input_ids, tags, mask=mask)
            loss.backward()
            opt.step()

        paths = model.decode(input_ids, mask=mask)
        gold = [tag_vocab.decode_sequence(tags[i, :n].tolist())
                for i, n in enumerate(lengths)]
        pred = [tag_vocab.decode_sequence(p) for p in paths]

        prf = precision_recall_f1(gold, pred)
        assert prf.f1 == pytest.approx(1.0)
        # and the decoded sequences are structurally valid BIO (no I-* without
        # a matching open span) — exactly what the CRF transitions enforce.
        for seq in pred:
            prev = "O"
            for tag in seq:
                if tag.startswith("I-"):
                    typ = tag[2:]
                    assert prev in (f"B-{typ}", f"I-{typ}")
                prev = tag

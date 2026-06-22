# Phase 5 — Vocabulary Builder

## 1. Theory

A neural model consumes **integer ids**, not strings. The vocabulary is the
bijection

```
token  ⇄  id          (word2idx / idx2word)
```

mapping each token to a row of the embedding matrix (Phase 7). Two special
tokens make this usable for batched training:

| Special | id | Purpose |
|---------|----|---------|
| `<PAD>` | 0 | Filler for padding short sequences to a common length (Phase 6). Masked → zero loss. |
| `<UNK>` | 1 | Bucket for any token unseen at training time. Without it, inference crashes on the first novel word. |

### Why a frequency cutoff?

Rare tokens (typos, one-off names) inflate the embedding table and never
receive a useful gradient. Mapping everything below `min_freq` to `<UNK>`:
1. shrinks the model, and
2. **trains** the UNK embedding on exactly the rare cases it will meet at test
   time. A cutoff is regularization, not just compression.

## 2. Two vocabularies

NER needs both, and `Vocabulary` serves both:

| Vocabulary | UNK? | Built by | Contents |
|------------|------|----------|----------|
| **word** | yes (open class) | `VocabularyBuilder` from corpus | `<PAD>`, `<UNK>`, corpus tokens |
| **tag** | no (closed set) | `build_tag_vocabulary()` | `<PAD>` + the 17 BIO tags |

The tag set is closed (Phase 2/4 define exactly 17 BIO tags), so it needs no
UNK; `<PAD>` lets the loss ignore padded label positions (Phase 6/8).

## 3. Determinism

`VocabularyBuilder.build` orders tokens reproducibly:

```
[<PAD>, <UNK>]  +  tokens sorted by (frequency desc, then alphabetical)
```

Identical corpus → identical id assignment on every machine and run. This
matters because a saved model's weights are tied to specific ids; a
non-deterministic vocab would silently corrupt a reloaded checkpoint.

## 4. API

```python
from app.datasets.vocabulary import VocabularyBuilder, Vocabulary, build_tag_vocabulary

# --- word vocabulary from a tokenized corpus ---
vb = VocabularyBuilder(lowercase=True)
vb.fit([["John", "works"], ["Jane", "works"]])      # many docs
vb.update(["one", "more", "doc"])                    # or one at a time
vocab = vb.build(min_freq=1, max_size=50_000)

vocab.encode("works")                 # -> int (UNK id if unseen)
vocab.decode(5)                       # -> token
vocab.encode_sequence(toks, max_len=128)   # ids, right-padded/truncated
vocab.decode_sequence(ids, strip_pad=True)
vocab.word2idx ; vocab.idx2word ; vocab.pad_id ; vocab.unk_id

# --- tag vocabulary (closed, no UNK) ---
tag_vocab = build_tag_vocabulary()    # <PAD>=0, O=1, B-PERSON=2, ...

# --- persistence (ids are stable across save/load) ---
vocab.save("models/word_vocab.json")
Vocabulary.load("models/word_vocab.json")
```

## 5. Worked example

```python
tok = Tokenizer(lowercase=True)
vb  = VocabularyBuilder(lowercase=True)
vb.fit(tok.tokens(a.text) for a in annotations)
word_vocab = vb.build()               # e.g. 31 entries on the sample dataset
tag_vocab  = build_tag_vocabulary()   # 18 entries (PAD + 17)

word_vocab.encode_sequence(["openai", "UNSEENWORD", "san"], max_len=5)
# -> [22, 1, 27, 0, 0]   (1 = UNK, trailing 0 = PAD)
```

## 6. Design notes

- **`Vocabulary` is immutable** after construction and validates its invariants
  (no duplicates, PAD/UNK present). Build via `VocabularyBuilder`,
  `build_tag_vocabulary`, or `Vocabulary.from_tokens` — never hand-mutate.
- **`lowercase`** is a property of the vocabulary, applied consistently in
  `encode`/`__contains__`, so the same casing policy the tokenizer used is
  enforced at lookup time.
- **Serialization stores `idx2token` as an ordered list** plus the special-token
  config, so reloaded ids match the originals exactly — essential for
  checkpoint compatibility.

## 7. Files

| Path | Purpose |
|------|---------|
| `backend/app/datasets/vocabulary.py` | `Vocabulary`, `VocabularyBuilder`, `build_tag_vocabulary` |
| `backend/tests/test_vocabulary.py` | 25 tests (specials, cutoff, determinism, serialization) |

## 8. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_vocabulary.py -v
```

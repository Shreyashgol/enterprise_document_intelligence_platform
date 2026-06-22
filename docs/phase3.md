# Phase 3 — Tokenizer (from scratch)

## 1. Theory

A tokenizer maps raw text → an ordered list of **tokens**, the atomic units the
model treats as indivisible. It sits between *characters* (storage) and *tokens*
(what the model embeds). Everything downstream depends on it:

- **Vocabulary** (Phase 5) is built over tokens.
- **NER model** (Phase 7) embeds token ids and predicts one tag per token.
- **BIO tagging** (Phase 4) assigns exactly one tag to each token.

## 2. The offset invariant (why this design matters)

Our ground-truth labels (Phase 2) are **character spans**. To convert them into
per-token BIO tags (Phase 4) we must know **where each token sits in the
source**. So every `Token` carries `(start, end)` and obeys:

```
text[token.start : token.end] == token.text     # (before lowercasing)
```

`str.split()` returns bare strings and destroys this information — it cannot be
aligned to span labels. That is the whole reason we evolve from V1 to V2.

## 3. Two versions

### V1 — `WhitespaceTokenizer`
The naive baseline. `tokens()` is literally `str.split()`; `tokenize()` recovers
offsets via `\S+`. Its weaknesses motivate V2:

```
"Hello, world!"  ->  ["Hello,", "world!"]      # punctuation glued
"call 123-4567."  ->  ["call", "123-4567."]    # trailing dot glued
```

### V2 — `Tokenizer`
A single-pass, **priority-ordered regex scanner** (`re.finditer`, so offsets are
free and the invariant holds by construction). Match priority:

| Priority | Kind | Behavior | Example |
|---------:|------|----------|---------|
| 1 | `EMAIL` | atomic | `john.doe+x@mail.co.uk` |
| 2 | `PHONE` | atomic (incl. spaces) | `+1 (555) 123-4567` |
| 3 | `ABBR` | atomic, keeps period | `Dr.` `U.S.A.` `e.g.` `Inc.` |
| 4 | `NUMBER` | atomic | `1,000.00` `3.14` |
| 5 | `WORD` | letters/digits, internal `'`/`-` | `don't` `state-of-the-art` `B2B` |
| 6 | `PUNCT` | one symbol | `,` `!` `$` `€` |

> `Token.kind` is a **coarse lexical hint** (how it matched), *not* a semantic
> entity label. Deciding "2024-01-15" is a DATE is the NER layer's job; the
> tokenizer only guarantees correct, gap-free segmentation with valid offsets.

## 4. V1 vs V2 on a real sentence

Input:
```
Dr. Lee at OpenAI Inc. emailed john@acme.com on Jan. 5, 2024; call +1 (555) 123-4567.
```

**V1** glues: `Inc.` `john@acme.com`(lucky) `5,` `2024;` `123-4567.`

**V2**:
```
Dr.            ABBR   [0:3]
Lee            WORD   [4:7]
...
Inc.           ABBR   [18:22]
john@acme.com  EMAIL  [31:44]
Jan.           ABBR   [48:52]
5              NUMBER [53:54]
,              PUNCT  [54:55]
2024           NUMBER [56:60]
;              PUNCT  [60:61]
+1 (555) 123-4567  PHONE [67:84]
.              PUNCT  [84:85]
```

## 5. API

```python
from app.tokenizer.tokenizer import Tokenizer, WhitespaceTokenizer

tok = Tokenizer(lowercase=False)        # configurable: lowercase, abbreviations
tok.tokenize(text)  # -> list[Token(text, start, end, kind)]
tok.tokens(text)    # -> list[str]  (convenience)

WhitespaceTokenizer().tokens(text)      # == text.split()
```

`Tokenizer(lowercase=True)` lowercases `Token.text` but leaves offsets pointing
at the original casing, so `text[start:end]` still returns the source slice.

## 6. Design decisions & limitations

- **Phones may include spaces** (`+44 20 7946 0958`), so the phone branch is a
  structured grouping pattern, not "digits + separators" — this prevents it from
  greedily swallowing adjacent prose.
- **Numeric-separator runs** like dates `2024-01-15` are kept as one token (kind
  may show `PHONE`); this is fine — tokenization keeps them atomic and the NER
  layer assigns the real label.
- **Abbreviation list is configurable** (`DEFAULT_ABBREVIATIONS`). A true
  sentence-final `Inc.` will absorb the terminal period — an accepted ambiguity
  shared by most rule tokenizers.
- This is a **deterministic, rule-based** tokenizer (appropriate for a
  word-level BiLSTM in Phase 7). Subword tokenization (BPE/WordPiece) is a
  separate concern introduced only if/when we move to Transformer models.

## 7. Files

| Path | Purpose |
|------|---------|
| `backend/app/tokenizer/tokenizer.py` | `Token`, `WhitespaceTokenizer`, `Tokenizer` |
| `backend/tests/test_tokenizer.py` | 33 tests (incl. the offset invariant) |

## 8. Running

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/test_tokenizer.py -v
```

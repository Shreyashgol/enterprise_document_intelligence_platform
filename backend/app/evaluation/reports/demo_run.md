# NER Evaluation Report

- generated: `2026-06-22T15:05:28.267382+00:00`
- sequences evaluated: **24**

## Overall (micro, entity-level)

| Precision | Recall | F1 | TP | FP | FN | Support |
|----------:|-------:|---:|---:|---:|---:|--------:|
| 1.000 | 1.000 | 1.000 | 48 | 0 | 0 | 48 |

## Per-label

| Label | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| ORG | 1.000 | 1.000 | 1.000 | 24 |
| PERSON | 1.000 | 1.000 | 1.000 | 24 |

## Confusion matrix (token-level, rows=gold, cols=pred)

| gold\pred | O | PERSON | ORG | EMAIL | PHONE | DATE | MONEY | LOCATION | PRODUCT |
|---|---|---|---|---|---|---|---|---|---|
| **O** | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **PERSON** | 0 | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **ORG** | 0 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | 0 |
| **EMAIL** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **PHONE** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **DATE** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **MONEY** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **LOCATION** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **PRODUCT** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

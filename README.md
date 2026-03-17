# BPE Tokenizer for Spanish

### Part I — Build a Transformer from Scratch

A Byte-Pair Encoding tokenizer implemented from scratch in pure Python, trained on the Spanish Wikipedia corpus. This is the first component of a larger project to build a Spanish language transformer model from the ground up, without relying on any ML frameworks.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Pipeline Overview](#pipeline-overview)
4. [Implementation](#implementation)
   - [Corpus & Preprocessing](#1-corpus--preprocessing)
   - [BPE Training](#2-bpe-training)
   - [Training Optimization](#3-training-optimization-incremental-pair-index)
   - [Persistence](#4-persistence)
   - [Encoding](#5-encoding)
   - [Decoding](#6-decoding)
5. [Properties & Guarantees](#properties--guarantees)
6. [Quick Start](#quick-start)
7. [Dependencies](#dependencies)

---

## Introduction

### What is BPE?

Byte-Pair Encoding (BPE) is a data-compression algorithm adapted for subword tokenization. It starts with a vocabulary of individual characters and iteratively merges the most frequent adjacent symbol pair in the corpus, building progressively larger tokens. This continues until the vocabulary reaches a target size.

The result is a tokenizer that handles the full spectrum between characters and words: common words become single tokens, rare or unknown words are decomposed into recognizable subword pieces rather than collapsing to `<UNK>`.

### Why BPE for Spanish?

Spanish is a morphologically rich language — the same root can produce dozens of surface forms through inflection and derivation (*corr-er*, *corr-iendo*, *corr-ió*). BPE handles this naturally: frequent morphemes get merged into tokens while the algorithm still gracefully handles rare combinations through subword decomposition. BPE was also chosen for being the algorithm behind the GPT family, making it the most relevant baseline for transformer-based generation. No external tokenization library is used — the entire algorithm is implemented in plain Python.

### Role in the Transformer

The tokenizer is the boundary between raw text and the numerical representations the model consumes. It is responsible for two things:

- **Encoding**: converting a string into a sequence of integer IDs that the embedding layer can look up.
- **Decoding**: converting the model's output IDs back into human-readable text.

Everything downstream — the embedding table, the positional encodings, the attention layers — depends on the vocabulary this tokenizer defines. Getting it right before building the rest of the model is not optional.

---

## Project Structure

```
tokenizerproject/
├── download_corpus.py   # Downloads the Spanish Wikipedia corpus from HuggingFace
├── preprocess.py        # Builds and serializes the word frequency dictionary
├── tokenizer.py         # BPETokenizer class: train, encode, decode, save, load
├── main.py              # Integration tests: roundtrip, segmentation, edge cases
├── data/
│   ├── corpus_es.txt    # Raw corpus (~5.5 GB, Spanish Wikipedia)
│   └── word_freqs.json  # Preprocessed word frequency dictionary
└── vocab/
    └── tokenizer.json   # Trained tokenizer (vocab + merge rules)
```

---

## Pipeline Overview

```
Raw corpus (5.5 GB)
        │
        ▼
  download_corpus.py        ← HuggingFace wikimedia/wikipedia 20231101.es
        │
        ▼
   preprocess.py            ← clean → pretokenize → word_to_symbols → Counter
        │
        ▼
  word_freqs.json           ← 681k unique words (min_freq=10), serialized
        │
        ▼
  BPETokenizer.train()      ← incremental pair index, 57 min, 5930 merges
        │
        ▼
  tokenizer.json            ← 6000-token vocabulary + ordered merge rules
        │
     ┌──┴──┐
     ▼     ▼
  encode   decode
```

---

## Implementation

### 1. Corpus & Preprocessing

**Corpus**: The Spanish Wikipedia dump from HuggingFace (`wikimedia/wikipedia`, `20231101.es`) — approximately 5.5 GB of clean Spanish text, stripped of markup and designed for NLP tasks.

**Text normalization** (`clean_text`) applies two transformations in order:

- `.lower()` — collapses case variants so "Casa", "CASA", and "casa" map to the same token.
- `unicodedata.normalize('NFC', text)` — critical for Spanish. The character `é` can be encoded as a single Unicode code point (NFC) or as `e` followed by a combining accent (NFD). Both render identically on screen but are distinct byte sequences. Without normalization, BPE would treat them as different characters, silently corrupting the vocabulary.

**Pretokenization** (`pretokenize`) uses the regex `[a-záéíóúüñ]+` to extract only valid Spanish words. All punctuation, numbers, symbols, and residual corpus artifacts are discarded. This defines the word boundaries that BPE will never cross.

**Symbol representation** (`word_to_symbols`) converts each word into a tuple of characters with a `</w>` marker appended to the last character. The marker is what allows BPE to distinguish a subword prefix from a word-final token through all subsequent merges — `bajo` (the position "down") and `bajo</w>` (the word "bajo") are treated as distinct tokens.

**Scale**: 63.7 million lines processed in 24 minutes, producing a frequency dictionary of 3.8 million unique words. After filtering words with fewer than 10 occurrences, 681k unique words remain as input to training.

---

### 2. BPE Training

Training begins with a **base vocabulary** of the unique characters found in the corpus (70 symbols for Spanish, including accented vowels and ñ) plus 4 fixed special tokens. From there, each iteration:

1. Finds the most frequent adjacent symbol pair across the entire corpus (weighted by word frequency).
2. Merges that pair into a new token everywhere it appears.
3. Adds the new token to the vocabulary.
4. Records the merge rule in order.

This repeats until the vocabulary reaches the target size. The ordered list of merge rules is what makes encoding deterministic — the same merge priority learned during training is applied exactly during inference.

**Result**: 70 base symbols → **6,000 tokens** via 5,930 learned merge rules.

---

### 3. Training Optimization: Incremental Pair Index

The naive BPE implementation scans the entire corpus on every iteration to recount all pair frequencies — O(corpus × iterations). With a 681k-word corpus and 5,930 iterations, this is prohibitively slow.

The optimized implementation builds the pair index **once** upfront and maintains it **incrementally**:

- `pair_counts` — a `Counter` mapping each symbol pair to its total frequency across the corpus.
- `pair_locations` — a `defaultdict(set)` mapping each pair to the exact set of words where it appears.

When a pair `(a, b)` is merged into `ab`, only the words that actually contained that pair need to be updated. For each affected word:

- **Pairs that vanish**: `(a, b)` itself, plus the pairs it formed with its immediate neighbors, are decremented.
- **Pairs that appear**: the new token `ab` forms new pairs with its neighbors, which are incremented.

Words not containing `(a, b)` are copied to the new vocabulary untouched. The best next candidate is read directly from `pair_counts.most_common(1)` in O(1) instead of recomputing it.

**Complexity**: O(corpus) to build the index once, plus O(affected_words × iterations) to maintain it — versus O(corpus × iterations) for the naive approach. In practice, only a small fraction of the corpus is affected per merge.

**Result**: full training on 681k words completes in **57 minutes**.

---

### 4. Persistence

The trained tokenizer is serialized to a single JSON file containing:

- `vocab` — the complete `{token_string: integer_id}` mapping.
- `merges` — the ordered list of merge rules as `[token_a, token_b]` pairs.
- `special_tokens` — the special token registry.

On load, `BPETokenizer.load()` reconstructs both `vocab` and `inverse_vocab` from the same source and builds `merges_index` — a `{(a, b): rank}` dictionary for O(1) merge priority lookup during encoding.

Separating preprocessing (run once, ~24 min) from training (run as many times as needed, ~57 min) and persisting both intermediate results means no stage ever needs to be rerun unless its inputs change.

---

### 5. Encoding

`encode(text)` converts a string to a list of integer token IDs through three stages:

**Stage 1 — Preprocessing**: `clean_text` followed by `pretokenize`, in the same order and with the same logic used during training. This guarantees the encoder sees text in exactly the format the vocabulary was built from.

**Stage 2 — BPE segmentation** (`_tokenize_word`): each word starts as a list of individual characters with the `</w>` marker. The algorithm then applies merges greedily by priority:

1. Scan all adjacent pairs in the current symbol list.
2. Look up each pair in `merges_index` (O(1) per lookup).
3. Apply the pair with the lowest rank (learned earliest = most frequent in training).
4. Repeat until no mergeable pair remains.

This deterministically reproduces the segmentation learned during training.

**Stage 3 — ID lookup**: each resulting symbol is mapped to its integer ID via `self.vocab`. Unknown symbols fall back to `<UNK>` (ID 1) via `.get(symbol, self.vocab['<UNK>'])`, ensuring no input ever raises a `KeyError`.

---

### 6. Decoding

`decode(ids)` inverts the encoding:

1. **Filter special tokens** — `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>` are control signals for the transformer, not text content. Including them would corrupt the output string.
2. **Concatenate** — symbols are joined with `''.join()` without any separator, because word boundaries are fully encoded in the `</w>` markers, not in explicit spaces.
3. **Restore word boundaries** — every `</w>` is replaced by a space, reconstructing the original word separations. `.strip()` removes the trailing space left by the last word.

---

## Properties & Guarantees

| Property | Detail |
|---|---|
| **Roundtrip fidelity** | `decode(encode(s)) ≈ s` for any clean Spanish text. The approximation reflects that preprocessing is lossy in one direction: lowercasing is irreversible. At the token level, reconstruction is exact. |
| **Training complexity** | O(corpus) for index construction + O(affected_words × iterations) for incremental updates, versus O(corpus × iterations) for the naive approach. |
| **Encoding complexity** | O(n_words × n_symbols_per_word × log(merges)) per text — dominated by the per-word merge loop, not vocabulary size. |
| **Vocabulary size** | 6,000 tokens: 4 special tokens + 66 base characters + 5,930 learned merges. |
| **OOV handling** | Any character sequence not covered by the vocabulary degrades gracefully to `<UNK>`. The Spanish regex pretokenizer makes this case rare in practice. |
| **Special token stability** | `<PAD>=0`, `<UNK>=1`, `<BOS>=2`, `<EOS>=3` are hardcoded before any training. These IDs are stable across all retrainings, which is a requirement for the transformer's attention masking logic. |
| **Determinism** | Encoding is fully deterministic. Given the same `tokenizer.json`, the same input always produces the same IDs. |
| **No external ML dependencies** | The full tokenizer is pure Python + stdlib. The only external dependencies are `tqdm` (progress bars) and `datasets` / `huggingface_hub` (corpus download, not needed for inference). |

---

## Quick Start

```python
from tokenizer import BPETokenizer

# Load a trained tokenizer
tokenizer = BPETokenizer.load("vocab/tokenizer.json")

# Encode text
ids = tokenizer.encode("el modelo aprende representaciones del lenguaje")
print(ids)  # [42, 87, 312, ...]

# Decode back to text
text = tokenizer.decode(ids)
print(text)  # "el modelo aprende representaciones del lenguaje"

# Inspect word segmentation
tokens = tokenizer._tokenize_word("anticonstitucional")
print(tokens)  # ['anti', 'con', 'sti', 'tu', 'cional</w>']
```

**To train from scratch:**

```bash
# 1. Download corpus (~5.5 GB)
python download_corpus.py

# 2. Build word frequency dictionary (~24 min)
python preprocess.py

# 3. Train BPE tokenizer (~57 min)
python tokenizer.py

# 4. Run integration tests
python main.py
```

---

## Dependencies

```
tqdm
datasets
huggingface_hub
```

---

> This tokenizer is **Part I** of a series building a Spanish transformer model from scratch — no PyTorch, no Hugging Face Transformers, no shortcuts. The goal is to understand every component from the ground up.

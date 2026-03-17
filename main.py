import streamlit as st
from collections import Counter
from tokenizer import BPETokenizer


@st.cache_resource
def load_tokenizer():
    return BPETokenizer.load("vocab/tokenizer.json")


def tokenize_word_traced(tokenizer, word: str):
    """Runs _tokenize_word and records every merge that was applied."""
    symbols = list(tokenizer._word_to_symbols(word))
    applied = []

    while len(symbols) > 1:
        best_pair = None
        best_rank = float("inf")

        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            rank = tokenizer.merges_index.get(pair)
            if rank is not None and rank < best_rank:
                best_pair = pair
                best_rank = rank

        if best_pair is None:
            break

        merged = best_pair[0] + best_pair[1]
        applied.append((best_pair[0], best_pair[1], merged))

        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols

    return symbols, applied


def get_pair_frequencies(tokenizer, words: list[str]) -> list[tuple]:
    pair_counts: Counter = Counter()
    for word in words:
        symbols = list(tokenizer._word_to_symbols(word))
        for i in range(len(symbols) - 1):
            pair_counts[(symbols[i], symbols[i + 1])] += 1
    return pair_counts.most_common(10)


@st.cache_data
def get_top_vocab(_tokenizer, n: int = 30) -> list[tuple[str, int]]:
    # Los tokens aprendidos primero tienen rank más bajo → más frecuentes en el corpus.
    merge_rank = {a + b: rank for rank, (a, b) in enumerate(_tokenizer.merges)}
    non_special = [t for t in _tokenizer.vocab if not t.startswith("<")]
    non_special.sort(key=lambda t: merge_rank.get(t, float("inf")))
    return [(t, merge_rank[t]) for t in non_special[:n] if t in merge_rank]


# ── UI ──────────────────────────────────────────────────────────────────────

st.title("BPE Tokenizer Explorer")
st.caption(
    "Spanish BPE tokenizer — trained on Wikipedia ES · 6 000 tokens · 5 930 merge rules")

tokenizer = load_tokenizer()

text_input = st.text_area("Input text", value="bajo bajando bajó", height=80)

if st.button("Tokenize"):
    words = tokenizer._pretokenize(tokenizer._clean_text(text_input))

    if not words:
        st.warning("No valid Spanish words found in the input.")
    else:
        # ── 1. Final tokens ──────────────────────────────────────────────
        st.subheader("1. Tokens")
        all_tokens: list[str] = []
        word_trace: list[tuple[str, list[str], list[tuple]]] = []

        for word in words:
            tokens, merges_applied = tokenize_word_traced(tokenizer, word)
            all_tokens.extend(tokens)
            word_trace.append((word, tokens, merges_applied))

        st.code(str(all_tokens))

        # ── 2. Step-by-step merges ───────────────────────────────────────
        st.subheader("2. Step-by-step merges")
        any_merge = False
        for word, tokens, merges_applied in word_trace:
            if merges_applied:
                any_merge = True
                st.markdown(f"**{word}**")
                lines = "\n".join(
                    f"  {a} + {b} → {ab}" for a, b, ab in merges_applied)
                st.text(lines)

        if not any_merge:
            st.text("No merges applied — all words are already single characters.")

        # ── 3. Pair frequencies ──────────────────────────────────────────
        st.subheader("3. Pair frequencies (top 10, character level)")
        pair_freqs = get_pair_frequencies(tokenizer, words)
        if pair_freqs:
            lines = "\n".join(
                f"  {pair}: {count}" for pair, count in pair_freqs)
            st.text(lines)
        else:
            st.text("Not enough symbols to compute pairs.")

        # ── Toggle: top tokens ───────────────────────────────────────────
        with st.expander("Mostrar tokens más frecuentes"):
            top_vocab = get_top_vocab(tokenizer, 30)
            rows = "\n".join(f"{t:<20} rank {r}" for t, r in top_vocab)
            st.text_area("", value=rows, height=260,
                         label_visibility="collapsed")

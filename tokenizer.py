"""
Tokenizador BPE entrenado desde cero en Python puro para español.
Componente de preprocesamiento del mini transformer — cubre entrenamiento, encoding y decoding.
"""
import json
import re
import unicodedata
import os
from collections import Counter, defaultdict
from tqdm import tqdm


class BPETokenizer:
    """Tokenizador Byte-Pair Encoding para español, implementado desde cero.

    El estado completo vive en tres estructuras: ``vocab`` (token → ID),
    ``inverse_vocab`` (ID → token) y ``merges`` (reglas de fusión en orden de
    aprendizaje). Mantener ambos dicts de vocabulario duplica la memoria pero
    hace el decoding O(1) sin búsquedas inversas.
    """

    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
    }

    def __init__(self):
        """Inicializa el tokenizador con los tokens especiales en IDs fijos.

        Los IDs 0-3 se reservan antes de que el entrenamiento empiece.
        Fijarlos aquí garantiza que <PAD>=0 siempre, independientemente de
        reentrenamientos, lo cual simplifica el enmascaramiento de atención
        en el transformer.
        """
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = []
        self.special_tokens = self.SPECIAL_TOKENS.copy()

        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

    def _clean_text(self, text: str) -> str:
        """Normaliza el texto antes de tokenizar.

        La normalización NFC es crítica para español: 'é' puede estar codificada
        como un solo codepoint o como 'e' + acento combinado. Ambas formas son
        visualmente idénticas pero distintas en bytes — sin esto el vocabulario
        acumula duplicados silenciosos.
        """
        text = text.lower()
        text = unicodedata.normalize('NFC', text)
        return text

    def _pretokenize(self, text: str) -> list[str]:
        """Extrae las palabras válidas en español del texto.

        El regex descarta puntuación, números y símbolos sin lógica adicional.
        BPE nunca cruzará los límites definidos aquí.
        """
        pattern = r'[a-záéíóúüñ]+'
        return re.findall(pattern, text)

    def _word_to_symbols(self, word: str) -> tuple[str, ...]:
        """Convierte una palabra en su representación de símbolos para BPE.

        El marcador </w> pegado al último carácter preserva el límite de palabra
        a través de todas las fusiones. Sin él, BPE no podría distinguir si un
        token es un prefijo o el final de una palabra.
        """
        return tuple(list(word[:-1]) + [word[-1] + '</w>'])

    def _build_pair_index(self, vocab: Counter) -> tuple[Counter, defaultdict]:
        """Construye el índice inicial de pares para el entrenamiento optimizado.

        Mantiene dos estructuras en paralelo: ``pair_counts`` para identificar
        el mejor par a fusionar y ``pair_locations`` para saber exactamente qué
        palabras actualizar sin escanear el corpus completo. Este es el costo
        único que reemplaza el reconteo en cada iteración del loop de entrenamiento.
        """
        pair_counts = Counter()
        pair_locations = defaultdict(set)

        for symbols in vocab.keys():
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += vocab[symbols]
                pair_locations[pair].add(symbols)

        return pair_counts, pair_locations

    def _merge_pair_fast(
        self,
        pair: tuple[str, str],
        vocab: Counter,
        pair_counts: Counter,
        pair_locations: defaultdict,
    ) -> tuple[Counter, Counter, defaultdict]:
        """Aplica una fusión y actualiza el índice de pares incrementalmente.

        Solo procesa las palabras que contienen el par — el resto pasa directo
        al vocabulario nuevo sin costo. Por cada palabra afectada, decrementa
        los pares que desaparecen (el par fusionado más sus vecinos) e incrementa
        los pares nuevos que forma el token resultante. Evita reconstruir el índice
        completo en cada iteración.
        """
        a, b = pair
        merged = a + b
        new_vocab = {}

        affected_words = pair_locations[pair].copy()

        for symbols in affected_words:
            if symbols not in vocab:
                continue

            freq = vocab[symbols]
            new_symbols = []
            i = 0

            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    if i > 0:
                        old_pair = (symbols[i - 1], a)
                        pair_counts[old_pair] -= freq
                        pair_locations[old_pair].discard(symbols)

                    if i < len(symbols) - 2:
                        old_pair = (b, symbols[i + 2])
                        pair_counts[old_pair] -= freq
                        pair_locations[old_pair].discard(symbols)

                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1

            new_symbols = tuple(new_symbols)
            new_vocab[new_symbols] = freq

            for i, sym in enumerate(new_symbols):
                if sym == merged:
                    if i > 0:
                        new_pair = (new_symbols[i - 1], merged)
                        pair_counts[new_pair] += freq
                        pair_locations[new_pair].add(new_symbols)
                    if i < len(new_symbols) - 1:
                        new_pair = (merged, new_symbols[i + 1])
                        pair_counts[new_pair] += freq
                        pair_locations[new_pair].add(new_symbols)

        for symbols, freq in vocab.items():
            if symbols not in affected_words:
                new_vocab[symbols] = freq

        del pair_counts[pair]
        del pair_locations[pair]

        return new_vocab, pair_counts, pair_locations

    def train(self, word_freqs_path: str, vocab_size: int = 6000) -> None:
        """Entrena el tokenizador BPE hasta alcanzar el tamaño de vocabulario objetivo.

        Args:
            word_freqs_path: ruta al JSON generado por preprocess.py.
            vocab_size: tokens totales en el vocabulario final, incluyendo
                especiales y símbolos base.
        """
        print("Cargando frecuencias...")
        with open(word_freqs_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        vocab = Counter({tuple(k.split(' ')): v for k, v in raw.items()})

        print("Inicializando vocabulario base...")
        next_id = len(self.special_tokens)
        for symbols in vocab.keys():
            for symbol in symbols:
                if symbol not in self.vocab:
                    self.vocab[symbol] = next_id
                    self.inverse_vocab[next_id] = symbol
                    next_id += 1

        print(f"Vocabulario base: {len(self.vocab)} símbolos")
        print("Construyendo índice de pares...")
        pair_counts, pair_locations = self._build_pair_index(vocab)

        print(f"Entrenando hasta vocab_size={vocab_size}...")
        with tqdm(total=vocab_size - len(self.vocab), desc="Entrenando BPE") as pbar:
            while len(self.vocab) < vocab_size:
                if not pair_counts:
                    break

                best_pair = pair_counts.most_common(1)[0][0]
                vocab, pair_counts, pair_locations = self._merge_pair_fast(
                    best_pair, vocab, pair_counts, pair_locations
                )

                new_token = best_pair[0] + best_pair[1]
                self.vocab[new_token] = next_id
                self.inverse_vocab[next_id] = new_token
                self.merges.append(best_pair)
                next_id += 1
                pbar.update(1)

        print(f"Entrenamiento completo. Vocab final: {len(self.vocab)} tokens")

    def save(self, path: str) -> None:
        """Serializa el tokenizador entrenado a disco en formato JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": self.special_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizador guardado en {path}")

    def _tokenize_word(self, word: str) -> list[str]:
        """Segmenta una palabra en tokens BPE aplicando las fusiones aprendidas.

        Las fusiones se aplican en orden de prioridad: rank más bajo significa
        aprendida antes, es decir, más frecuente en el corpus. ``merges_index``
        es un dict para que cada consulta sea O(1) — iterar la lista ordenada
        de fusiones sería prohibitivo multiplicado por todas las palabras de un texto.
        """
        symbols = list(self._word_to_symbols(word))

        while len(symbols) > 1:
            best_pair = None
            best_rank = float('inf')

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merges_index.get(pair, None)
                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            merged = best_pair[0] + best_pair[1]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str) -> list[int]:
        """Convierte texto a una secuencia de IDs de tokens.

        Aplica el mismo pipeline de preprocesamiento del entrenamiento, en el
        mismo orden. Invertirlos produce tokens inconsistentes con el vocabulario.
        """
        tokens = []

        words = self._pretokenize(self._clean_text(text))

        for word in words:
            word_tokens = self._tokenize_word(word)
            for symbol in word_tokens:
                token_id = self.vocab.get(symbol, self.vocab['<UNK>'])
                tokens.append(token_id)

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Convierte una secuencia de IDs de vuelta a texto legible.

        Los tokens especiales se filtran porque son señales de control para el
        transformer, no contenido — incluirlos corrompería el string resultante.
        Los símbolos se concatenan sin separador porque el espaciado está
        codificado en los marcadores </w>, no en espacios explícitos.
        """
        special_tokens = {'<PAD>', '<BOS>', '<EOS>', '<UNK>'}

        tokens = []
        for id in ids:
            symbol = self.inverse_vocab.get(id, '<UNK>')
            if symbol not in special_tokens:
                tokens.append(symbol)

        text = ''.join(tokens)
        text = text.replace('</w>', ' ')

        return text.strip()

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Carga un tokenizador previamente guardado con ``save``.

        ``merges_index`` se reconstruye en memoria en vez de persistirlo porque
        el rank de cada par está implícito en el orden de ``merges`` — guardarlo
        sería duplicar información.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.vocab = {k: int(v) for k, v in data["vocab"].items()}
        tokenizer.inverse_vocab = {int(v): k for k, v in data["vocab"].items()}
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer.special_tokens = data["special_tokens"]

        tokenizer.merges_index = {pair: rank for rank,
                                  pair in enumerate(tokenizer.merges)}

        return tokenizer


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.train("data/word_freqs.json", vocab_size=6000)
    tokenizer.save("vocab/tokenizer.json")

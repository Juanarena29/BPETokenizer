"""
Pipeline de preprocesamiento del corpus para el tokenizador BPE.
Lee el corpus crudo, extrae y filtra frecuencias de palabras, y las persiste en JSON,
desacoplando esta etapa del entrenamiento BPE.
"""
import re
import os
import unicodedata
import json
from collections import Counter
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Normaliza el texto antes de la tokenización.

    La normalización NFC es crítica para español: sin ella, 'é' y 'e' + acento
    combinado se tratan como caracteres distintos y el vocabulario acumula
    duplicados silenciosos.
    """
    text = text.lower()
    text = unicodedata.normalize('NFC', text)
    return text


def pretokenize(text: str) -> list[str]:
    """Extrae las palabras válidas en español del texto.

    El regex descarta puntuación, números y símbolos sin lógica adicional.
    BPE nunca cruzará los límites definidos aquí.
    """
    pattern = r'[a-záéíóúüñ]+'
    words = re.findall(pattern, text)
    return words


def word_to_symbols(word: str) -> tuple[str, ...]:
    """Convierte una palabra en su representación de símbolos para BPE.

    El marcador </w> pegado al último carácter preserva el límite de palabra
    a través de todas las fusiones. Sin él, BPE no podría distinguir si un
    token es un prefijo o el final de una palabra.
    """
    return tuple(list(word[:-1]) + [word[-1] + '</w>'])


def build_vocab(
    corpus_path: str,
    max_lines: int | None = None,
    min_freq: int = 3,
) -> Counter:
    """Construye el diccionario de frecuencias de palabras desde el corpus.

    Procesa el archivo línea a línea para mantener el uso de memoria constante —
    con un corpus de varios GB, leerlo completo en memoria no es viable.
    ``max_lines`` existe para iterar rápido durante el desarrollo sin esperar
    al corpus completo. ``min_freq`` filtra el ruido: palabras raras que el
    tokenizador nunca podrá generalizar bien.
    """
    vocab = Counter()

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Construyendo vocabulario")):
            if max_lines and i >= max_lines:
                break

            line = clean_text(line)
            words = pretokenize(line)

            for word in words:
                symbols = word_to_symbols(word)
                vocab[symbols] += 1

    vocab = Counter({k: v for k, v in vocab.items() if v >= min_freq})
    return vocab


def save_vocab(vocab: Counter, output_path: str) -> None:
    """Persiste el vocabulario de frecuencias en un archivo JSON.

    JSON no soporta tuplas como claves, así que cada tupla de símbolos se
    serializa como string con espacios: ('h','o','l','a</w>') → 'h o l a</w>'.
    ``ensure_ascii=False`` es necesario para que los acentos y la ñ no se
    escapen como secuencias unicode ilegibles.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vocab_serializable = {
        ' '.join(symbols): freq for symbols, freq in vocab.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)

    print(f"Vocabulario guardado en {output_path} ({len(vocab)} entradas)")


def load_vocab(input_path: str) -> Counter:
    """Carga un vocabulario previamente guardado con ``save_vocab``.

    Invierte la serialización: los strings 'h o l a</w>' vuelven a ser
    tuplas ('h','o','l','a</w>') para mantener compatibilidad con el
    pipeline de entrenamiento.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        vocab_serializable = json.load(f)

    return Counter({tuple(k.split(' ')): v for k, v in vocab_serializable.items()})


if __name__ == "__main__":
    print("Procesando corpus completo...")
    vocab = build_vocab("data/corpus_es.txt", min_freq=10)
    save_vocab(vocab, "data/word_freqs.json")
    print(f"Palabras únicas totales: {len(vocab)}")

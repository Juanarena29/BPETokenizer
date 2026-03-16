Borrador detallado — BPETokenizer en español

Estructura general del proyecto
Antes de hablar de pasos, el código va a vivir en dos archivos:
preprocess.py — todo lo que transforma texto crudo en algo que BPE puede consumir. Se ejecuta una sola vez sobre el corpus.
tokenizer.py — la clase BPETokenizer con entrenamiento, encoding, decoding y serialización.

preprocess.py — 3 pasos

Paso 1 — Leer y limpiar el corpus
El corpus de Wikipedia en español viene en un formato que tiene ruido: tags XML, títulos de secciones, referencias, tablas. Necesitás texto puro.
Lo que se hace: abrís el archivo línea por línea (no todo junto en memoria porque puede ser grande), y por cada línea aplicás limpieza. Usar open() con encoding utf-8 es obligatorio para español.
La limpieza incluye bajar todo a minúsculas con .lower(), y normalizar Unicode con unicodedata.normalize('NFC', linea) para que todos los acentos estén en forma canónica. Sin esto, la misma palabra puede aparecer como dos tokens distintos dependiendo de cómo fue codificada.
Por qué NFC específicamente: es la forma que compone los caracteres combinados en un solo carácter. é como un carácter único en lugar de e + acento separado.

Paso 2 — Pretokenización
Antes de dividir en caracteres, dividís el texto en palabras. Esto es la pretokenización y define los límites dentro de los cuales BPE puede fusionar — nunca va a fusionar tokens que crucen el límite entre dos palabras.
Lo que se hace: usás re.findall() con un patrón que capture palabras del español. El patrón tiene que incluir letras con acentos y la ñ, algo como [a-záéíóúüñ]+. Cada palabra encontrada recibe el marcador </w> al final de su último carácter para indicar fin de palabra.
Por qué findall y no split: split en espacios pierde información sobre puntuación y caracteres especiales de formas difíciles de controlar. findall con un patrón explícito es más predecible.
El resultado de este paso es una lista de palabras donde cada palabra es una tupla de caracteres: ('h','o','l','a','</w>').

Paso 3 — Construir el diccionario de frecuencias
BPE no trabaja sobre el corpus como secuencia de palabras — trabaja sobre un diccionario donde cada entrada es una palabra (como tupla de símbolos) y su valor es cuántas veces aparece en el corpus.
Lo que se hace: usás collections.Counter para contar cuántas veces aparece cada tupla de caracteres. El resultado es algo como {('h','o','l','a','</w>'): 3847, ('c','a','s','a','</w>'): 2910, ...}.
Por qué Counter y no un dict manual: tiene el método .update() que suma frecuencias en lugar de sobreescribir, y .most_common() que vas a usar después. Hace exactamente lo que necesitás sin código extra.
Este diccionario de frecuencias es el único input que necesita el entrenamiento. Guardalo con json para no tener que reprocesar el corpus cada vez que experimentes con distintos vocab_size.

tokenizer.py — la clase BPETokenizer
La clase tiene cuatro responsabilidades separadas: entrenamiento, encoding, decoding y serialización. Las implementás como métodos distintos.

Paso 4 — __init__ — inicializar el estado
El estado del tokenizador son exactamente tres cosas:
self.vocab — un diccionario {token_string: id_entero}. Empieza con los caracteres únicos del corpus más los tokens especiales, y crece con cada fusión durante el entrenamiento.
self.merges — una lista de tuplas [(símbolo_a, símbolo_b), ...] en orden de aprendizaje. Este orden es sagrado — definís la lista una vez y nunca la reordenás.
self.special_tokens — un diccionario con los tokens especiales y sus IDs reservados. Los IDs especiales van primero, antes de cualquier token del vocabulario, para que nunca colisionen. Los tokens que necesitás son <PAD> con ID 0, <UNK> con ID 1, <BOS> con ID 2 y <EOS> con ID 3.
Por qué los especiales van primero con IDs fijos: cuando después construyas el transformer, el embedding layer necesita saber de antemano qué ID es padding para ignorarlo en la atención. Tener <PAD>=0 siempre es una convención que simplifica todo lo que viene después.

Paso 5 — _get_pairs — contar pares (método auxiliar)
Este es el método que más se llama durante el entrenamiento — una vez por iteración. Tiene que ser lo más eficiente posible.
Lo que recibe: el diccionario de frecuencias actual, donde las claves son tuplas de símbolos y los valores son frecuencias.
Lo que hace: recorre todas las palabras del diccionario, y por cada palabra recorre todos los pares de símbolos adyacentes. Cada par se suma al Counter pesado por la frecuencia de la palabra. Es decir, si ('casa</w>') aparece 1000 veces, el par ('c','a') recibe 1000 votos de esa sola palabra.
Lo que devuelve: un Counter donde las claves son tuplas (símbolo_a, símbolo_b) y los valores son frecuencias totales en el corpus.
Por qué es un método privado con underscore: es un detalle de implementación que solo usa train internamente. No forma parte de la interfaz pública de la clase.

Paso 6 — _merge_pair — aplicar una fusión (método auxiliar)
Dado el par ganador de una iteración, actualizás el diccionario de frecuencias reemplazando todas las ocurrencias de ese par por el símbolo fusionado.
Lo que recibe: el diccionario de frecuencias actual, y el par ganador como tupla (símbolo_a, símbolo_b).
Lo que hace: construye un nuevo diccionario. Por cada palabra, si contiene el par adyacente (a, b) en alguna posición, lo reemplaza por ab (la concatenación). La frecuencia de la palabra no cambia — solo cambia su representación.
Lo que devuelve: el diccionario actualizado con las fusiones aplicadas.
Por qué construís un diccionario nuevo en lugar de modificar el existente: modificar una estructura mientras la iterás es una fuente clásica de bugs en Python. Construir uno nuevo es más seguro y no es significativamente más lento.

Paso 7 — train — el loop principal
Acá se conectan todos los piezas anteriores.
Lo que recibe: el diccionario de frecuencias del corpus (output del preprocesamiento), y vocab_size como entero — el tamaño objetivo del vocabulario.
Lo que hace, paso a paso:
Primero inicializa el vocabulario con todos los caracteres únicos que aparecen en el corpus. Recorrés todas las palabras del diccionario de frecuencias, descomponés cada tupla en sus símbolos individuales, y los agregás al vocab. Le asignás IDs empezando desde 4 (los primeros 4 están reservados para los especiales).
Después arranca el loop. La condición de parada es len(self.vocab) < vocab_size. En cada iteración: llamás a _get_pairs para obtener el Counter de pares, usás .most_common(1) para obtener el par más frecuente, llamás a _merge_pair para actualizar el corpus, agregás el nuevo símbolo fusionado al vocab con el siguiente ID disponible, y agregás el par a self.merges. Todo envuelto en tqdm para ver el progreso.
Por qué la condición es sobre len(vocab) y no sobre número de iteraciones: porque cada iteración agrega exactamente un token al vocabulario, así que son equivalentes. Pero pensar en términos de vocab_size es más intuitivo — es el parámetro que vos controlás.

Paso 8 — encode — texto a IDs
Lo que recibe: un string de texto, y opcionalmente flags booleanos add_bos=True y add_eos=True para agregar los tokens especiales de inicio y fin.
Lo que hace:
Primero aplica el mismo preprocesamiento que en el entrenamiento — minúsculas, normalización NFC, pretokenización con la misma regex. Es crítico que sea exactamente el mismo proceso, si no el encoding es inconsistente.
Después, por cada palabra, arranca con la representación en caracteres individuales y aplica las fusiones de self.merges en orden. Recorrés self.merges de arriba a abajo, y por cada merge verificás si ese par aparece en la palabra actual. Si aparece, lo fusionás. Seguís hasta que no quede ningún par fusionable.
Finalmente convertís cada símbolo a su ID usando self.vocab. Si un símbolo no está en el vocab (cosa rara con BPE pero posible con caracteres muy exóticos), usás el ID de <UNK>. Si add_bos es True, prependés el ID de <BOS>. Si add_eos es True, appendeás el ID de <EOS>.
Por qué aplicás los merges en orden y no buscás el más frecuente: durante el encoding no tenés un corpus para contar frecuencias. El orden de los merges es el criterio de prioridad, aprendido durante el entrenamiento.

Paso 9 — decode — IDs a texto
Lo que recibe: una lista de IDs enteros.
Lo que hace: construye el vocabulario inverso {id: token_string} (o lo tenés precalculado como atributo). Por cada ID, buscás el token string correspondiente. Filtrás los tokens especiales como <PAD>, <BOS>, <EOS>. Concatenás todos los strings y reemplazás el marcador </w> por un espacio para reconstruir el texto legible.
Por qué </w> se convierte en espacio: durante el entrenamiento, </w> marcaba el fin de cada palabra. Al decodificar, ese marcador indica dónde van los espacios entre palabras. Es la información que preserva los límites de palabras a través del proceso de tokenización.

Paso 10 — save y load — serialización
save recibe un path y escribe un JSON con dos claves: "vocab" con el diccionario completo y "merges" con la lista de fusiones como listas de dos elementos (JSON no soporta tuplas).
load es un class method — se llama como BPETokenizer.load('vocab/tokenizer.json') y devuelve una instancia ya lista para usar. Lee el JSON, reconstruye el vocab, convierte las listas de merges de vuelta a tuplas, y llama a __init__ con esos valores.
Por qué load es classmethod y no un método de instancia: porque para cargar necesitás crear la instancia. Un classmethod puede devolver cls(...) con los parámetros correctos, que es más limpio que crear una instancia vacía y después poblarla.

Resumen visual del flujo completo
corpus_wikipedia.txt
        ↓
  preprocess.py
  [limpiar → pretokenizar → contar frecuencias]
        ↓
  word_freqs.json
        ↓
  BPETokenizer.train(word_freqs, vocab_size=16000)
  [init vocab → loop: get_pairs → merge_pair → actualizar vocab]
        ↓
  tokenizer.json
        ↓
  BPETokenizer.load('tokenizer.json')
  .encode("texto nuevo")  →  [IDs]
  .decode([IDs])          →  "texto nuevo"

¿Todo claro? ¿Arrancamos con el código?

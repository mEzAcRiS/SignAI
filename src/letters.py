"""Abecedario de la Lengua de Señas Mexicana (LSM) - dactilologia A-Z.

Guia resumida de como formar cada letra con la mano, usada por:

* ``src/collect_data.py`` -> muestra en pantalla como formar la letra
  que se esta capturando.
* ``docs/abecedario_lsm.md`` -> guia impresa para el equipo.

Notas importantes:

* Aqui se describen las configuraciones **estaticas** de la mano (lo que
  una camara captura en un fotograma).
* 6 letras son dinamicas (J, K, Ñ, Q, X, Z): su postura base se describe
  aqui y su trazo lo analiza ``src/motion.py`` (ver MOTION_TRACE_HINTS).
* Fuente de referencia: abecedario oficial SEP / Aprende.mx,
  "Manos con voz" (Fleischmann y Gonzalez Perez, 2011) y el dataset
  "static and dynamic signs for the Mexican Sign Language alphabet".
"""

from __future__ import annotations

# Formacion de cada letra (texto corto, en espanol).
LETTER_HINTS: dict[str, str] = {
    "A": "Puno cerrado con el pulgar al lado del indice (punta arriba).",
    "B": "Mano plana, cuatro dedos estirados y juntos, pulgar doblado sobre la palma.",
    "C": "Mano curvada en forma de C, todos los dedos arqueados.",
    "D": "Indice estirado arriba; los otros tres dedos doblados y el pulgar los toca.",
    "E": "Cuatro dedos doblados hacia abajo con las puntas tocando el pulgar.",
    "F": "Indice y pulgar forman un circulo; los otros tres dedos estirados.",
    "G": "Indice y pulgar estirados y paralelos, apuntando hacia un lado.",
    "H": "Indice y medio estirados juntos, apuntando hacia un lado.",
    "I": "Solo el menique estirado hacia arriba; los demas doblados.",
    "J": "Menique estirado (forma de J); en LSM se completa con movimiento.",
    "K": "Indice y medio arriba en forma de V, el pulgar en medio de los dos.",
    "L": "Indice arriba y pulgar hacia afuera, formando una L.",
    "M": "Indice, medio y anular doblados sobre el pulgar.",
    "N": "Indice y medio doblados sobre el pulgar.",
    "O": "Todas las puntas de los dedos tocan el pulgar formando una O.",
    "P": "Como la K pero con los dedos apuntando hacia abajo.",
    "Q": "Como la G pero con los dedos apuntando hacia abajo.",
    "R": "Indice y medio estirados y cruzados el uno sobre el otro.",
    "S": "Puno cerrado con el pulgar por enfrente de los dedos.",
    "T": "Pulgar entre el indice y el medio, ambos doblados.",
    "U": "Indice y medio estirados y juntos hacia arriba.",
    "V": "Indice y medio estirados y separados (en forma de V).",
    "W": "Indice, medio y anular estirados y separados (en forma de W).",
    "X": "Indice estirado pero enganchado (curvado hacia adentro).",
    "Y": "Pulgar y menique estirados; los tres dedos del medio doblados.",
    "Z": "Indice estirado apuntando; en LSM se completa con movimiento.",
}


def hint_for(letter: str, max_chars: int = 0) -> str:
    """Devuelve la pista de la letra; ``max_chars`` recorta el texto."""
    text = LETTER_HINTS.get(letter.upper(), "")
    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 1] + "..."
    return text


def ascii_hint(letter: str, max_chars: int = 46) -> str:
    """Pista sin acentos ni enies: OpenCV (putText) solo dibuja ASCII."""
    import unicodedata

    text = hint_for(letter, max_chars)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

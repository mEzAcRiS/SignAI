# Abecedario de la Lengua de Señas Mexicana (LSM) — guía de captura

SignAI reconoce las **26 letras** de la dactilología de la Lengua de Señas
Mexicana (LSM). Este documento indica cómo formar cada letra para capturar
muestras con `src/collect_data.py` y para realizar las señas frente a la
cámara en tiempo real.

> **Limitación importante (J y Z):** en la LSM, las letras **J** y **Z** se
> completan con un movimiento (trazo con la letra). El sistema de visión de
> este prototipo clasifica un fotograma estático, por lo que captura la
> **postura base** de ambas letras: el trazo no se modela. Se documenta aquí
> y en el README como limitación conocida del proyecto.

**Postura general de la mano**

- Mostrar la palma o el dorso hacia la cámara según lo pida la letra (si no
  se reconoce, gire la mano: el modelo se entrenó con ambas orientaciones).
- Mantener la mano dentro de la imagen, con buena luz y sin movimientos
  bruscos.
- La normalización del proyecto hace que el tamaño y la posición de la mano
  no cambien la predicción.

## Letras A–Z

| Letra | Cómo formarla |
|---|---|
| **A** | Puño cerrado con el pulgar al lado del índice (punta arriba). |
| **B** | Mano plana, cuatro dedos estirados y juntos, pulgar doblado sobre la palma. |
| **C** | Mano curvada en forma de C, todos los dedos arqueados. |
| **D** | Índice estirado arriba; los otros tres dedos doblados y el pulgar los toca. |
| **E** | Cuatro dedos doblados hacia abajo con las puntas tocando el pulgar. |
| **F** | Índice y pulgar forman un círculo; los otros tres dedos estirados. |
| **G** | Índice y pulgar estirados y paralelos, apuntando hacia un lado. |
| **H** | Índice y medio estirados juntos, apuntando hacia un lado. |
| **I** | Solo el meñique estirado hacia arriba; los demás doblados. |
| **J** | Meñique estirado (forma de J); en LSM se completa con movimiento. ⚠️ |
| **K** | Índice y medio arriba en forma de V, el pulgar en medio de los dos. |
| **L** | Índice arriba y pulgar hacia afuera, formando una L. |
| **M** | Índice, medio y anular doblados sobre el pulgar. |
| **N** | Índice y medio doblados sobre el pulgar. |
| **O** | Todas las puntas de los dedos tocan el pulgar formando una O. |
| **P** | Como la K pero con los dedos apuntando hacia abajo. |
| **Q** | Como la G pero con los dedos apuntando hacia abajo. |
| **R** | Índice y medio estirados y cruzados el uno sobre el otro. |
| **S** | Puño cerrado con el pulgar por enfrente de los dedos. |
| **T** | Pulgar entre el índice y el medio, ambos doblados. |
| **U** | Índice y medio estirados y juntos hacia arriba. |
| **V** | Índice y medio estirados y separados (en forma de V). |
| **W** | Índice, medio y anular estirados y separados (en forma de W). |
| **X** | Índice estirado pero enganchado (curvado hacia adentro). |
| **Y** | Pulgar y meñique estirados; los tres dedos del medio doblados. |
| **Z** | Índice estirado apuntando; en LSM se completa con movimiento. ⚠️ |

⚠️ = letra que en la LSM requiere movimiento (ver limitación arriba).

## Letras que suelen confundirse

Algunas letras tienen posturas muy parecidas y son las más difíciles del
clasificador (se ven en la matriz de confusión de `docs/`):

- **I / J**: mismo meñique arriba (en J se espera el trazo).
- **M / N**: dedos doblados sobre el pulgar (M = tres dedos, N = dos).
- **U / V / W**: dedos arriba juntos (U) o separados (V = 2, W = 3).
- **A / S**: puño cerrado; en A el pulgar queda al lado, en S por enfrente.

Consejo al capturar: hacer **150 muestras por letra** (default `--per-class`)
y variar ligeramente el ángulo de la mano entre tomas.

## Fuentes de referencia

- Abecedario oficial de la SEP / Aprende.mx (dactilología de la LSM).
- *Manos con voz: diccionario bilingüe Lengua de Señas Mexicana*,
  Serafín de Fleischman y Elsa González de Pérez, 2011.

# Abecedario de la Lengua de Señas Mexicana (LSM) — guía de captura

SignAI reconoce las **26 letras** de la dactilología de la Lengua de Señas
Mexicana (LSM). Este documento indica cómo formar cada letra para capturar
muestras con `src/collect_data.py` y para realizar las señas frente a la
cámara en tiempo real.

> **Reconocimiento híbrido para J y Z:** en la LSM, las letras **J** y **Z**
> se completan con un movimiento (trazo en el aire). Este prototipo implementa
> un **clasificador de segundo nivel (híbrido)**:
> 1. **Clasificador estático** (Random Forest): detecta la postura base en cada frame
> 2. **Clasificador de movimiento** (SVM): cuando la postura coincide con I/J/Z,
>    analiza la trayectoria del dedo relevante (meñique=J, índice=Z) sobre
>    ~20 frames consecutivos para decidir si es J o Z
>
> **Para capturar datos de movimiento:** usa `python src/collect_data.py --person Nombre --motion`
> y sigue las instrucciones de trazo abajo. Se recomiendan ~30-50 secuencias por letra.

**Postura general de la mano**

- Mostrar la palma o el dorso hacia la cámara según lo pida la letra (si no
  se reconoce, gire la mano: el modelo se entrenó con ambas orientaciones).
- Mantener la mano dentro de la imagen, con buena luz y sin movimientos
  bruscos.
- La normalización del proyecto hace que el tamaño y la posición de la mano
  no cambien la predicción.

## Letras A–Z

| Letra | Cómo formarla (postura base) |
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
| **J** | Meñique estirado (postura base); **trazo: gancho hacia adentro con el meñique** 🎯 |
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
| **Z** | Índice estirado (postura base); **trazo: zigzag horizontal con el índice** 🎯 |

🎯 = letra con reconocimiento de movimiento (trazo capturado en modo `--motion`).

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

# SignAI — Reconocimiento de dígitos 0–5 con visión por computadora

Prototipo de reconocimiento de señas estáticas (dígitos **0 a 5**) en tiempo real.
La cámara captura la mano, **MediaPipe** extrae 21 puntos clave (landmarks), se forma
un vector de **63 coordenadas (x, y, z)** y un **Random Forest** clasifica la seña,
mostrando el resultado en pantalla.

- **Rama de IA:** Visión por Computadora
- **Equipo 4** — Materia: Fundamentos de IA
- **Repositorio:** https://github.com/mEzAcRiS/SignAI.git

## Estado

| Componente | Estado |
|---|---|
| Entorno (Python 3.14 + MediaPipe 1.0.1) | Listo |
| Detección de landmarks (HandLandmarker) | Verificado con imagen real |
| Código del prototipo | En construcción |
| Dataset (dígitos 0–5) | Pendiente de recolección |
| `.exe` | Pendiente |

## Instalación

```powershell
git clone https://github.com/mEzAcRiS/SignAI.git
cd SignAI
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Ejecución (se completará en la Entrega 1)

```powershell
# Entrenar con los datos de prueba incluidos
.\.venv\Scripts\python.exe src\train.py

# Modo demostración sin cámara (video o imagen)
.\.venv\Scripts\python.exe src\app.py --demo docs\demo.mp4

# Tiempo real con webcam (requiere cámara)
.\.venv\Scripts\python.exe src\app.py
```

## Créditos y licencias

| Dependencia | Versión | Licencia | Fuente |
|---|---|---|---|
| MediaPipe (Hand Landmarker) | 1.0.1 | Apache-2.0 | https://developers.google.com/mediapipe |
| Modelo `hand_landmarker.task` | float16/1 | Apache-2.0 | https://storage.googleapis.com/mediapipe-models/ |
| scikit-learn | 1.9.1 | BSD-3-Clause | https://scikit-learn.org |
| OpenCV (`opencv-contrib-python`) | 5.0.0.93 | Apache-2.0 | https://opencv.org |
| pandas | 3.0.6 | BSD-3-Clause | https://pandas.pydata.org |
| NumPy | 2.5.3 | BSD-3-Clause | https://numpy.org |
| joblib | 1.6.0 | BSD-3-Clause | https://joblib.readthedocs.io |

**Uso de IA generativa:** las partes del proyecto generadas o asistidas por IA y las
modificadas por el equipo se detallarán en esta sección en la Entrega 1 (requisito del
enunciado). La bitácora de prompts se entrega como `06_Bitacora_de_prompts_y_reflexion.docx`.

@echo off
REM ============================================================
REM SignAI - generacion del ejecutable .exe con PyInstaller
REM Requisitos: .venv activo con las dependencias de requirements.txt
REM              y el modelo entrenado en models\model.joblib
REM Salida:      dist\SignAI\SignAI.exe
REM ============================================================
setlocal

set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo [1/3] Verificando modelo entrenado (estatico)...
if not exist "models\model.joblib" (
    echo      No existe models\model.joblib. Entrenando con datos de prueba...
    %PY% src\train.py --data data\samples\hand_landmarks_sample.csv || goto :error
)

echo [1b/3] Verificando modelo de movimiento (J/Z) [opcional]...
set "ADD_MOTION="
if exist "models\motion_model.joblib" (
    echo      Modelo de movimiento encontrado.
    set ADD_MOTION=--add-data "models\motion_model.joblib;models"
) else (
    echo      No existe models\motion_model.joblib - opcional: J/Z usaran solo postura estatica.
)

echo [2/3] Verificando modelo de landmarks...
if not exist "models\hand_landmarker.task" (
    %PY% scripts\download_model.py || goto :error
)

echo [3/3] Construyendo el ejecutable...
%PY% -m PyInstaller --noconfirm --clean ^
    --name SignAI ^
    --onedir ^
    --console ^
    --collect-all mediapipe ^
    --collect-all sklearn ^
    --add-data "models\hand_landmarker.task;models" ^
    --add-data "models\model.joblib;models" ^
    %ADD_MOTION% ^
    --add-data "data\samples;data\samples" ^
    --paths src ^
    src\app.py || goto :error

echo.
echo ============================================================
echo  EXE listo: dist\SignAI\SignAI.exe
echo  Prueba:     dist\SignAI\SignAI.exe --demo-image data\samples\hand_test.jpg
echo ============================================================
exit /b 0

:error
echo ERROR: la construccion fallo.
exit /b 1

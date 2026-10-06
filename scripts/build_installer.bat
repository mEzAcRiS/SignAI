@echo off
REM ============================================================
REM SignAI - generacion del instalador Setup_SignAI_1.0.exe
REM Requisitos: Inno Setup 6 y dist\SignAI\ ya construido
REM              (si no existe, primero se ejecuta build_exe.bat)
REM Salida:      dist\Setup_SignAI_1.0.exe
REM ============================================================
setlocal

set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=ISCC.exe"

echo [1/2] Verificando dist\SignAI\SignAI.exe...
if not exist "dist\SignAI\SignAI.exe" (
    echo      No existe dist\SignAI - construyendo el ejecutable primero...
    call "scripts\build_exe.bat" || goto :error
)

echo [2/2] Compilando el instalador con Inno Setup...
"%ISCC%" "scripts\SignAI.iss" || goto :error

echo.
echo ============================================================
echo  Instalador listo: dist\Setup_SignAI_1.0.0.exe
for %%F in ("dist\Setup_SignAI_1.0.0.exe") do echo  Tamano: %%~zF bytes
echo  Prueba rapida:  dist\Setup_SignAI_1.0.0.exe /VERYSILENT /DIR=%%TEMP%%\signai_test
echo ============================================================
exit /b 0

:error
echo ERROR: la construccion del instalador fallo.
exit /b 1

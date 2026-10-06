@echo off
REM ============================================================
REM SignAI - generacion del instalador Setup_SignAI_1.0.0.exe
REM Requisitos: Inno Setup 6 y dist\SignAI\ ya construido
REM              (si no existe, primero se ejecuta build_exe.bat)
REM Salida:      dist\Setup_SignAI_1.0.0.exe
REM Nota: se compila en %%TEMP%% y se mueve a dist\ porque
REM       Windows/antivirus bloquea la escritura directa de
REM       "Setup_SignAI_*.exe" en dist\ (error de acceso).
REM ============================================================
setlocal

set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=ISCC.exe"
set "OUTDIR=%TEMP%\signai_build"

echo [1/3] Verificando dist\SignAI\SignAI.exe...
if not exist "dist\SignAI\SignAI.exe" (
    echo      No existe dist\SignAI - construyendo el ejecutable primero...
    call "scripts\build_exe.bat" || goto :error
)

echo [2/3] Compilando el instalador con Inno Setup en %%TEMP%%...
if exist "%OUTDIR%" rmdir /s /q "%OUTDIR%"
"%ISCC%" /O"%OUTDIR%" "scripts\SignAI.iss" || goto :error

echo [3/3] Moviendo el instalador a dist\...
del /f /q "dist\Setup_SignAI_1.0.0.exe" >nul 2>&1
move "%OUTDIR%\Setup_SignAI_1.0.0.exe" "dist\Setup_SignAI_1.0.0.exe" >nul || goto :error
rmdir /s /q "%OUTDIR%" >nul 2>&1

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

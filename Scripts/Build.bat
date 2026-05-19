@echo off
cd /d "%~dp0.."
if exist Build\CMakeCache.txt (
    cd Build
    cmake --build . --config Debug
) else (
    if exist Build rmdir /s /q Build
    mkdir Build
    cd Build
    cmake -G "Visual Studio 17 2022" -A x64 ..
    cmake --build . --config Debug
)
if %errorlevel% equ 0 (
    echo.
    echo BUILD SUCCEEDED
) else (
    echo.
    echo BUILD FAILED
)
pause
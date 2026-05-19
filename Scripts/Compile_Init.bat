@echo off
cd /d "%~dp0.."

:: --- Locate Host Compiler using vswhere ---
set "VS_PATH="
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VS_PATH=%%i"

if "%VS_PATH%"=="" (
    echo ERROR: Cannot find Visual Studio installation.
    pause
    exit /b 1
)

set "MSVC_VER="
for /d %%i in ("%VS_PATH%\VC\Tools\MSVC\14.*") do set "MSVC_VER=%%~nxi"
set "CL_DIR=%VS_PATH%\VC\Tools\MSVC\%MSVC_VER%\bin\Hostx64\x64"

:: --- Locate CUDA Toolkit ---
if "%CUDA_PATH%"=="" (
    echo ERROR: CUDA_PATH environment variable is not set.
    pause
    exit /b 1
)

:: --- Compile ---
set "OUT_DIR=Build\Bin\Debug"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo Compiling...
"%CUDA_PATH%\bin\nvcc.exe" Applications\Init\main.cu -ccbin "%CL_DIR%" -I"%CUDA_PATH%\include" -I"Helpers" -std=c++20 -g -G -arch=sm_75 -o "%OUT_DIR%\Init.exe" 2>&1

if %errorlevel% equ 0 (
    echo COMPILE SUCCEEDED
) else (
    echo COMPILE FAILED
)
pause
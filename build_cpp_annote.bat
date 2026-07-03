@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
cd /d E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX\cpp-annote

REM Build zlib first
set ZLIB_SRC=E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX\cpp-annote\zlib
set ZLIB_BUILD=%ZLIB_SRC%\build
if not exist "%ZLIB_BUILD%\zconf.h" (
    if not exist "%ZLIB_BUILD%" mkdir "%ZLIB_BUILD%"
    cd /d "%ZLIB_BUILD%"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release -- /m
    cd /d E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX\cpp-annote
)

REM Check zconf.h generated
if not exist "%ZLIB_BUILD%\zconf.h" copy "%ZLIB_SRC%\zconf.h.in" "%ZLIB_BUILD%\zconf.h"

set ONNXRUNTIME_ROOT=E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX\cpp-annote\onnxruntime-win-x64-1.27.0
set ZLIB_INCLUDE_DIR=%ZLIB_SRC%;%ZLIB_BUILD%
set ZLIB_LIBRARY=%ZLIB_BUILD%\Release\zlib.lib

cmake -B build -G "Visual Studio 18 2026" -DCMAKE_BUILD_TYPE=Release ^
    -DZLIB_INCLUDE_DIR="%ZLIB_SRC%;%ZLIB_BUILD%" ^
    -DZLIB_LIBRARY="%ZLIB_LIBRARY%"
echo CONFIG EXIT CODE: %ERRORLEVEL%
if %ERRORLEVEL% EQU 0 (
    cmake --build build --config Release -- /m
    echo BUILD EXIT CODE: %ERRORLEVEL%
)
pause

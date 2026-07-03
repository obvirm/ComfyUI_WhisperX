@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
cd /d E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX\cpp-annote\zlib
cmake -B build -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% EQU 0 cmake --build build --config Release -- /m
echo ZLIB BUILD EXIT: %ERRORLEVEL%
pause

@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
cd /d E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX\cpp-annote\build
cmake --build . --config Release --target cpp-annote-cli -- /m
echo BUILD EXIT: %ERRORLEVEL%
pause

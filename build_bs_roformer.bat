@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
cd /d E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX
E:\Ai\ComfyUI\python_embeded\python.exe build_bs_roformer.py %*
echo BUILD EXIT CODE: %ERRORLEVEL%
pause

@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
cd /d E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX

rem Fix CUDA_PATH mismatch
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2

python build_whisper_cpp.py --gpu

echo.
echo === Copying DLL ke root ===
copy /Y whisper.cpp\build\bin\Release\whisper.dll .\whisper.dll
copy /Y whisper.cpp\build\bin\Release\ggml-base.dll .\ggml-base.dll
copy /Y whisper.cpp\build\bin\Release\ggml-cpu.dll .\ggml-cpu.dll
copy /Y whisper.cpp\build\bin\Release\ggml.dll .\ggml.dll
copy /Y whisper.cpp\build\bin\Release\ggml-vulkan.dll .\ggml-vulkan.dll 2>nul
copy /Y whisper.cpp\build\bin\Release\ggml-opencl.dll .\ggml-opencl.dll 2>nul

echo.
echo === Files ===
dir whisper.dll ggml-base.dll ggml-cpu.dll ggml.dll 2>nul
echo Done.
pause

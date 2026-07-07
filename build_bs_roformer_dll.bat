@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
cd /d E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX

rem Fix CUDA_PATH mismatch
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2

python build_bs_roformer.py --vulkan --opencl

echo.
echo === Copying DLL ke root ===
copy /Y bs_roformer.cpp\build\Release\bs_roformer.dll .\bs_roformer.dll
dir bs_roformer.dll
echo Done.
pause

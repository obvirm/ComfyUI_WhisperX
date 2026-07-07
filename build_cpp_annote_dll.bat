@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
cd /d E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX

rem Fix CUDA_PATH mismatch
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2

python build_cpp_annote.py

echo.
echo === Copying DLL ke root ===
copy /Y cpp-annote\build\Release\cpp_annote.dll .\cpp_annote.dll
copy /Y cpp-annote\onnxruntime-win-x64-1.27.0\lib\onnxruntime.dll .\onnxruntime.dll
copy /Y cpp-annote\onnxruntime-win-x64-1.27.0\lib\onnxruntime_providers_shared.dll .\onnxruntime_providers_shared.dll

echo.
echo === Files ===
dir cpp_annote.dll onnxruntime.dll 2>nul
echo Done.
pause

@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
cd /d E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX\bs_roformer.cpp

cmake -B build -G "Visual Studio 18 2026" -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=OFF -DBSR_BUILD_CLI=OFF -DBSR_BUILD_SHARED=ON
echo CONFIG: %ERRORLEVEL%

if %ERRORLEVEL% EQU 0 (
    cmake --build build --config Release --target bs_roformer_shared -- /m
    echo BUILD: %ERRORLEVEL%
)

echo.
echo === Copying DLL ke root ===
copy /Y build\Release\bs_roformer.dll ..\bs_roformer.dll
dir ..\bs_roformer.dll
echo Done.
pause

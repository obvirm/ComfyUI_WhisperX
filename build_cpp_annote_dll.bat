@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
cd /d E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX\cpp-annote

set ONNXRUNTIME_ROOT=E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX\cpp-annote\onnxruntime-win-x64-1.27.0
set ZLIB_ROOT=E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX\cpp-annote\zlib\build
set ZLIB_INCLUDE_DIR=E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX\cpp-annote\zlib;E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX\cpp-annote\zlib\build
set ZLIB_LIBRARY=E:\Ai\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WhisperXX\cpp-annote\zlib\build\Release\zlibstatic.lib
set CMAKE_PREFIX_PATH=%ZLIB_ROOT%

cmake -B build -G "Visual Studio 18 2026" -DCMAKE_BUILD_TYPE=Release ^
    -DZLIB_ROOT="%ZLIB_ROOT%" ^
    -DZLIB_INCLUDE_DIR="%ZLIB_INCLUDE_DIR%" ^
    -DZLIB_LIBRARY="%ZLIB_LIBRARY%" ^
    -DCPPANNOTE_BUILD_SHARED=ON
echo CONFIG: %ERRORLEVEL%

if %ERRORLEVEL% EQU 0 (
    cmake --build build --config Release --target cpp_annote_shared -- /m
    echo BUILD: %ERRORLEVEL%
)

echo.
echo === Copying DLL ke root ===
copy /Y build\Release\cpp_annote.dll ..\cpp_annote.dll

echo === Copying ONNX Runtime DLL ===
copy /Y "%ONNXRUNTIME_ROOT%\lib\onnxruntime.dll" ..\onnxruntime.dll
copy /Y "%ONNXRUNTIME_ROOT%\lib\onnxruntime_providers_shared.dll" ..\onnxruntime_providers_shared.dll

echo.
echo === Files ===
dir ..\cpp_annote.dll ..\onnxruntime.dll 2>nul
echo Done.
pause

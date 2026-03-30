# Compilation
Compiled using `x64 Native Tools Command Prompt for VS`, i.e. `vcvars64.bat`.

Generate Ninja build scripts using.
```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```
Invoke Ninja: build/compile and link the source files.
```cmd
cmake --build build
```
run using.
```cmd
.\build\gpu_app.exe
.\build\cpu_app.exe
```
## Optional: Build and run
```cmd
cmake --build build --target run_cpu
cmake --build build --target run_gpu
```

# Results
- **CPU Vector Add Time:** 55ms
- **GPU Vector Add Time:** 5ms
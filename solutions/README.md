# Hardware
- **GPU:** NVIDIA GeForce GTX 1660 Ti
- **Compute Capability:** 7.5

# Software
- **OS:** `Windows 11 23H2`
- **Driver Version:** `595.71`
- **CUDA Version:** `13.2`
- **C++ Compiler:** `Microsoft (R) C/C++ Optimizing Compiler Version 19.50.35725 for x64`
- **NVCC:** `Cuda compilation tools, release 13.2, V13.2.51; Build cuda_13.2.r13.2/compiler.37434383_0`
- **Build System:** `ninja 1.13.2`

# Compilation
Compiled using `x64 Native Tools Command Prompt for VS`, i.e. `vcvars64.bat`.
```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
run using 
```
.\build\cuda_app.exe
```
## Single command to compile and execute
```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --target run
```


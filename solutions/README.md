# Hardware
- **CPU:** AMD Ryzen 7 7800X3D 8-Core Processor
- **GPU:** NVIDIA GeForce GTX 1660 Ti
- **Compute Capability:** 7.5

Queried specs from the GPU.
```
CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce GTX 1660 Ti"
  CUDA Driver Version / Runtime Version          13.2 / 13.2
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 6144 MBytes (6442123264 bytes)
  (024) Multiprocessors, (064) CUDA Cores/MP:    1536 CUDA Cores
  GPU Max Clock rate:                            1830 MHz (1.83 GHz)
  Memory Clock rate:                             6001 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 1572864 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        65536 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 13.2, CUDA Runtime Version = 13.2, NumDevs = 1
Result = PAS
```

# Software
- **OS:** `Windows 11 23H2`
- **Driver Version:** `595.71`
- **CUDA Version:** `13.2`
- **C++ Compiler:** `Microsoft (R) C/C++ Optimizing Compiler Version 19.50.35725 for x64`
- **NVCC:** `Cuda compilation tools, release 13.2, V13.2.51; Build cuda_13.2.r13.2/compiler.37434383_0`
- **Build System:** `ninja 1.13.2`

# Compilation
Generally compilation follows these steps, unless specified oterwise in the README of the day. 

Compiled using `x64 Native Tools Command Prompt for VS`, i.e. `vcvars64.bat`.

Generate Ninja build scripts using.
```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```
Invoke Ninja: build/compile and link the source files.
```cmd
cmake --build build
```cmd
run using.
```cmd
.\build\cuda_app.exe
```
## Single command to compile and execute
```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --target run
```


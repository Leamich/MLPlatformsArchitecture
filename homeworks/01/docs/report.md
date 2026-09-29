# Отчет по Практике-1

*Леонтьев Михаил*

## CPU

**CPU**: Intel(R) Core(TM) Ultra 5 225H

**GPU**: Intel(R) Arc(TM) 130T GPU (16GB) (встроенный)

14 физических ядер, 1 поток на ядро: `4P + 8E + 2LPE`.

**Instructions sets**: `MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, EM64T, AES, AVX, AVX2, FMA3, SHA`

* **L1 Data cache**: `4 x 48 KB + 8 x 32 KB + 2 x 32 KB` (64-byte line)
* **L1 Instruction cache**:	`4 x 64 KB + 8 x 64 KB + 2 x 64 KB` (64-byte line)
* **L2 cache**:		`4 x 3 MB + 2 x 4 MB + 2 MB` (64-byte line)
* **L3 cache**:		`18 MB` (64-byte line)

(то есть L2-кэш для E и LPE ядер сгруппирован по 2 ядра)

Одна нода, NUMA неактивна.

## clinfo

```
Number of platforms                               1
  Platform Name                                   Intel(R) OpenCL Graphics
  Platform Vendor                                 Intel(R) Corporation
  Platform Version                                OpenCL 3.0 
  Platform Profile                                FULL_PROFILE
<...>
  Platform Numeric Version                        0xc00000 (3.0.0)
  Platform Extensions function suffix             INTEL
  Platform Host timer resolution                  1ns
  Platform External memory handle types           DMA buffer

  Platform Name                                   Intel(R) OpenCL Graphics
Number of devices                                 1
  Device Name                                     Intel(R) Graphics [0x7d51]
  Device Vendor                                   Intel(R) Corporation
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 3.0 NEO 
  Device UUID                                     8680517d-0300-0000-0002-000000000000
  Driver UUID                                     32362e30-352e-3033-3730-323000000000
  Valid Device LUID                               No
  Device LUID                                     6c10-000000000000
  Device Node Mask                                0
  Device Numeric Version                          0xc00000 (3.0.0)
<...>
```

## Скомпилированные программы

```
$cmake --build --preset linux-debug --target cpu_probe && ./build/linux-debug/homeworks/01/cpu_probe

[100%] Built target cpu_probe

=== CPU Probe (C++ Runtime) ===

Brand: Intel(R) Core(TM) Ultra 5 225H
Logical cores (std::thread): 14

SIMD Extensions:
  SSE: YES ✓
  SSE2: YES ✓
  SSE4.1: YES ✓
  AVX: YES ✓
  AVX2: YES ✓
  AVX-512F: NO ✗
  FMA: YES ✓
  FP16C: YES ✓

Note: std::thread::hardware_concurrency() returns LOGICAL cores.
To get PHYSICAL cores, use lscpu (Linux) or Coreinfo (Windows).
```

```
$ cmake --build --preset linux-debug --target cl_query && ./build/linux-debug/homeworks/01/cl_query

[100%] Built target cl_query

Platform 0: Intel(R) OpenCL Graphics
----------------------------------------
  Device Name: Intel(R) Graphics [0x7d51]
  Vendor: Intel(R) Corporation
  Max Compute Units: 112
  Max Clock Frequency (MHz): 2200
  Global Memory (MB): 16831
  Type: GPU 
```

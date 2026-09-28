# Практическая работа: Настройка окружения для гетерогенных вычислений (OpenCL и CUDA)

**Цель работы:** Научиться исследовать аппаратную архитектуру вычислительной системы, настраивать кроссплатформенное окружение для разработки на C++ с использованием OpenCL и CUDA, конфигурировать систему сборки (CMake) и получать программный доступ к гетерогенным вычислительным ресурсам (CPU/GPU) ноутбука.

---

## Этап 1. Первичный аудит оборудования

Прежде чем устанавливать софт, необходимо понять, какие вычислительные устройства находятся в вашем ноутбуке.

**Задание:** Определить модель CPU и наличие/модель GPU (встроенного и дискретного).

*   **Linux:** Выполните в терминале:
    ```bash
    lscpu | grep "Model name"
    lspci | grep -iE 'vga|3d|display'
    ```
*   **Windows:** Откройте «Диспетчер устройств» → «Видеоадаптеры» или выполните `dxdiag` и посмотрите вкладки «Экран» (Display).

**CPU**: Intel(R) Core(TM) Ultra 5 225H

**GPU**: Intel(R) Arc(TM) 130T GPU (16GB) (встроенный)

> **Примечание для пользователей MacBook (Apple Silicon M1/M2/M3):** OpenCL и CUDA на них официально не поддерживаются.

---

## Этап 2. Глубокое исследование архитектуры CPU

Понимание микроархитектуры процессора критически важно для гетерогенных вычислений. Когда OpenCL-ядро запускается на CPU (через POCL или Intel CPU Runtime), именно эти характеристики определяют, насколько эффективно будет работать ваш код.

### 2.1. Количество ядер и потоков

Современные процессоры используют технологию **SMT (Simultaneous Multithreading)** — Hyper-Threading у Intel, SMT у AMD. Это значит, что одно физическое ядро может выполнять два потока одновременно.

**Задание:** Определить количество физических ядер, логических процессоров и потоков на ядро.

*   **Linux:**
    ```bash
    # Краткая сводка
    lscpu | grep -E 'Socket|Core|Thread|CPU\(s\):|Model name'

    # Детальная таблица: какое логическое ядро на каком физическом ядре и в каком сокете
    lscpu -e
    ```
    *Пример вывода:* `CPU(s): 16, Thread(s) per core: 2, Core(s) per socket: 8, Socket(s): 1` — это значит 8 физических ядер, 16 логических потоков.

*   **Windows (PowerShell):**
    ```powershell
    Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
    ```

*   **Windows (cmd):**
    ```cmd
    wmic cpu get Name, NumberOfCores, NumberOfLogicalProcessors
    ```

```
CPU(s):                                  8
Model name:                              Intel(R) Core(TM) Ultra 5 225H
Thread(s) per core:                      1
Core(s) per socket:                      8
Socket(s):                               1
NUMA node0 CPU(s):                       0-7
CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE
  0    0      0    0 0:0:0:0          yes
  1    0      0    1 1:1:1:0          yes
  2    0      0    2 2:2:2:0          yes
  3    0      0    3 3:3:3:0          yes
  4    0      0    4 4:4:4:0          yes
  5    0      0    5 5:5:5:0          yes
  6    0      0    6 6:6:6:0          yes
  7    0      0    7 7:7:7:0          yes
```

```
Name                           NumberOfCores NumberOfLogicalProcessors
----                           ------------- -------------------------
Intel(R) Core(TM) Ultra 5 225H            14                        14
```


### 2.2. Поддерживаемые инструкции и SIMD-расширения

Набор инструкций CPU определяет, какие векторные операции можно использовать для ускорения вычислений. Для гетерогенных вычислений наиболее важны **SIMD-расширения** (Single Instruction, Multiple Data):

| Расширение | Ширина вектора | Типичное применение |
|---|---|---|
| SSE / SSE2 | 128 бит (4 × float) | Базовые векторные операции |
| SSE4.1 / SSE4.2 | 128 бит | Строковые операции, CRC |
| AVX | 256 бит (8 × float) | Матричные вычисления |
| AVX2 | 256 бит (целые + FMA) | Обработка изображений |
| AVX-512 | 512 бит (16 × float) | Научные вычисления, ML |
| FMA (Fused Multiply-Add) | — | `a * b + c` за один такт |
| FP16C | 16 бит float | Инференс нейросетей |

**Задание:** Определить, какие SIMD-расширения поддерживает ваш процессор.

*   **Linux:**
    ```bash
    # Все флаги в одну строку
    lscpu | grep "Flags" | tr ' ' '\n' | grep -iE 'sse|avx|fma|fp16' | sort

    # Или через /proc/cpuinfo (для первого ядра)
    grep -m1 flags /proc/cpuinfo | tr ' ' '\n' | grep -iE 'sse|avx|fma'
    ```
    *Если в выводе есть `avx2` и `fma` — ваш CPU поддерживает 256-битные векторные вычисления. Если есть `avx512f` — поздравляем, у вас серверный или топовый десктопный CPU.*

*   **Windows:**
    *   Скачайте утилиту [Coreinfo](https://learn.microsoft.com/en-us/sysinternals/downloads/coreinfo) от Sysinternals и запустите:
        ```cmd
        coreinfo -f
        ```
    *   Или используйте бесплатную утилиту [CPU-Z](https://www.cpuid.com/softwares/cpu-z.html) (вкладка «Instructions»).

```
Instructions sets	MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, EM64T, AES, AVX, AVX2, FMA3, SHA
```

### 2.3. Иерархия кэш-памяти

Кэш — это быстрая память внутри процессора. Понимание её размеров критически важно для оптимизации OpenCL-ядер: если ваш рабочий набор данных помещается в L1-кэш (~32–48 КБ на ядро), производительность будет максимальной. Если данные «вылезают» в L3 или RAM — скорость падает в 10–100 раз.

Типичная иерархия:
```
Регистры (~1 КБ, ~1 такт)
  └─ L1 Data + L1 Instruction (~32–48 КБ на ядро, ~3–5 тактов)
       └─ L2 (~256 КБ – 1.25 МБ на ядро, ~10–15 тактов)
            └─ L3 (~4–32 МБ на весь чип, общий, ~30–50 тактов)
                 └─ RAM (~8–64 ГБ, ~100–300 тактов)
```

**Задание:** Определить размеры кэшей L1, L2, L3.

*   **Linux:**
    ```bash
    # Кратко
    lscpu | grep -i cache

    # Подробно через sysfs
    for i in 0 1 2 3; do
        echo "L$i: $(cat /sys/devices/system/cpu/cpu0/cache/index$i/size 2>/dev/null || echo 'N/A') \
              ($(cat /sys/devices/system/cpu/cpu0/cache/index$i/type 2>/dev/null))"
    done

    # Альтернативно, через getconf
    getconf -a | grep CACHE
    ```

*   **Windows (PowerShell):**
    ```powershell
    Get-CimInstance Win32_CacheMemory | Select-Object Level, MaxCacheSize, CacheType
    ```

*   **Windows (Coreinfo):**
    ```cmd
    coreinfo -c
    ```

```
L1d cache:                               384 KiB (8 instances)
L1i cache:                               512 KiB (8 instances)
L2 cache:                                24 MiB (8 instances)
L3 cache:                                18 MiB (1 instance)

L0: 48K           (Data)
L1: 64K           (Instruction)
L2: 3072K           (Unified)
L3: 18432K           (Unified)

L1 Data cache		4 x 48 KB (12-way, 64-byte line) + 8 x 32 KB (8-way, 64-byte line) + 2 x 32 KB (8-way, 64-byte line)
L1 Instruction cache	4 x 64 KB (16-way, 64-byte line) + 8 x 64 KB (8-way, 64-byte line) + 2 x 64 KB (8-way, 64-byte line)
L2 cache		4 x 3 MB (12-way, 64-byte line) + 2 x 4 MB (16-way, 64-byte line) + 2 MB (16-way, 64-byte line)
L3 cache		18 MB (12-way, 64-byte line)
```

### 2.4. NUMA-топология

**NUMA (Non-Uniform Memory Access)** — это архитектура, при которой каждый процессор (или группа ядер) имеет «свою» локальную память, доступ к которой быстрее, чем к памяти другого процессора. На **ноутбуках** NUMA, как правило, отсутствует (одна NUMA-нода), но на **серверах** с двумя и более сокетами это критически важно для производительности OpenCL/CUDA.

**Задание:** Определить количество NUMA-нод и привязку ядер к ним.

*   **Linux:**
    ```bash
    # Установить утилиту, если нет
    sudo apt install numactl

    # Показать NUMA-топологию
    numactl --hardware
    ```
    *Ожидаемый вывод для ноутбука:* `available: 1 nodes (0)` — одна нода, NUMA неактивна.
    *Ожидаемый вывод для сервера:* `available: 2 nodes (0-1)` с привязкой ядер к каждой ноде.

    Для визуализации можно также использовать:
    ```bash
    sudo apt install hwloc
    lstopo --no-io
    ```
    Эта команда выведет красивую схему всей иерархии: сокеты → NUMA-ноды → L3 → L2 → L1 → ядра → потоки.

*   **Windows (Coreinfo):**
    ```cmd
    coreinfo -n
    ```

```
available: 1 nodes (0)
node 0 cpus: 0 1 2 3 4 5 6 7
node 0 size: 15993 MB
node 0 free: 14882 MB
node distances:
node     0
   0:   10
```

### 2.5. Программное исследование CPU на C++

Напишите программу, которая **программно** определяет характеристики CPU. Это полезно, когда вы пишете OpenCL-хост-код и хотите автоматически подбирать размер work-group под конкретное железо.

**Файл `CMakeLists.txt` (добавьте к существующему):**
```cmake
cmake_minimum_required(VERSION 3.10)
project(cpu_probe LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(cpu_probe cpu_probe.cpp)
target_link_libraries(cpu_probe PRIVATE Threads::Threads)
```

**Файл `cpu_probe.cpp`:**
```cpp
#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define HAS_CPUID 1
    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
#endif

struct CpuInfo {
    std::string brand;
    unsigned int logicalCores = 0;
    bool hasSSE   = false;
    bool hasSSE2  = false;
    bool hasSSE41 = false;
    bool hasAVX   = false;
    bool hasAVX2  = false;
    bool hasAVX512F = false;
    bool hasFMA   = false;
    bool hasFP16C = false;
};

#ifdef HAS_CPUID
void runCpuid(unsigned int leaf, unsigned int subleaf,
              unsigned int& eax, unsigned int& ebx,
              unsigned int& ecx, unsigned int& edx) {
    #if defined(_MSC_VER)
        int regs[4];
        __cpuidex(regs, leaf, subleaf);
        eax = regs[0]; ebx = regs[1]; ecx = regs[2]; edx = regs[3];
    #else
        __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
    #endif
}
#endif

CpuInfo probeCpu() {
    CpuInfo info;
    info.logicalCores = std::thread::hardware_concurrency();

#ifdef HAS_CPUID
    // --- Brand string (leaves 0x80000002..0x80000004) ---
    char brand[49] = {};
    for (unsigned int i = 0; i < 3; ++i) {
        unsigned int eax, ebx, ecx, edx;
        runCpuid(0x80000002 + i, 0, eax, ebx, ecx, edx);
        std::memcpy(brand + i * 16,     &eax, 4);
        std::memcpy(brand + i * 16 + 4, &ebx, 4);
        std::memcpy(brand + i * 16 + 8, &ecx, 4);
        std::memcpy(brand + i * 16 + 12, &edx, 4);
    }
    info.brand = brand;

    // --- Feature flags (leaf 1 and leaf 7) ---
    unsigned int eax, ebx, ecx, edx;

    // Leaf 1: SSE, SSE2, SSE4.1, AVX, FMA, FP16C
    runCpuid(1, 0, eax, ebx, ecx, edx);
    info.hasSSE   = (edx >> 25) & 1;
    info.hasSSE2  = (edx >> 26) & 1;
    info.hasSSE41 = (ecx >> 19) & 1;
    info.hasAVX   = (ecx >> 28) & 1;
    info.hasFMA   = (ecx >> 12) & 1;
    info.hasFP16C = (ecx >> 29) & 1;

    // Leaf 7, subleaf 0: AVX2, AVX-512F
    runCpuid(7, 0, eax, ebx, ecx, edx);
    info.hasAVX2    = (ebx >> 5) & 1;
    info.hasAVX512F = (ebx >> 16) & 1;
#endif

    return info;
}

int main() {
    CpuInfo cpu = probeCpu();

    std::cout << "=== CPU Probe (C++ Runtime) ===\n\n";
    std::cout << "Brand: " << cpu.brand << "\n";
    std::cout << "Logical cores (std::thread): " << cpu.logicalCores << "\n\n";

    std::cout << "SIMD Extensions:\n";
    auto printFlag = [](const char* name, bool val) {
        std::cout << "  " << name << ": " << (val ? "YES ✓" : "NO ✗") << "\n";
    };
    printFlag("SSE",       cpu.hasSSE);
    printFlag("SSE2",      cpu.hasSSE2);
    printFlag("SSE4.1",    cpu.hasSSE41);
    printFlag("AVX",       cpu.hasAVX);
    printFlag("AVX2",      cpu.hasAVX2);
    printFlag("AVX-512F",  cpu.hasAVX512F);
    printFlag("FMA",       cpu.hasFMA);
    printFlag("FP16C",     cpu.hasFP16C);

    std::cout << "\nNote: std::thread::hardware_concurrency() returns LOGICAL cores.\n";
    std::cout << "To get PHYSICAL cores, use lscpu (Linux) or Coreinfo (Windows).\n";

    return 0;
}
```

---

## Этап 3. Настройка окружения OpenCL

OpenCL использует механизм **ICD (Installable Client Driver)**. Это значит, что вам нужно установить *OpenCL Loader* (библиотеку-посредник) и *ICD драйвер* (непосредственно реализацию для вашего железа или софтверный фоллбэк).

### Путь А: Аппаратный OpenCL (Если есть рабочий GPU)

#### Linux (Ubuntu / Debian)
```bash
# 1. Базовые заголовки и загрузчик ICD
sudo apt update
sudo apt install opencl-headers ocl-icd-libopencl1 clinfo

# 2. Runtime для вашего железа (выберите нужное):
sudo apt install intel-opencl-icd      # Для Intel CPU и встроенной графики
sudo apt install amd-opencl-icd        # Для AMD CPU и графики
sudo apt install nvidia-opencl-icd-535 # Для Nvidia (укажите актуальную версию)
```

#### 🪟 Windows
1. Убедитесь, что установлены свежие драйверы GPU (они сами кладут `OpenCL.dll` в `System32`).
2. Для CPU (если нет дискретной видеокарты) скачайте [Intel® CPU Runtime for OpenCL](https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html).
3. Установите библиотеку и заголовки через **vcpkg** (рекомендуется):
   ```cmd
   vcpkg install opencl:x64-windows
   ```
   *Интегрируйте vcpkg в CMake через `CMAKE_TOOLCHAIN_FILE`.*

### Путь Б: Программный OpenCL (Если нет GPU или проблемы с драйверами)

Если на ноутбуке нет GPU, или драйверы работают нестабильно, мы используем **POCL (Portable Computing Language)** — открытую реализацию OpenCL, работающую исключительно на CPU через LLVM.

#### Linux
```bash
sudo apt install pocl-opencl-icd clinfo
```

#### Windows
Готового `.msi` установщика нет. Используйте один из вариантов:
1. **Через Conda (Рекомендуется):**
   ```cmd
   conda create -n opencl_env -c conda-forge pocl clinfo
   conda activate opencl_env
   ```
2. **Через MSYS2 / MINGW64:**
   ```bash
   pacman -S mingw-w64-x86_64-pocl mingw-w64-x86_64-clinfo
   ```

---

## Этап 4. Проверка окружения OpenCL

Прежде чем писать код, убедитесь, что ОС «видит» устройства.

**Задание:** Запустите утилиту `clinfo` в терминале.
*   Вы должны увидеть разделы `Platform Name` (например, *Portable Computing Language*, *Intel(R) OpenCL*, *NVIDIA CUDA*) и `Device Name`.
*   Если `clinfo` выдает `Number of platforms: 0` — окружение настроено неверно. Проверьте наличие ICD-файлов в `/etc/OpenCL/vendors/` (Linux) или записей в реестре (Windows).

---

## Этап 5. Написание кода и сборка (OpenCL + CMake)

Для первой работы мы используем **чистый C API OpenCL**, обернутый в современный C++ код.

### 1. Файл `CMakeLists.txt` (для OpenCL)
```cmake
cmake_minimum_required(VERSION 3.10)
project(OpenCL_Env_Check LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

add_executable(cl_query main_cl.cpp)
target_compile_definitions(cl_query PRIVATE CL_TARGET_OPENCL_VERSION=300)
target_link_libraries(cl_query PRIVATE OpenCL::OpenCL)
```

### 2. Файл `main_cl.cpp`
```cpp
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>

void printDeviceInfo(cl_device_id device) {
    char buffer[1024];
    size_t size;

    auto getStringInfo = [&](cl_device_info param) {
        clGetDeviceInfo(device, param, sizeof(buffer), buffer, &size);
        return std::string(buffer);
    };

    auto getUintInfo = [&](cl_device_info param) {
        cl_ulong val = 0;
        clGetDeviceInfo(device, param, sizeof(val), &val, nullptr);
        return val;
    };

    std::cout << "  Device Name: " << getStringInfo(CL_DEVICE_NAME) << "\n";
    std::cout << "  Vendor: " << getStringInfo(CL_DEVICE_VENDOR) << "\n";
    std::cout << "  Max Compute Units: " << getUintInfo(CL_DEVICE_MAX_COMPUTE_UNITS) << "\n";
    std::cout << "  Max Clock Frequency (MHz): " << getUintInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY) << "\n";
    std::cout << "  Global Memory (MB): " << (getUintInfo(CL_DEVICE_GLOBAL_MEM_SIZE) / (1024 * 1024)) << "\n";

    cl_device_type type;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
    std::cout << "  Type: ";
    if (type & CL_DEVICE_TYPE_CPU) std::cout << "CPU ";
    if (type & CL_DEVICE_TYPE_GPU) std::cout << "GPU ";
    std::cout << "\n\n";
}

int main() {
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    if (numPlatforms == 0) {
        std::cerr << "OpenCL platforms not found!\n";
        return 1;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    for (cl_uint i = 0; i < numPlatforms; ++i) {
        char platName[256];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platName), platName, nullptr);
        std::cout << "Platform " << i << ": " << platName << "\n";
        std::cout << std::string(40, '-') << "\n";

        cl_uint numDevices = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);

        if (numDevices > 0) {
            std::vector<cl_device_id> devices(numDevices);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
            for (cl_uint j = 0; j < numDevices; ++j) printDeviceInfo(devices[j]);
        } else {
            std::cout << "  No devices found.\n\n";
        }
    }
    return 0;
}
```

---

## Этап 6. Настройка окружения CUDA (Только для NVIDIA GPU)

*Этот этап выполняется только если на ноутбуке есть видеокарта NVIDIA. CUDA — это проприетарная платформа параллельных вычислений от NVIDIA, которая работает быстрее OpenCL на их железе, но не кроссплатформенна.*

### 1. Установка CUDA Toolkit
*   **Linux:** Рекомендуется ставить через `apt`. Убедитесь, что стоит проприетарный драйвер NVIDIA.
    ```bash
    sudo apt install nvidia-cuda-toolkit
    ```
*   **Windows:** Скачайте [CUDA Toolkit Installer](https://developer.nvidia.com/cuda-downloads) с сайта NVIDIA. При установке можно снять галочку с драйвера, если у вас уже стоит свежий Game Ready / Studio драйвер.

### 2. Проверка установки
Откройте терминал и выполните:
```bash
nvcc --version       # Должна вывести версию компилятора CUDA
nvidia-smi           # Должна вывести таблицу с вашей видеокартой и версией драйвера
```

### 3. Код и сборка (CUDA + CMake)

**Файл `CMakeLists.txt` (для CUDA):**
```cmake
cmake_minimum_required(VERSION 3.18) # CUDA требует CMake 3.18+
project(CUDA_Env_Check LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Адаптируйте список архитектур под ваш GPU (см. Compute Capability из cuda_query).
# Пример: 6.1 (GTX 10xx), 7.5 (RTX 20xx / T-series), 8.6 (RTX 30xx), 8.9 (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

add_executable(cuda_query main_cuda.cu)
target_link_libraries(cuda_query PRIVATE CUDA::cudart)
```

**Файл `main_cuda.cu`:**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return 1; \
        } \
    } while (0)

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, i));

        std::cout << "Device " << i << ": " << props.name << "\n";
        std::cout << std::string(40, '-') << "\n";
        std::cout << "  Compute Capability: " << props.major << "." << props.minor << "\n";
        std::cout << "  Multiprocessors (SM): " << props.multiProcessorCount << "\n";
        std::cout << "  Max Threads per Block: " << props.maxThreadsPerBlock << "\n";
        std::cout << "  Total Global Memory (MB): " << (props.totalGlobalMem / (1024 * 1024)) << "\n";
        std::cout << "  Clock Rate (MHz): " << props.clockRate / 1000 << "\n\n";
    }
    return 0;
}
```

---

## Этап 7. Отчет

Студент должен предоставить преподавателю отчет (PDF/Markdown), содержащий:
1. Таблицу характеристик CPU (ядра, кэши, SIMD, NUMA) — из Этапа 2.
2. Скриншот вывода `clinfo` (и `nvidia-smi` / `nvcc --version` для CUDA).
3. Скриншоты вывода скомпилированных программ (`cpu_probe`, `cl_query`, `cuda_query`).

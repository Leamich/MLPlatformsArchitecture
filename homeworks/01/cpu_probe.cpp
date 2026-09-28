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

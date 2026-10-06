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

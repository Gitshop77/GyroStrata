// Small SDK-compiled probe: no guessed HIP property offsets or display-adapter numbering.
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>

int main() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) return 1;
    std::cout << '[';
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t properties{};
        if (hipGetDeviceProperties(&properties, i) != hipSuccess) return 2;
        // The pinned Strata Windows HIP backend uses this same SDK-provided LUID.
        uint32_t low = 0, high = 0;
        static_assert(sizeof properties.luid == 8, "Unexpected HIP Windows LUID size");
        std::memcpy(&low, properties.luid, 4);
        std::memcpy(&high, properties.luid + 4, 4);
        char identity[18];
        std::snprintf(identity, sizeof identity, "%08x_%08x", high, low);
        char arch[256];
        std::snprintf(arch, sizeof arch, "%s", properties.gcnArchName);
        if (char* feature = std::strchr(arch, ':')) *feature = 0;
        if (i) std::cout << ',';
        std::cout << "{\"index\":" << i << ",\"name\":" << std::quoted(properties.name)
                  << ",\"arch\":" << std::quoted(arch) << ",\"luid\":" << std::quoted(identity)
                  << ",\"total_mib\":" << properties.totalGlobalMem / (1024 * 1024)
                  << ",\"wave_size\":" << properties.warpSize
                  << ",\"integrated\":" << (properties.integrated ? "true" : "false") << '}';
    }
    std::cout << "]\n";
    return 0;
}

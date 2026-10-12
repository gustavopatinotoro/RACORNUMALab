//
//  SystemInfo.cpp
//  RACORNUMALab
//
//  Implementación macOS/Apple Silicon de las consultas de hardware.
//

#include "platform/apple/SystemInfo.hpp"

#include <sys/sysctl.h>

#include <cstdint>
#include <string>
#include <vector>

namespace racorn_uma::platform::apple {

namespace {

std::string read_sysctl_string(const char* name) {

    std::size_t size = 0;

    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) {
        return "desconocido";
    }

    std::vector<char> buffer(size);

    if (sysctlbyname(name, buffer.data(), &size, nullptr, 0) != 0) {
        return "desconocido";
    }

    if (!buffer.empty() && buffer.back() == '\0') {
        return std::string(buffer.data());
    }

    return std::string(buffer.begin(), buffer.end());
}

} // namespace

std::string SystemInfo::hardware_model() {
    return read_sysctl_string("hw.model");
}

std::string SystemInfo::machine() {
    return read_sysctl_string("hw.machine");
}

std::uint64_t SystemInfo::physical_memory_bytes() {

    std::uint64_t value = 0;
    std::size_t size = sizeof(value);

    if (sysctlbyname(
            "hw.memsize",
            &value,
            &size,
            nullptr,
            0
        ) != 0) {

        return 0;
    }

    return value;
}

int SystemInfo::logical_cpu_count() {

    int value = 0;
    std::size_t size = sizeof(value);

    if (sysctlbyname(
            "hw.logicalcpu",
            &value,
            &size,
            nullptr,
            0
        ) != 0) {

        return 0;
    }

    return value;
}

} // namespace racorn_uma::platform::apple

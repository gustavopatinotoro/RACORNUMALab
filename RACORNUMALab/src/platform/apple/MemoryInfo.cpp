//
//  MemoryInfo.cpp
//  RACORNUMALab
//
//  Propósito:
//  Implementar telemetría de memoria del proceso mediante TASK_VM_INFO.
//
//  Plataforma:
//  macOS / Darwin / Apple Silicon.
//
//  Seguridad:
//  task_info() se consulta únicamente sobre mach_task_self().
//  No se inspeccionan otros procesos.
//

#include "platform/apple/MemoryInfo.hpp"

#include <mach/mach.h>
#include <mach/task_info.h>

#include <cstdint>

namespace racorn_uma::platform::apple {

ProcessMemorySnapshot
MemoryInfo::capture() noexcept {

    task_vm_info_data_t info{};

    mach_msg_type_number_t count =
        TASK_VM_INFO_COUNT;

    const kern_return_t status =
        task_info(
            mach_task_self(),
            TASK_VM_INFO,
            reinterpret_cast<task_info_t>(&info),
            &count
        );

    if (status != KERN_SUCCESS) {
        return {};
    }

    ProcessMemorySnapshot snapshot;

    snapshot.resident_bytes =
        static_cast<std::uint64_t>(
            info.resident_size
        );

    snapshot.resident_peak_bytes =
        static_cast<std::uint64_t>(
            info.resident_size_peak
        );

    snapshot.physical_footprint_bytes =
        static_cast<std::uint64_t>(
            info.phys_footprint
        );

    snapshot.compressed_bytes =
        static_cast<std::uint64_t>(
            info.compressed
        );

    snapshot.valid = true;

    return snapshot;
}

} // namespace racorn_uma::platform::apple

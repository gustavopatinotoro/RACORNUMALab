//
//  MetalInfo.hpp
//  RACORNUMALab
//
//  Propósito:
//  Exponer hacia el núcleo C++ una caracterización mínima, reproducible
//  y segura del dispositivo Metal disponible en macOS.
//
//  Gate:
//  G-UMA0 — Caracterización Metal / memoria unificada.
//
//  Responsabilidades:
//  - Detectar el dispositivo Metal predeterminado.
//  - Verificar si CPU y GPU comparten memoria física.
//  - Consultar límites de memoria relevantes.
//  - Verificar que puede crearse una MTLCommandQueue.
//  - Verificar que puede asignarse un MTLBuffer Shared.
//  - Mantener todos los tipos Objective-C fuera del resto del sistema.
//
//  Frontera arquitectónica:
//  Este archivo es C++ puro. La interacción con Metal se implementa
//  exclusivamente en MetalInfo.mm mediante Objective-C++.
//
//  Invariantes:
//  - No ejecuta kernels.
//  - No modifica RACORN.
//  - No modifica el índice HNSW.
//  - No crea estado persistente.
//  - No mide rendimiento.
//  - No interpreta recommendedMaxWorkingSetSize como memoria libre.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Sin escrituras persistentes.
//  - Sin punteros Objective-C expuestos al código C++.
//

#pragma once

#include <cstdint>
#include <string>

namespace racorn_uma::platform::apple {

struct MetalDeviceSnapshot final {

    bool metal_available{};

    bool unified_memory{};

    bool command_queue_available{};

    bool shared_buffer_available{};

    bool low_power{};

    bool removable{};

    bool headless{};

    std::string device_name;

    std::uint64_t registry_id{};

    std::uint64_t recommended_max_working_set_size{};

    std::uint64_t current_allocated_size_before_probe{};

    std::uint64_t current_allocated_size_after_probe{};

    std::uint64_t max_buffer_length{};

    std::uint64_t max_threadgroup_memory_length{};

    std::uint64_t max_threads_per_threadgroup_width{};

    std::uint64_t max_threads_per_threadgroup_height{};

    std::uint64_t max_threads_per_threadgroup_depth{};

    std::uint64_t shared_probe_buffer_length{};

    bool valid{};
};

class MetalInfo final {

public:

    [[nodiscard]]
    static MetalDeviceSnapshot capture();
};

} // namespace racorn_uma::platform::apple

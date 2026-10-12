//
//  SystemInfo.hpp
//  RACORNUMALab
//
//  Propósito:
//  Obtener propiedades fundamentales del hardware Apple sobre el cual
//  se ejecuta RACORNUMALab.
//
//  Responsabilidades:
//  - Identificar modelo de hardware.
//  - Identificar arquitectura reportada por Darwin.
//  - Determinar memoria física disponible en el sistema.
//  - Determinar número de CPU lógicas.
//
//  Contrato:
//  Las funciones de esta unidad son exclusivamente de lectura.
//
//  Seguridad:
//  Utiliza sysctl de macOS únicamente para consultar propiedades
//  públicas del sistema operativo.
//

#pragma once

#include <cstdint>
#include <string>

namespace racorn_uma::platform::apple {

struct SystemInfo final {

    [[nodiscard]]
    static std::string hardware_model();

    [[nodiscard]]
    static std::string machine();

    [[nodiscard]]
    static std::uint64_t physical_memory_bytes();

    [[nodiscard]]
    static int logical_cpu_count();
};

} // namespace racorn_uma::platform::apple

//
//  MemoryInfo.hpp
//  RACORNUMALab
//
//  Propósito:
//  Proporcionar telemetría de memoria del proceso RACORNUMALab
//  utilizando las interfaces Mach disponibles en macOS.
//
//  Gate:
//  G3B — Instrumentación de memoria.
//
//  Responsabilidades:
//  - Medir memoria residente actual.
//  - Medir pico de memoria residente.
//  - Medir physical footprint atribuido al proceso.
//  - Medir memoria comprimida reportada por el kernel.
//
//  Contrato:
//  La medición describe el proceso completo.
//  No representa directamente el consumo exclusivo del índice HNSW.
//
//  Invariantes:
//  - Solo lectura.
//  - No modifica el estado del proceso.
//  - No utiliza APIs privadas.
//  - No depende de RACORN ni USEARCH.
//

#pragma once

#include <cstdint>

namespace racorn_uma::platform::apple {

struct ProcessMemorySnapshot final {

    std::uint64_t resident_bytes{};
    std::uint64_t resident_peak_bytes{};
    std::uint64_t physical_footprint_bytes{};
    std::uint64_t compressed_bytes{};

    bool valid{};
};

class MemoryInfo final {

public:

    [[nodiscard]]
    static ProcessMemorySnapshot capture() noexcept;
};

} // namespace racorn_uma::platform::apple

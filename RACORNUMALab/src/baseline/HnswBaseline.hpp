//
//  HnswBaseline.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir un experimento HNSW parametrizable para estudiar el
//  escalado de memoria sobre Apple Silicon.
//
//  Gate:
//  G3C — Escalado controlado de memoria HNSW.
//
//  Variable experimental:
//  - Número de vectores N.
//
//  Variables controladas:
//  - Dimensión.
//  - M.
//  - efConstruction.
//  - efSearch.
//  - Número de hilos.
//  - Métrica.
//  - Semilla lógica del dataset.
//
//  Invariantes:
//  - ACORN desactivado.
//  - RACORN desactivado.
//  - 3-hop desactivado.
//  - Exact fallback desactivado.
//  - Un único hilo.
//

#pragma once

#include "platform/apple/MemoryInfo.hpp"

#include <cstddef>
#include <vector>

namespace racorn_uma::baseline {

struct HnswBaselineResult final {

    std::size_t vector_count{};
    std::size_t dimensions{};
    std::size_t connectivity{};
    std::size_t ef_construction{};
    std::size_t ef_search{};
    std::size_t k{};

    std::vector<std::size_t> exact_keys;
    std::vector<std::size_t> hnsw_keys;

    double recall_at_k{};

    platform::apple::ProcessMemorySnapshot
        memory_start;

    platform::apple::ProcessMemorySnapshot
        memory_after_vectors;

    platform::apple::ProcessMemorySnapshot
        memory_after_reserve;

    platform::apple::ProcessMemorySnapshot
        memory_after_build;

    platform::apple::ProcessMemorySnapshot
        memory_after_search;

    bool passed{};
};

class HnswBaseline final {

public:

    [[nodiscard]]
    static HnswBaselineResult run(
        std::size_t vector_count
    );
};

} // namespace racorn_uma::baseline

//
//  HnswBaseline.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir un experimento HNSW parametrizable para separar empíricamente
//  el coste de almacenamiento vectorial del coste estructural del grafo.
//
//  Gate:
//  G3D — Descomposición memoria VectorStore / HNSW.
//
//  Variables experimentales:
//  - Número de vectores N.
//  - Dimensionalidad D.
//
//  Variables controladas:
//  - M.
//  - efConstruction.
//  - efSearch.
//  - Número de hilos.
//  - Métrica.
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
#include <cstdint>
#include <vector>

namespace racorn_uma::baseline {

struct HnswBaselineResult final {

    std::size_t vector_count{};
    std::size_t dimensions{};
    std::size_t connectivity{};
    std::size_t ef_construction{};
    std::size_t ef_search{};
    std::size_t k{};

    std::uint64_t expected_vector_bytes{};

    std::vector<std::size_t> exact_keys;
    std::vector<std::size_t> hnsw_keys;

    double recall_at_k{};

    platform::apple::ProcessMemorySnapshot memory_start;
    platform::apple::ProcessMemorySnapshot memory_after_vectors;
    platform::apple::ProcessMemorySnapshot memory_after_reserve;
    platform::apple::ProcessMemorySnapshot memory_after_build;
    platform::apple::ProcessMemorySnapshot memory_after_search;

    bool passed{};
};

class HnswBaseline final {

public:

    [[nodiscard]]
    static HnswBaselineResult run(
        std::size_t vector_count,
        std::size_t dimensions
    );
};

} // namespace racorn_uma::baseline

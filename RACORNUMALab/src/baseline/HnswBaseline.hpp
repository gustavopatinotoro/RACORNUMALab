//
//  HnswBaseline.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir el experimento mínimo reproducible del HNSW base incluido
//  en el artefacto RACORN/USEARCH.
//
//  Gate:
//  G3A — Baseline funcional HNSW.
//
//  Responsabilidades:
//  - Construir un índice HNSW pequeño y determinista.
//  - Ejecutar búsqueda aproximada HNSW.
//  - Ejecutar búsqueda exacta como oráculo.
//  - Calcular Recall@k.
//  - Verificar que la ruta HNSW funciona sobre Apple Silicon arm64.
//
//  Invariantes:
//  - No activa ACORN.
//  - No activa RACORN.
//  - No activa 3-hop.
//  - No utiliza Metal.
//  - Utiliza un único hilo.
//  - Los vectores permanecen en memoria durante toda la vida del índice.
//
//  Alcance:
//  Este Gate verifica correctitud funcional, no rendimiento.
//

#pragma once

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

    bool passed{};
};

class HnswBaseline final {

public:

    [[nodiscard]]
    static HnswBaselineResult run();
};

} // namespace racorn_uma::baseline

//
//  FilteredSearchBaseline.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir el primer experimento comparativo de búsqueda vectorial
//  filtrada entre HNSW, ACORN-1 y RACORN-1.
//
//  Gate:
//  G4A — Activación funcional de búsqueda filtrada.
//
//  Objetivos:
//  - Utilizar exactamente el mismo índice HNSW para los tres modos.
//  - Utilizar exactamente las mismas consultas.
//  - Utilizar exactamente el mismo predicado.
//  - Comparar cada resultado contra ground truth exacto filtrado.
//  - Verificar que RACORN activa realmente Adaptive Search Fallback.
//
//  Invariantes:
//  - Un único hilo.
//  - Sin Metal.
//  - Sin 3-hop.
//  - Sin RACORN-1+.
//  - bridge_budget_ratio = 1.0.
//  - Predicado determinista.
//  - Dataset y queries reproducibles.
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace racorn_uma::filtered {

struct FilteredModeStats final {

    std::string name;

    double mean_recall_at_k{};

    double mean_result_count{};

    double mean_computed_distances{};

    double mean_visited_members{};

    double mean_predicate_evaluations{};

    double mean_fallback_count{};

    double mean_fallback_selectivity{};

    std::size_t queries_with_fallback{};

    bool all_results_satisfy_filter{};
};

struct FilteredSearchResult final {

    std::size_t vector_count{};
    std::size_t dimensions{};
    std::size_t connectivity{};
    std::size_t ef_construction{};
    std::size_t ef_search{};
    std::size_t k{};
    std::size_t query_count{};

    std::size_t eligible_count{};

    double nominal_selectivity{};
    double observed_selectivity{};

    FilteredModeStats hnsw;
    FilteredModeStats acorn;
    FilteredModeStats racorn;

    bool exact_ground_truth_complete{};
    bool racorn_fallback_observed{};
    bool passed{};
};

class FilteredSearchBaseline final {

public:

    [[nodiscard]]
    static FilteredSearchResult run();
};

} // namespace racorn_uma::filtered

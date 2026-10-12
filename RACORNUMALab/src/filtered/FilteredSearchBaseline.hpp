//
//  FilteredSearchBaseline.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir el experimento reproducible de búsqueda vectorial filtrada
//  utilizado para estudiar la variabilidad topológica de HNSW, ACORN-1
//  y RACORN-1 bajo una selectividad nominal constante.
//
//  Gate:
//  G4C — Replicación topológica del filtro.
//
//  Pregunta experimental:
//  ¿La selectividad s es suficiente para predecir el comportamiento de
//  ACORN/RACORN, o distintas realizaciones del mismo filtro producen
//  diferencias relevantes debido a su interacción con la topología HNSW?
//
//  Estrategia:
//  Para una selectividad definida por:
//
//      threshold / modulus
//
//  se ejecutan varias realizaciones del predicado desplazando la clase
//  residual:
//
//      shifted(key, residue) < threshold
//
//  Con threshold = 1:
//
//      key % modulus == residue
//
//  Cada realización conserva aproximadamente la misma selectividad,
//  pero selecciona un subconjunto diferente de nodos.
//
//  Variables controladas:
//  - N = 20.000.
//  - D = 32.
//  - M = 16.
//  - efConstruction = 100.
//  - efSearch = 200.
//  - k = 10.
//  - 32 consultas por realización.
//  - Mismo índice HNSW.
//  - Mismas consultas.
//  - Un único hilo.
//  - Sin Metal.
//  - Sin 3-hop.
//  - Sin RACORN-1+.
//  - bridge_budget_ratio = 1.0.
//  - stride sampling activado.
//
//  Métricas:
//  - Recall@k medio.
//  - Recall medio por realización.
//  - Desviación estándar entre realizaciones.
//  - Recall mínimo y máximo entre realizaciones.
//  - Distancias calculadas.
//  - Nodos visitados.
//  - Evaluaciones del predicado.
//  - Fallback RACORN.
//  - Selectividad observada.
//
//  Invariantes:
//  - El índice se construye exactamente una vez por ejecución.
//  - Ningún modo modifica la topología.
//  - Todas las rutas utilizan el mismo índice y las mismas consultas.
//  - El código vendorizado RACORN permanece intacto.
//
//  Seguridad:
//  - Sin red.
//  - Sin entrada de archivos.
//  - Parámetros externos validados.
//  - Gestión de memoria mediante RAII.
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace racorn_uma::filtered {

struct FilteredModeStats final {

    std::string name;

    double mean_recall_at_k{};

    double realization_recall_stddev{};

    double min_realization_recall{};

    double max_realization_recall{};

    double mean_result_count{};

    double mean_computed_distances{};

    double mean_visited_members{};

    double mean_predicate_evaluations{};

    double mean_fallback_count{};

    double mean_fallback_selectivity{};

    std::size_t total_query_evaluations{};

    std::size_t queries_with_fallback{};

    double fallback_query_fraction{};

    bool all_results_satisfy_filter{};
};

struct FilterRealizationStats final {

    std::size_t realization_index{};

    std::size_t residue{};

    std::size_t eligible_count{};

    double observed_selectivity{};

    double hnsw_recall_at_k{};

    double acorn_recall_at_k{};

    double racorn_recall_at_k{};

    double racorn_mean_distances{};

    double racorn_mean_fallback_count{};

    double racorn_fallback_query_fraction{};
};

struct FilteredSearchResult final {

    std::size_t vector_count{};
    std::size_t dimensions{};
    std::size_t connectivity{};
    std::size_t ef_construction{};
    std::size_t ef_search{};
    std::size_t k{};
    std::size_t query_count{};

    std::size_t filter_modulus{};
    std::size_t filter_threshold{};

    std::size_t realization_count{};

    std::size_t min_eligible_count{};
    std::size_t max_eligible_count{};

    double nominal_selectivity{};

    FilteredModeStats hnsw;
    FilteredModeStats acorn;
    FilteredModeStats racorn;

    std::vector<FilterRealizationStats> realizations;

    bool exact_ground_truth_complete{};
    bool filters_valid{};
    bool racorn_fallback_observed{};
    bool passed{};
};

class FilteredSearchBaseline final {

public:

    [[nodiscard]]
    static FilteredSearchResult run(
        std::size_t filter_modulus,
        std::size_t filter_threshold,
        std::size_t realization_count
    );
};

} // namespace racorn_uma::filtered

//
//  BridgeBudgetSweep.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir el experimento de ablación del presupuesto de bridges
//  utilizado por RACORN-1.
//
//  Gate:
//  G4D — Sweep de bridge_budget_ratio.
//
//  Pregunta experimental:
//  ¿Cuánto presupuesto de bridges necesita RACORN-1 para mantener
//  Recall@k bajo filtros de baja selectividad, y cuál es el coste
//  computacional asociado?
//
//  Variable experimental:
//  - bridge_budget_ratio.
//
//  Valores evaluados:
//  - 0.25
//  - 0.50
//  - 0.75
//  - 1.00
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
//  - Mismos filtros.
//  - Un único hilo.
//  - two_hop_threshold = 0.30.
//  - stride sampling activado.
//  - 3-hop desactivado.
//  - RACORN-1+ desactivado.
//  - Sin Metal.
//
//  Controles:
//  - HNSW filtrado.
//  - ACORN-1.
//  - Ground truth exacto filtrado.
//
//  Métricas:
//  - Recall@k.
//  - Desviación entre realizaciones.
//  - Recall mínimo y máximo.
//  - Distancias.
//  - Nodos visitados.
//  - Evaluaciones del predicado.
//  - Número de fallbacks.
//  - Fracción de queries con fallback.
//  - Selectividad observada en fallback.
//
//  Invariantes:
//  - El índice se construye una única vez.
//  - Ningún modo modifica la topología.
//  - RACORN upstream permanece intacto.
//  - No se utiliza latencia como criterio en este Gate.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Parámetros externos validados.
//  - Gestión de recursos mediante RAII.
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace racorn_uma::filtered {

struct BridgeModeStats final {

    std::string name;

    double bridge_budget_ratio{};

    double mean_recall_at_k{};

    double realization_recall_stddev{};

    double min_realization_recall{};

    double max_realization_recall{};

    double mean_result_count{};

    double mean_computed_distances{};

    double mean_visited_members{};

    double mean_predicate_evaluations{};

    double mean_fallback_count{};

    double fallback_query_fraction{};

    double mean_fallback_selectivity{};

    std::size_t total_query_evaluations{};

    std::size_t queries_with_fallback{};

    bool all_results_satisfy_filter{};
};

struct BridgeBudgetResult final {

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

    BridgeModeStats hnsw;
    BridgeModeStats acorn;

    std::vector<BridgeModeStats> racorn_modes;

    bool exact_ground_truth_complete{};
    bool filters_valid{};
    bool passed{};
};

class BridgeBudgetSweep final {

public:

    [[nodiscard]]
    static BridgeBudgetResult run(
        std::size_t filter_modulus,
        std::size_t filter_threshold,
        std::size_t realization_count
    );
};

} // namespace racorn_uma::filtered

//
//  FilteredSearchBaseline.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir el experimento reproducible de búsqueda vectorial filtrada
//  utilizado para comparar HNSW, ACORN-1 y RACORN-1 sobre exactamente
//  el mismo índice, consultas y predicado.
//
//  Gate:
//  G4B — Barrido controlado de selectividad.
//
//  Objetivos:
//  - Parametrizar la selectividad del filtro.
//  - Mantener constante la topología HNSW.
//  - Comparar HNSW filtrado, ACORN-1 y RACORN-1.
//  - Utilizar búsqueda exacta filtrada como ground truth.
//  - Medir Recall@k.
//  - Medir distancias calculadas.
//  - Medir nodos visitados.
//  - Medir evaluaciones del predicado.
//  - Medir activación del fallback RACORN.
//  - Verificar que todos los resultados respetan el filtro.
//
//  Diseño experimental:
//  Cada configuración de selectividad debe ejecutarse en un proceso
//  independiente. Esto evita contaminación entre experimentos y mantiene
//  una frontera metodológica clara.
//
//  Invariantes:
//  - N = 20.000.
//  - D = 32.
//  - M = 16.
//  - efConstruction = 100.
//  - efSearch = 200.
//  - k = 10.
//  - 32 consultas.
//  - Un único hilo.
//  - Sin Metal.
//  - Sin 3-hop.
//  - Sin RACORN-1+.
//  - bridge_budget_ratio = 1.0.
//  - Stride sampling activado.
//  - Dataset determinista.
//  - Consultas deterministas.
//
//  Seguridad:
//  - No existe entrada de red.
//  - No se modifican archivos.
//  - No se modifica código RACORN vendorizado.
//  - Los parámetros externos son validados por el punto de entrada.
//

#pragma once

#include <cstddef>
#include <string>

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

    std::size_t filter_modulus{};
    std::size_t filter_threshold{};

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
    static FilteredSearchResult run(
        std::size_t filter_modulus,
        std::size_t filter_threshold
    );
};

} // namespace racorn_uma::filtered

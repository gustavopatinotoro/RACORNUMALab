//
//  RacornSiftValidation.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir G-DATA1, primera validación de RACORN sobre geometría
//  vectorial real SIFT.
//
//  Dataset:
//      SIFT1M-base[0:100000]
//      D = 128
//      métrica = L2
//      queries reales SIFT = 1000
//
//  Metadata:
//      Metadata determinista validada previamente por G-DATA0.
//      La metadata permanece independiente de la geometría SIFT.
//
//  Objetivo:
//      Determinar cómo cambia la frontera de bridge_budget_ratio al
//      sustituir la geometría sintética por una geometría vectorial real.
//
//  Sweep:
//      BR ∈ {0.50, 0.65, 0.75, 1.00, 1.25, 1.50, 2.00}
//
//  Restricción robusta:
//      mean Recall@10 >= 0.99
//      AND
//      min Recall@10 >= 0.99
//
//  Nota:
//      Con k=10, min Recall@10 >= 0.99 implica Recall@10 = 1.0 para
//      cada query individual.
//
//  El Gate experimental se considera válido aunque ningún BR satisfaga
//  la restricción de calidad. No encontrar un punto robusto es un
//  resultado científico legítimo.
//
//  Invariantes:
//  - RACORN upstream no se modifica.
//  - Navegación exclusivamente CPU.
//  - Sin Metal.
//  - Sin GPU.
//  - Sin bitset.
//  - Ground truth filtrado exacto independiente.
//

#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace racorn_uma::core {

struct SiftValidationCase final {

    double bridge_budget_ratio{};

    double mean_recall_at_k{};
    double min_recall_at_k{};

    double recall_p01{};
    double recall_p05{};

    double perfect_query_fraction{};

    std::size_t perfect_query_count{};
    std::size_t imperfect_query_count{};
    std::size_t worst_query_index{};

    double mean_computed_distances{};
    double mean_visited_members{};
    double mean_predicate_evaluations{};
    double mean_fallback_count{};

    std::size_t result_count_min{};
    std::size_t result_count_max{};

    bool mean_recall_pass{};
    bool min_recall_pass{};
    bool robust_quality_pass{};

    bool ground_truth_complete{};
    bool returned_results_filter_valid{};

    bool valid{};
};

struct SiftRobustPoint final {

    bool found{};

    double bridge_budget_ratio{};

    double mean_recall_at_k{};
    double min_recall_at_k{};

    double recall_p01{};

    double mean_computed_distances{};
    double mean_predicate_evaluations{};
};

struct RacornSiftValidationResult final {

    std::filesystem::path data_directory;

    std::size_t vector_count{};
    std::size_t dimensions{};
    std::size_t query_count{};

    std::size_t connectivity{};
    std::size_t ef_construction{};
    std::size_t ef_search{};

    std::size_t k{};

    double two_hop_threshold{};

    double mean_recall_target{};
    double min_recall_target{};

    double mean_selectivity{};

    std::size_t selected_min{};
    std::size_t selected_max{};

    bool base_file_valid{};
    bool query_file_valid{};
    bool metadata_file_valid{};

    bool ground_truth_complete{};

    std::vector<SiftValidationCase> cases;

    SiftRobustPoint minimum_robust_point;

    bool all_cases_valid{};

    bool racorn_modified{};

    bool valid{};
};

class RacornSiftValidation final {

public:

    [[nodiscard]]
    static RacornSiftValidationResult run(
        std::filesystem::path const& data_directory
    );
};

} // namespace racorn_uma::core

//
//  RacornSiftTailAudit.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir G-DATA1A, auditoría de la cola de errores observada durante
//  G-DATA1 sobre SIFT100K.
//
//  Objetivo:
//  Separar tres fenómenos distintos:
//
//  1. Query perfecta:
//         result_count == k
//         Recall@k == 1.0
//
//  2. Underfill:
//         result_count < k
//
//  3. Resultado completo pero topológicamente incorrecto:
//         result_count == k
//         Recall@k < 1.0
//
//  Principio metodológico:
//  Un underfill es un fallo de calidad del algoritmo, no una invalidez
//  experimental.
//
//  El experimento solamente se considera inválido si:
//  - el ground truth no puede construirse;
//  - algún resultado viola el predicado;
//  - RACORN devuelve más de k resultados;
//  - los archivos de entrada son inconsistentes.
//
//  Invariantes:
//  - RACORN upstream intacto.
//  - CPU solamente.
//  - Sin Metal.
//  - Sin GPU.
//  - Sin bitsets.
//  - Ground truth exacto filtrado.
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace racorn_uma::core {

struct SiftTailQueryRecord final {

    double bridge_budget_ratio{};

    std::size_t query_index{};

    std::uint32_t tenant_required{};

    std::size_t selected_count{};
    std::size_t result_count{};

    double recall_at_k{};

    std::size_t hits{};
    std::size_t missing_hits{};

    std::size_t computed_distances{};
    std::size_t visited_members{};
    std::size_t predicate_evaluations{};
    std::size_t fallback_count{};

    bool underfilled{};
    bool empty{};
    bool full_but_inexact{};
};

struct SiftTailAuditCase final {

    double bridge_budget_ratio{};

    double mean_recall_at_k{};
    double min_recall_at_k{};

    double recall_p01{};
    double recall_p05{};

    std::size_t worst_query_index{};

    std::size_t perfect_queries{};
    std::size_t imperfect_queries{};

    std::size_t underfilled_queries{};
    std::size_t empty_queries{};

    std::size_t full_but_inexact_queries{};

    std::size_t result_count_min{};
    std::size_t result_count_max{};

    std::array<std::size_t, 11>
        recall_histogram{};

    std::array<std::size_t, 11>
        result_count_histogram{};

    double mean_computed_distances{};
    double mean_visited_members{};
    double mean_predicate_evaluations{};
    double mean_fallback_count{};

    bool ground_truth_complete{};
    bool returned_results_filter_valid{};
    bool result_cardinality_valid{};

    bool robust_quality_pass{};

    bool valid{};

    std::vector<SiftTailQueryRecord>
        imperfect_records;
};

struct RacornSiftTailAuditResult final {

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

    std::vector<SiftTailAuditCase> cases;

    bool all_cases_valid{};

    bool racorn_modified{};

    bool valid{};
};

class RacornSiftTailAudit final {

public:

    [[nodiscard]]
    static RacornSiftTailAuditResult run(
        std::filesystem::path const& data_directory
    );
};

} // namespace racorn_uma::core

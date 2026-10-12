//
//  RacornSiftTargetingAudit.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir G-DATA2A — Exact-Fallback Targeting Audit.
//
//  Pregunta científica:
//  ¿La señal utilizada por RACORN-1+ identifica correctamente las queries
//  que RACORN-1 no logra resolver con Recall@10 perfecto?
//
//  Diseño:
//
//      BR fijo = 1.0
//
//      baseline:
//          RACORN-1
//          fallback_exact_threshold = 0
//
//      sweep:
//          tau ∈ {
//              0.0010,
//              0.0011,
//              0.0012,
//              0.0013,
//              0.0014,
//              0.0015,
//              0.0016,
//              0.0017,
//              0.0018,
//              0.0019,
//              0.0020
//          }
//
//  Etiqueta verdadera:
//
//      hard(q) = RACORN-1 no devuelve exactamente el top-k correcto.
//
//  Predicción:
//
//      predicted_hard(q, tau) = fallback_to_exact.
//
//  Matriz de confusión:
//
//      TP = hard && fallback_to_exact
//      FP = easy && fallback_to_exact
//      FN = hard && !fallback_to_exact
//      TN = easy && !fallback_to_exact
//
//  Métricas:
//  - Precision.
//  - Coverage/Recall del detector.
//  - Specificity.
//  - F1.
//  - Balanced accuracy.
//  - Recall@10 final.
//  - Exact fallback fraction.
//  - Queries difíciles rescatadas.
//  - Queries fáciles degradadas.
//  - Coste algorítmico.
//
//  Importante:
//  La etiqueta hard/easy se calcula una sola vez con RACORN-1 sobre el
//  mismo índice y las mismas queries.
//
//  Invariantes:
//  - RACORN upstream intacto.
//  - CPU solamente.
//  - Sin Metal.
//  - Sin GPU.
//  - Sin bitsets.
//  - Sin entrenamiento de modelos.
//  - Ground truth exacto filtrado.
//

#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace racorn_uma::core {

struct TargetingDistributionStats final {

    std::size_t samples{};

    double minimum{};
    double p05{};
    double median{};
    double mean{};
    double p95{};
    double maximum{};
};

struct ExactFallbackTargetingCase final {

    double threshold{};

    std::size_t true_positive{};
    std::size_t false_positive{};
    std::size_t false_negative{};
    std::size_t true_negative{};

    double precision{};
    double coverage{};
    double specificity{};
    double f1{};
    double balanced_accuracy{};

    std::size_t exact_fallback_queries{};
    double exact_fallback_fraction{};

    std::size_t rescued_hard_queries{};
    std::size_t remaining_hard_queries{};
    std::size_t newly_imperfect_easy_queries{};

    std::size_t perfect_queries{};
    std::size_t imperfect_queries{};
    std::size_t underfilled_queries{};

    double mean_recall_at_k{};
    double min_recall_at_k{};
    double recall_p01{};
    double recall_p05{};

    double mean_computed_distances{};
    double mean_visited_members{};
    double mean_predicate_evaluations{};
    double mean_fallback_count{};

    std::size_t fallback_selectivity_samples{};
    double mean_fallback_selectivity{};

    bool robust_quality_pass{};

    bool returned_results_filter_valid{};
    bool cardinality_valid{};
    bool valid{};

    std::vector<std::size_t> missed_hard_query_ids;
    std::vector<std::size_t> false_positive_query_ids;
};

struct RacornSiftTargetingAuditResult final {

    std::filesystem::path data_directory;

    std::size_t vector_count{};
    std::size_t dimensions{};
    std::size_t query_count{};

    std::size_t connectivity{};
    std::size_t ef_construction{};
    std::size_t ef_search{};
    std::size_t k{};

    double bridge_budget_ratio{};
    double two_hop_threshold{};

    double mean_recall_target{};
    double min_recall_target{};

    double mean_selectivity{};

    std::size_t selected_min{};
    std::size_t selected_max{};

    double baseline_mean_recall{};
    double baseline_min_recall{};

    std::size_t baseline_perfect_queries{};
    std::size_t baseline_hard_queries{};

    std::vector<std::size_t>
        baseline_hard_query_ids;

    TargetingDistributionStats
        baseline_hard_fallback_selectivity;

    TargetingDistributionStats
        baseline_easy_fallback_selectivity;

    bool base_file_valid{};
    bool query_file_valid{};
    bool metadata_file_valid{};
    bool ground_truth_complete{};
    bool baseline_valid{};

    std::vector<ExactFallbackTargetingCase>
        cases;

    bool all_cases_valid{};

    bool best_f1_found{};
    double best_f1_threshold{};
    double best_f1{};

    bool racorn_modified{};

    bool valid{};
};

class RacornSiftTargetingAudit final {

public:

    [[nodiscard]]
    static RacornSiftTargetingAuditResult run(
        std::filesystem::path const& data_directory
    );
};

} // namespace racorn_uma::core

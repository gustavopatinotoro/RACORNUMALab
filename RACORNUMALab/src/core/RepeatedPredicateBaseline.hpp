//
//  RepeatedPredicateBaseline.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir G-REP0 — Repeated Predicate Sanity Baseline.
//
//  Pregunta:
//  Si el mismo predicado P se reutiliza en muchas consultas vectoriales,
//  ¿sigue teniendo sentido pagar reparación topológica transitoria RACORN
//  en cada consulta, o resulta más barato materializar una vez los IDs
//  elegibles y ejecutar búsqueda exacta únicamente sobre ellos?
//
//  Estrategias:
//
//  A) RACORN-Cached
//      - P se evalúa una sola vez.
//      - Se materializa un bitset compacto.
//      - El mismo bitset se reutiliza en todas las queries.
//      - RACORN sigue realizando reparación transitoria por query.
//
//  B) Exact-Materialized
//      - P se evalúa una sola vez.
//      - Se materializa la lista de IDs elegibles.
//      - Cada query calcula distancia únicamente contra esos IDs.
//      - Recall@k exacto por construcción.
//
//  Predicado fijo:
//
//      tenant == 7
//      year >= 2022
//      language >= 2
//      category IN {3,7,9}
//      security_level <= 2
//
//  Importante:
//  El HNSW se construye una sola vez y queda fuera del coste amortizado.
//
//  Criterio de este Gate:
//  No seleccionar todavía una arquitectura persistente.
//  Determinar primero si Exact-Materialized ya domina en el régimen actual.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>


namespace racorn_uma::core {

struct RepeatedPredicateStrategyStats final {

    double mean_ms{};
    double p50_ms{};
    double p95_ms{};
    double p99_ms{};

    double qps_from_mean{};
};

struct RepeatedPredicateCostPoint final {

    std::size_t repetitions{};

    double racorn_cumulative_ms{};
    double exact_cumulative_ms{};

    std::string winner;
};

struct RepeatedPredicateBaselineResult final {

    std::filesystem::path data_directory;

    std::size_t vector_count{};
    std::size_t dimensions{};
    std::size_t query_count{};
    std::size_t k{};

    std::size_t connectivity{};
    std::size_t ef_construction{};
    std::size_t ef_search{};

    double bridge_budget_ratio{};
    double two_hop_threshold{};

    std::uint32_t tenant_required{};
    std::uint32_t language_min{};

    std::size_t eligible_count{};
    double selectivity{};

    std::size_t bitset_logical_bytes{};
    std::size_t eligible_logical_bytes{};
    std::size_t eligible_allocated_bytes{};

    std::size_t materialization_samples{};

    double racorn_materialization_mean_ms{};
    double racorn_materialization_p50_ms{};

    double exact_materialization_mean_ms{};
    double exact_materialization_p50_ms{};

    RepeatedPredicateStrategyStats racorn;
    RepeatedPredicateStrategyStats exact;

    double mean_recall_at_k{};
    double min_recall_at_k{};

    std::size_t perfect_queries{};
    std::size_t imperfect_queries{};
    std::size_t underfilled_queries{};

    double mean_racorn_computed_distances{};
    double mean_racorn_visited_members{};
    double mean_racorn_predicate_evaluations{};
    double mean_racorn_fallback_count{};

    double exact_distance_evaluations_per_query{};

    std::size_t representation_mismatches{};
    std::size_t exact_reference_mismatch_queries{};
    std::size_t invalid_racorn_result_queries{};

    bool base_file_valid{};
    bool query_file_valid{};
    bool metadata_file_valid{};

    bool ground_truth_complete{};
    bool exact_implementation_valid{};
    bool predicate_materialization_valid{};
    bool racorn_results_filter_valid{};

    bool exact_query_faster{};
    bool exact_build_cheaper{};

    bool exact_break_even_exists{};
    double exact_break_even_repetitions{};

    double exact_speedup_vs_racorn{};

    std::vector<RepeatedPredicateCostPoint>
        cumulative_costs;

    std::size_t warmup_rounds{};
    std::size_t measured_rounds{};

    std::size_t benchmark_sink{};

    bool racorn_modified{};
    bool valid{};
};

class RepeatedPredicateBaseline final {

public:

    [[nodiscard]]
    static RepeatedPredicateBaselineResult run(
        std::filesystem::path const& data_directory
    );
};

} // namespace racorn_uma::core

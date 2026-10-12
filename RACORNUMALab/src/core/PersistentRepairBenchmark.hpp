//
//  PersistentRepairBenchmark.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir G-REPAIR0.
//
//  Comparación científica:
//
//      A) RACORN-Transient
//         La reparación topológica se recalcula durante cada consulta.
//
//      B) Persistent-Repair
//         El predicado se materializa una vez.
//         Se construye una reparación topológica externa una sola vez.
//         Todas las consultas posteriores reutilizan esa reparación.
//
//  Hipótesis:
//
//      Si el mismo predicado P se reutiliza R veces y:
//
//          L_P < L_T
//
//      existe un punto de equilibrio:
//
//          R* = B_P / (L_T - L_P)
//
//      donde:
//
//          B_P = coste one-shot de construir la reparación persistente.
//          L_T = coste/query de RACORN transitorio.
//          L_P = coste/query del overlay persistente.
//
//  Invariantes:
//
//  - RACORN upstream permanece intacto.
//  - HNSW base permanece intacto.
//  - El overlay almacena únicamente topología.
//  - Los vectores nunca se duplican.
//  - Sin Metal.
//  - Sin RACORN-1+.
//  - El mismo predicado se usa para todas las queries.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace racorn_uma::core {

struct RepairLatencyStats final {

    double mean_ms{};
    double p50_ms{};
    double p95_ms{};
    double p99_ms{};

    double qps_mean{};

    double mean_computed_distances{};
    double mean_visited_members{};
};

struct RepairCostPoint final {

    std::size_t repetitions{};

    double transient_cumulative_ms{};
    double persistent_cumulative_ms{};

    std::string winner;
};

struct PersistentRepairBenchmarkResult final {

    std::filesystem::path data_directory;

    std::size_t vector_count{};
    std::size_t dimensions{};
    std::size_t query_count{};
    std::size_t k{};

    std::size_t connectivity{};
    std::size_t connectivity_base{};
    std::size_t ef_construction{};
    std::size_t ef_search{};
    std::size_t persistent_ef_search{};

    double bridge_budget_ratio{};
    double two_hop_threshold{};

    std::uint32_t tenant_required{};
    std::uint32_t language_min{};

    std::size_t eligible_count{};
    double selectivity{};

    std::size_t predicate_bitset_bytes{};

    //
    // Snapshot externo del HNSW.
    //

    std::size_t serialized_index_bytes{};

    std::size_t parsed_l0_directed_edges{};
    std::size_t index_l0_directed_edges{};

    double topology_snapshot_ms{};

    bool topology_snapshot_valid{};

    //
    // Reparación persistente.
    //

    std::size_t shortcut_proposal_budget{};

    std::size_t induced_undirected_edges{};
    std::size_t two_hop_shortcut_edges{};
    std::size_t completion_edges{};
    std::size_t final_undirected_edges{};

    std::size_t components_induced{};
    std::size_t components_after_two_hop{};
    std::size_t components_final{};

    std::size_t overlay_logical_bytes{};
    std::size_t overlay_allocated_bytes{};

    double repair_build_ms{};
    double total_persistent_build_ms{};

    //
    // Calidad.
    //

    double transient_mean_recall{};
    double transient_min_recall{};

    double persistent_mean_recall{};
    double persistent_min_recall{};

    std::size_t transient_perfect_queries{};
    std::size_t persistent_perfect_queries{};

    std::size_t transient_underfilled_queries{};
    std::size_t persistent_underfilled_queries{};

    //
    // Trabajo RACORN específico.
    //

    double transient_mean_predicate_evaluations{};
    double transient_mean_fallback_count{};

    //
    // Latencia.
    //

    RepairLatencyStats transient;
    RepairLatencyStats persistent;

    //
    // Amortización.
    //

    bool persistent_quality_admissible{};
    bool persistent_query_faster{};

    bool break_even_exists{};

    double break_even_total_repetitions{};
    double break_even_repair_only_repetitions{};

    double persistent_speedup_vs_transient{};

    std::vector<RepairCostPoint>
        cumulative_costs;

    //
    // Validez.
    //

    std::size_t invalid_transient_results{};
    std::size_t invalid_persistent_results{};

    bool base_file_valid{};
    bool query_file_valid{};
    bool metadata_file_valid{};

    bool ground_truth_complete{};
    bool overlay_connected{};

    std::string decision;

    std::size_t warmup_rounds{};
    std::size_t measured_rounds{};

    std::size_t benchmark_sink{};

    bool racorn_modified{};
    bool valid{};
};

class PersistentRepairBenchmark final {

public:

    [[nodiscard]]
    static PersistentRepairBenchmarkResult run(
        std::filesystem::path const& data_directory
    );
};

} // namespace racorn_uma::core

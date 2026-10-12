//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar la primera integración semántica entre RACORN CPU y un
//  bitset de elegibilidad generado por GPU Metal sobre memoria UMA.
//
//  Gate:
//  G-UMA4 — RACORN CPU consumiendo bitset GPU/UMA.
//
//  Hipótesis inmediata:
//  Sustituir la evaluación repetida del predicado por un lookup O(1)
//  sobre un bitset semánticamente equivalente no debe modificar la
//  navegación RACORN.
//
//  El Gate NO mide rendimiento.
//
//  PASS exige identidad de:
//  - Top-k.
//  - Recall.
//  - computed_distances.
//  - visited_members.
//  - predicate_evaluations.
//  - fallback_count.
//
//  RACORN permanece inalterado.
//

#include "core/BuildInfo.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/RacornUmaSemanticProbe.hpp"
#include "platform/apple/SystemInfo.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>

namespace {

double bytes_to_kib(
    std::uint64_t bytes
) noexcept {

    return
        static_cast<double>(
            bytes
        ) /
        1024.0;
}

char const* yes_no(
    bool value
) noexcept {

    return
        value
            ? "SI"
            : "NO";
}

} // namespace

int main() {

    using racorn_uma::core::
        BuildInfo;

    using racorn_uma::integration::
        RacornProbe;

    using racorn_uma::platform::apple::
        RacornUmaSemanticProbe;

    using racorn_uma::platform::apple::
        SystemInfo;

    const auto result =
        RacornUmaSemanticProbe::run();

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G-UMA4 - RACORN + GPU Bitset\n"
        << "========================================\n\n";

    std::cout
        << "[Plataforma]\n";

    std::cout
        << "Compilador                 : "
        << BuildInfo::compiler()
        << '\n';

    std::cout
        << "Estandar C++               : "
        << BuildInfo::cpp_standard()
        << '\n';

    std::cout
        << "Arquitectura               : "
        << BuildInfo::architecture()
        << '\n';

    std::cout
        << "Modelo hardware            : "
        << SystemInfo::hardware_model()
        << "\n\n";

    std::cout
        << "[RACORN congelado]\n";

    std::cout
        << "Repositorio                 : "
        << RacornProbe::repository()
        << '\n';

    std::cout
        << "Version                     : "
        << RacornProbe::version()
        << '\n';

    std::cout
        << "Commit                      : "
        << RacornProbe::commit()
        << "\n\n";

    std::cout
        << "[Configuracion RACORN]\n";

    std::cout
        << "N                           : "
        << result.vector_count
        << '\n';

    std::cout
        << "D                           : "
        << result.dimensions
        << '\n';

    std::cout
        << "M                           : "
        << result.connectivity
        << '\n';

    std::cout
        << "efConstruction              : "
        << result.ef_construction
        << '\n';

    std::cout
        << "efSearch                    : "
        << result.ef_search
        << '\n';

    std::cout
        << "Queries                     : "
        << result.query_count
        << '\n';

    std::cout
        << "k                           : "
        << result.k
        << '\n';

    std::cout
        << "two_hop_threshold           : "
        << std::fixed
        << std::setprecision(2)
        << result.two_hop_threshold
        << '\n';

    std::cout
        << "bridge_budget_ratio         : "
        << std::fixed
        << std::setprecision(2)
        << result.bridge_budget_ratio
        << "\n\n";

    std::cout
        << "[GPU Predicate Engine]\n";

    std::cout
        << "Metal disponible            : "
        << yes_no(
            result.metal_available
        )
        << '\n';

    std::cout
        << "Dispositivo                 : "
        << result.device_name
        << '\n';

    std::cout
        << "Memoria unificada           : "
        << yes_no(
            result.unified_memory
        )
        << '\n';

    std::cout
        << "Biblioteca Metal            : "
        << yes_no(
            result.library_available
        )
        << '\n';

    std::cout
        << "Kernel                      : "
        << result.kernel_name
        << '\n';

    std::cout
        << "Kernel encontrado           : "
        << yes_no(
            result.kernel_available
        )
        << '\n';

    std::cout
        << "Pipeline                    : "
        << yes_no(
            result.pipeline_available
        )
        << '\n';

    std::cout
        << "Metadata Shared             : "
        << yes_no(
            result.metadata_buffer_available
        )
        << '\n';

    std::cout
        << "Bitset Shared               : "
        << yes_no(
            result.bitset_buffer_available
        )
        << '\n';

    std::cout
        << "GPU completada              : "
        << yes_no(
            result.command_completed
        )
        << "\n\n";

    std::cout
        << "[Representacion UMA]\n";

    std::cout
        << "Columnas metadata           : "
        << result.metadata_column_count
        << '\n';

    std::cout
        << "Metadata total              : "
        << std::fixed
        << std::setprecision(2)
        << bytes_to_kib(
            result.metadata_bytes
        )
        << " KiB\n";

    std::cout
        << "Palabras bitset             : "
        << result.bitset_word_count
        << '\n';

    std::cout
        << "Bitset                      : "
        << std::fixed
        << std::setprecision(2)
        << bytes_to_kib(
            result.bitset_bytes
        )
        << " KiB\n";

    std::cout
        << "Seleccionados CPU           : "
        << result.cpu_selected_count
        << '\n';

    std::cout
        << "Seleccionados GPU           : "
        << result.gpu_selected_count
        << '\n';

    std::cout
        << "Selectividad observada      : "
        << std::fixed
        << std::setprecision(6)
        << (
            result.observed_selectivity *
            100.0
        )
        << " %\n";

    std::cout
        << "Diferencias predicate/bitset: "
        << result.predicate_bit_mismatch_count
        << '\n';

    std::cout
        << "Predicate exacto            : "
        << yes_no(
            result.predicate_exact_match
        )
        << "\n\n";

    std::cout
        << "[RACORN - Predicate CPU]\n";

    std::cout
        << "Recall@"
        << result.k
        << " medio             : "
        << std::fixed
        << std::setprecision(4)
        << result.cpu_mean_recall_at_k
        << '\n';

    std::cout
        << "Distancias medias           : "
        << std::fixed
        << std::setprecision(4)
        << result.cpu_mean_computed_distances
        << '\n';

    std::cout
        << "Visitados medios            : "
        << result.cpu_mean_visited_members
        << '\n';

    std::cout
        << "Predicados medios           : "
        << result.cpu_mean_predicate_evaluations
        << '\n';

    std::cout
        << "Fallback medio              : "
        << result.cpu_mean_fallback_count
        << '\n';

    std::cout
        << "Filtro valido               : "
        << yes_no(
            result.cpu_results_filter_valid
        )
        << "\n\n";

    std::cout
        << "[RACORN - GPU Bitset Shared]\n";

    std::cout
        << "Recall@"
        << result.k
        << " medio             : "
        << std::fixed
        << std::setprecision(4)
        << result.gpu_bitset_mean_recall_at_k
        << '\n';

    std::cout
        << "Distancias medias           : "
        << std::fixed
        << std::setprecision(4)
        << result.gpu_mean_computed_distances
        << '\n';

    std::cout
        << "Visitados medios            : "
        << result.gpu_mean_visited_members
        << '\n';

    std::cout
        << "Predicados medios           : "
        << result.gpu_mean_predicate_evaluations
        << '\n';

    std::cout
        << "Fallback medio              : "
        << result.gpu_mean_fallback_count
        << '\n';

    std::cout
        << "Filtro valido               : "
        << yes_no(
            result.gpu_results_filter_valid
        )
        << "\n\n";

    std::cout
        << "[Equivalencia de navegacion]\n";

    std::cout
        << "Ground truth completo       : "
        << yes_no(
            result.exact_ground_truth_complete
        )
        << '\n';

    std::cout
        << "Top-k diferentes           : "
        << result.topk_mismatch_queries
        << " queries\n";

    std::cout
        << "Recall diferente           : "
        << result.recall_mismatch_queries
        << " queries\n";

    std::cout
        << "Distancias diferentes      : "
        << result
            .computed_distances_mismatch_queries
        << " queries\n";

    std::cout
        << "Visitados diferentes       : "
        << result
            .visited_members_mismatch_queries
        << " queries\n";

    std::cout
        << "Pred-evals diferentes      : "
        << result
            .predicate_evaluations_mismatch_queries
        << " queries\n";

    std::cout
        << "Fallback diferente         : "
        << result
            .fallback_count_mismatch_queries
        << " queries\n";

    std::cout
        << "Semantica identica         : "
        << yes_no(
            result.navigation_semantics_identical
        )
        << "\n\n";

    if (
        !result.metal_error.empty()
    ) {

        std::cout
            << "[Error Metal]\n"
            << result.metal_error
            << "\n\n";
    }

    std::cout
        << "[Invariantes]\n";

    std::cout
        << "RACORN modificado           : "
        << yes_no(
            result.racorn_modified
        )
        << '\n';

    std::cout
        << "Navegacion RACORN           : CPU\n";

    std::cout
        << "Evaluacion metadata masiva  : GPU\n";

    std::cout
        << "Bitset consumido por        : CPU\n";

    std::cout
        << "Navegacion GPU              : NO\n";

    std::cout
        << "Heaps GPU                   : NO\n";

    std::cout
        << "Bridges GPU                 : NO\n";

    std::cout
        << "Copia explicita a VRAM      : NO\n\n";

    if (
        !result.valid
    ) {

        std::cerr
            << "[G-UMA4] FAIL\n";

        return 1;
    }

    std::cout
        << "[G-UMA4] PASS\n";

    return 0;
}

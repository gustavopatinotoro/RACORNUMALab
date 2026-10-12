//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar G-DATA2.
//
//  Comparación:
//      RACORN-1 vs RACORN-1+ Adaptive Exact Fallback.
//
//  Directorio SIFT:
//      --data-dir <ruta>
//
//  o:
//      RACORNUMA_SIFT_DIR=<ruta>
//
//  Ejecutar en Release.
//

#include "core/BuildInfo.hpp"
#include "core/RacornSiftExactFallback.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/SystemInfo.hpp"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

char const* yes_no(
    bool value
) noexcept {

    return
        value
            ? "SI"
            : "NO";
}

std::filesystem::path resolve_data_directory(
    int argc,
    char* argv[]
) {

    for (
        int index = 1;
        index < argc;
        ++index
    ) {

        const std::string argument =
            argv[index];

        if (
            argument ==
            "--data-dir"
        ) {

            if (
                index +
                    1 >=
                argc
            ) {

                throw std::runtime_error(
                    "--data-dir requiere una ruta."
                );
            }

            return
                std::filesystem::path(
                    argv[
                        index +
                            1
                    ]
                );
        }
    }

    if (
        char const* environment =
            std::getenv(
                "RACORNUMA_SIFT_DIR"
            );
        environment != nullptr &&
        environment[0] != '\0'
    ) {

        return
            std::filesystem::path(
                environment
            );
    }

    throw std::runtime_error(
        "Especifique --data-dir <ruta> "
        "o RACORNUMA_SIFT_DIR."
    );
}

} // namespace

int main(
    int argc,
    char* argv[]
) {

    using racorn_uma::core::
        BuildInfo;

    using racorn_uma::core::
        RacornSiftExactFallback;

    using racorn_uma::integration::
        RacornProbe;

    using racorn_uma::platform::apple::
        SystemInfo;

    try {

        const auto data_directory =
            resolve_data_directory(
                argc,
                argv
            );

        const auto result =
            RacornSiftExactFallback::run(
                data_directory
            );

        std::cout
            << "\n========================================\n"
            << " RACORNUMALab\n"
            << " Gate G-DATA2 - RACORN-1 vs RACORN-1+\n"
            << "========================================\n\n";

        std::cout
            << "[Plataforma]\n"
            << "Compilador                 : "
            << BuildInfo::compiler()
            << '\n'
            << "Estandar C++               : "
            << BuildInfo::cpp_standard()
            << '\n'
            << "Arquitectura               : "
            << BuildInfo::architecture()
            << '\n'
            << "Modelo hardware            : "
            << SystemInfo::hardware_model()
            << "\n\n";

        std::cout
            << "[RACORN congelado]\n"
            << "Repositorio                 : "
            << RacornProbe::repository()
            << '\n'
            << "Version                     : "
            << RacornProbe::version()
            << '\n'
            << "Commit                      : "
            << RacornProbe::commit()
            << "\n\n";

        std::cout
            << "[Dataset]\n"
            << "Dataset                     : SIFT1M\n"
            << "Subset                      : train[0:100000]\n"
            << "N                           : "
            << result.vector_count
            << '\n'
            << "D                           : "
            << result.dimensions
            << '\n'
            << "Queries                     : "
            << result.query_count
            << '\n'
            << "k                           : "
            << result.k
            << '\n'
            << "Selectividad media          : "
            << std::fixed
            << std::setprecision(4)
            << (
                result.mean_selectivity *
                100.0
            )
            << "%\n"
            << "Elegibles/query             : "
            << result.selected_min
            << ".."
            << result.selected_max
            << "\n\n";

        std::cout
            << "[Configuracion fija]\n"
            << "M                           : "
            << result.connectivity
            << '\n'
            << "efConstruction              : "
            << result.ef_construction
            << '\n'
            << "efSearch                    : "
            << result.ef_search
            << '\n'
            << "two_hop_threshold           : "
            << std::fixed
            << std::setprecision(2)
            << result.two_hop_threshold
            << '\n'
            << "Mean Recall target          : "
            << std::setprecision(4)
            << result.mean_recall_target
            << '\n'
            << "Min Recall target           : "
            << result.min_recall_target
            << "\n\n";

        std::cout
            << "[Matriz RACORN-1 / RACORN-1+]\n\n";

        std::cout
            << std::left
            << std::setw(7)
            << "BR"

            << std::setw(9)
            << "Tau"

            << std::setw(9)
            << "Modo"

            << std::right
            << std::setw(10)
            << "Recall"

            << std::setw(10)
            << "MinRec"

            << std::setw(9)
            << "Perfect"

            << std::setw(9)
            << "Under"

            << std::setw(11)
            << "FullWrong"

            << std::setw(10)
            << "Exact"

            << std::setw(10)
            << "Exact%"

            << std::setw(10)
            << "Robust"

            << '\n';

        for (
            auto const& item :
            result.cases
        ) {

            std::cout
                << std::left
                << std::fixed
                << std::setprecision(2)
                << std::setw(7)
                << item.bridge_budget_ratio

                << std::setprecision(3)
                << std::setw(9)
                << item.fallback_exact_threshold

                << std::setw(9)
                << (
                    item.racorn_one_plus
                        ? "R-1+"
                        : "R-1"
                )

                << std::right
                << std::setprecision(4)

                << std::setw(10)
                << item.mean_recall_at_k

                << std::setw(10)
                << item.min_recall_at_k

                << std::setw(9)
                << item.perfect_queries

                << std::setw(9)
                << item.underfilled_queries

                << std::setw(11)
                << item.full_but_inexact_queries

                << std::setw(10)
                << item.exact_fallback_queries

                << std::setw(9)
                << std::setprecision(2)
                << (
                    item.exact_fallback_fraction *
                    100.0
                )

                << "%"

                << std::setw(9)
                << (
                    item.robust_quality_pass
                        ? "PASS"
                        : "FAIL"
                )

                << '\n';
        }

        std::cout
            << "\n[Trabajo algorítmico]\n";

        for (
            auto const& item :
            result.cases
        ) {

            std::cout
                << "BR="
                << std::fixed
                << std::setprecision(2)
                << item.bridge_budget_ratio

                << " tau="
                << std::setprecision(3)
                << item.fallback_exact_threshold

                << " | dist="
                << std::setprecision(3)
                << item.mean_computed_distances

                << " | visited="
                << item.mean_visited_members

                << " | pred="
                << item.mean_predicate_evaluations

                << " | fallback="
                << item.mean_fallback_count

                << " | fbSel=";

            if (
                item.fallback_selectivity_samples >
                0U
            ) {

                std::cout
                    << std::setprecision(6)
                    << item.mean_fallback_selectivity;

            } else {

                std::cout
                    << "NA";
            }

            std::cout
                << " | fbSelN="
                << item.fallback_selectivity_samples

                << " | exact="
                << item.exact_fallback_queries

                << "/"
                << result.query_count

                << '\n';
        }

        std::cout
            << "\n[Cola de Recall]\n";

        for (
            auto const& item :
            result.cases
        ) {

            std::cout
                << "BR="
                << std::fixed
                << std::setprecision(2)
                << item.bridge_budget_ratio

                << " tau="
                << std::setprecision(3)
                << item.fallback_exact_threshold

                << " | Recall="
                << std::setprecision(4)
                << item.mean_recall_at_k

                << " | Min="
                << item.min_recall_at_k

                << " | P01="
                << item.recall_p01

                << " | P05="
                << item.recall_p05

                << " | perfect="
                << item.perfect_queries

                << " | imperfect="
                << item.imperfect_queries

                << '\n';
        }

        std::cout
            << "\n[Validez]\n"
            << "Base valida                 : "
            << yes_no(
                result.base_file_valid
            )
            << '\n'
            << "Queries validas              : "
            << yes_no(
                result.query_file_valid
            )
            << '\n'
            << "Metadata valida              : "
            << yes_no(
                result.metadata_file_valid
            )
            << '\n'
            << "Ground truth completo        : "
            << yes_no(
                result.ground_truth_complete
            )
            << '\n'
            << "Todos los casos validos      : "
            << yes_no(
                result.all_cases_valid
            )
            << '\n'
            << "Configuracion robusta hallada: "
            << yes_no(
                result.robust_configuration_found
            )
            << "\n\n";

        std::cout
            << "[Invariantes]\n"
            << "RACORN modificado            : "
            << yes_no(
                result.racorn_modified
            )
            << '\n'
            << "Navegacion RACORN           : CPU\n"
            << "Metal                       : NO\n"
            << "GPU                         : NO\n"
            << "Bitset                      : NO\n"
            << "Exact fallback              : upstream RACORN-1+\n"
            << "Underfill invalida harness  : NO\n"
            << "Ground truth                : exacto filtrado\n\n";

        if (
            !result.valid
        ) {

            std::cerr
                << "[G-DATA2] FAIL\n";

            return 1;
        }

        std::cout
            << "[G-DATA2] PASS\n";

        return 0;

    } catch (
        std::exception const& error
    ) {

        std::cerr
            << "\n[G-DATA2] FAIL: "
            << error.what()
            << '\n';

        return 1;
    }
}

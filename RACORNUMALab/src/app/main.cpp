//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar G-DATA1.
//
//  G-DATA1 introduce SIFT como primera geometría vectorial real del
//  laboratorio.
//
//  La ubicación de los datos NO se codifica en el proyecto.
//  Debe proporcionarse mediante:
//
//      --data-dir <ruta>
//
//  o:
//
//      RACORNUMA_SIFT_DIR=<ruta>
//
//  Ejemplo:
//
//      --data-dir
//      /Users/gapt/Documents/Developer/DataRacornUmaLab/data/sift
//
//  Invariantes:
//  - RACORN upstream intacto.
//  - CPU solamente.
//  - Sin Metal/GPU.
//  - Ground truth filtrado exacto.
//
//  Ejecutar en Release.
//

#include "core/BuildInfo.hpp"
#include "core/RacornSiftValidation.hpp"
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
                index + 1 >=
                argc
            ) {

                throw std::runtime_error(
                    "--data-dir requiere una ruta."
                );
            }

            return
                std::filesystem::path(
                    argv[
                        index + 1
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
        "No se especifico el directorio SIFT.\n"
        "Use --data-dir <ruta> o defina RACORNUMA_SIFT_DIR."
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
        RacornSiftValidation;

    using racorn_uma::integration::
        RacornProbe;

    using racorn_uma::platform::apple::
        SystemInfo;

    try {

        const std::filesystem::path
            data_directory =
                resolve_data_directory(
                    argc,
                    argv
                );

        const auto result =
            RacornSiftValidation::run(
                data_directory
            );

        std::cout
            << "\n========================================\n"
            << " RACORNUMALab\n"
            << " Gate G-DATA1 - RACORN sobre SIFT\n"
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
            << "Subset base                 : train[0:100000]\n"
            << "Directorio externo          : "
            << result.data_directory.string()
            << '\n'
            << "N                           : "
            << result.vector_count
            << '\n'
            << "D                           : "
            << result.dimensions
            << '\n'
            << "Queries reales              : "
            << result.query_count
            << '\n'
            << "Metrica                     : L2\n"
            << "Metadata/geometria          : independientes\n"
            << "Ground truth original SIFT  : NO\n"
            << "Ground truth filtrado       : exacto independiente\n\n";

        std::cout
            << "[Configuracion RACORN]\n"
            << "M                           : "
            << result.connectivity
            << '\n'
            << "efConstruction              : "
            << result.ef_construction
            << '\n'
            << "efSearch                    : "
            << result.ef_search
            << '\n'
            << "k                           : "
            << result.k
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
            << "[Selectividad]\n"
            << "Media                       : "
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
            << "[Sweep Bridge Budget]\n\n";

        std::cout
            << std::left
            << std::setw(8)
            << "BR"

            << std::right
            << std::setw(11)
            << "Recall"

            << std::setw(11)
            << "MinRec"

            << std::setw(11)
            << "P01"

            << std::setw(11)
            << "P05"

            << std::setw(11)
            << "Perfect%"

            << std::setw(10)
            << "WorstQ"

            << std::setw(12)
            << "Dist"

            << std::setw(12)
            << "Visited"

            << std::setw(14)
            << "Pred"

            << std::setw(11)
            << "Fallback"

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
                << std::setw(8)
                << item.bridge_budget_ratio

                << std::right
                << std::setprecision(4)

                << std::setw(11)
                << item.mean_recall_at_k

                << std::setw(11)
                << item.min_recall_at_k

                << std::setw(11)
                << item.recall_p01

                << std::setw(11)
                << item.recall_p05

                << std::setw(11)
                << (
                    item.perfect_query_fraction *
                    100.0
                )

                << std::setw(10)
                << item.worst_query_index

                << std::setw(12)
                << item.mean_computed_distances

                << std::setw(12)
                << item.mean_visited_members

                << std::setw(14)
                << item.mean_predicate_evaluations

                << std::setw(11)
                << item.mean_fallback_count

                << std::setw(10)
                << (
                    item.robust_quality_pass
                        ? "PASS"
                        : "FAIL"
                )

                << '\n';
        }

        std::cout
            << "\n[Queries perfectas]\n";

        for (
            auto const& item :
            result.cases
        ) {

            std::cout
                << "BR="
                << std::fixed
                << std::setprecision(2)
                << item.bridge_budget_ratio

                << " | perfectas="
                << item.perfect_query_count
                << "/"
                << result.query_count

                << " | imperfectas="
                << item.imperfect_query_count

                << " | minRecall="
                << std::setprecision(4)
                << item.min_recall_at_k

                << '\n';
        }

        std::cout
            << "\n[Punto robusto minimo observado]\n";

        if (
            result.minimum_robust_point.found
        ) {

            std::cout
                << "BR*="
                << std::fixed
                << std::setprecision(2)
                << result
                    .minimum_robust_point
                    .bridge_budget_ratio

                << " | Recall="
                << std::setprecision(4)
                << result
                    .minimum_robust_point
                    .mean_recall_at_k

                << " | MinRecall="
                << result
                    .minimum_robust_point
                    .min_recall_at_k

                << " | P01="
                << result
                    .minimum_robust_point
                    .recall_p01

                << " | dist="
                << result
                    .minimum_robust_point
                    .mean_computed_distances

                << " | pred="
                << result
                    .minimum_robust_point
                    .mean_predicate_evaluations

                << '\n';

        } else {

            std::cout
                << "BR*=NO ENCONTRADO "
                << "en el sweep evaluado.\n";
        }

        std::cout
            << "\n[Validez]\n"
            << "Base externa valida          : "
            << yes_no(
                result.base_file_valid
            )
            << '\n'
            << "Queries externas validas     : "
            << yes_no(
                result.query_file_valid
            )
            << '\n'
            << "Metadata externa valida      : "
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
            << "\n\n";

        std::cout
            << "[Invariantes]\n"
            << "RACORN modificado            : "
            << yes_no(
                result.racorn_modified
            )
            << '\n'
            << "Navegacion                  : CPU\n"
            << "Metal                       : NO\n"
            << "GPU                         : NO\n"
            << "Bitset                      : NO\n"
            << "Predicate                   : CPU directo\n"
            << "Geometria                   : SIFT real\n"
            << "Datos vendorizados          : NO\n"
            << "Ground truth                : exacto filtrado\n\n";

        if (
            !result.valid
        ) {

            std::cerr
                << "[G-DATA1] FAIL\n";

            return 1;
        }

        std::cout
            << "[G-DATA1] PASS\n";

        return 0;

    } catch (
        std::exception const& error
    ) {

        std::cerr
            << "\n[G-DATA1] FAIL: "
            << error.what()
            << '\n';

        return 1;
    }
}

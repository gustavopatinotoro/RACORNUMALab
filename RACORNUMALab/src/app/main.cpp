//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar G-DATA1A — Tail Failure Audit.
//
//  Este Gate analiza las queries imperfectas observadas durante G-DATA1
//  y distingue:
//
//      - underfill;
//      - resultado vacío;
//      - top-k completo pero incorrecto.
//
//  Una degradación de calidad NO invalida el experimento.
//
//  El directorio SIFT se proporciona con:
//
//      --data-dir <ruta>
//
//  o:
//
//      RACORNUMA_SIFT_DIR=<ruta>
//
//  Ejecutar en Release.
//

#include "core/BuildInfo.hpp"
#include "core/RacornSiftTailAudit.hpp"
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
        RacornSiftTailAudit;

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
            RacornSiftTailAudit::run(
                data_directory
            );

        std::cout
            << "\n========================================\n"
            << " RACORNUMALab\n"
            << " Gate G-DATA1A - Tail Failure Audit\n"
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
            << "[Resumen de cola]\n\n";

        std::cout
            << std::left
            << std::setw(8)
            << "BR"

            << std::right
            << std::setw(11)
            << "Recall"

            << std::setw(11)
            << "MinRec"

            << std::setw(9)
            << "Perfect"

            << std::setw(10)
            << "Underfill"

            << std::setw(8)
            << "Empty"

            << std::setw(11)
            << "FullWrong"

            << std::setw(9)
            << "ResMin"

            << std::setw(9)
            << "WorstQ"

            << std::setw(10)
            << "Valid"

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

                << std::setw(9)
                << item.perfect_queries

                << std::setw(10)
                << item.underfilled_queries

                << std::setw(8)
                << item.empty_queries

                << std::setw(11)
                << item.full_but_inexact_queries

                << std::setw(9)
                << item.result_count_min

                << std::setw(9)
                << item.worst_query_index

                << std::setw(10)
                << (
                    item.valid
                        ? "PASS"
                        : "FAIL"
                )

                << '\n';
        }

        //
        // Histogramas.
        //

        std::cout
            << "\n[Histogramas Recall@10]\n";

        for (
            auto const& item :
            result.cases
        ) {

            std::cout
                << "BR="
                << std::fixed
                << std::setprecision(2)
                << item.bridge_budget_ratio
                << " | ";

            for (
                std::size_t bin = 0;
                bin <= 10U;
                ++bin
            ) {

                if (
                    item.recall_histogram[
                        bin
                    ] ==
                    0U
                ) {
                    continue;
                }

                std::cout
                    << std::fixed
                    << std::setprecision(1)
                    << (
                        static_cast<double>(
                            bin
                        ) /
                        10.0
                    )
                    << ":"
                    << item.recall_histogram[
                        bin
                    ]
                    << " ";
            }

            std::cout
                << '\n';
        }

        std::cout
            << "\n[Histogramas result_count]\n";

        for (
            auto const& item :
            result.cases
        ) {

            std::cout
                << "BR="
                << std::fixed
                << std::setprecision(2)
                << item.bridge_budget_ratio
                << " | ";

            for (
                std::size_t count = 0;
                count <= result.k;
                ++count
            ) {

                if (
                    item.result_count_histogram[
                        count
                    ] ==
                    0U
                ) {
                    continue;
                }

                std::cout
                    << count
                    << ":"
                    << item.result_count_histogram[
                        count
                    ]
                    << " ";
            }

            std::cout
                << '\n';
        }

        //
        // Trabajo.
        //

        std::cout
            << "\n[Trabajo RACORN]\n";

        for (
            auto const& item :
            result.cases
        ) {

            std::cout
                << "BR="
                << std::fixed
                << std::setprecision(2)
                << item.bridge_budget_ratio

                << " | dist="
                << std::setprecision(3)
                << item.mean_computed_distances

                << " | visited="
                << item.mean_visited_members

                << " | pred="
                << item.mean_predicate_evaluations

                << " | fallback="
                << item.mean_fallback_count

                << " | P01="
                << std::setprecision(4)
                << item.recall_p01

                << " | P05="
                << item.recall_p05

                << '\n';
        }

        //
        // Auditoría query por query.
        //

        std::cout
            << "\n[Queries imperfectas]\n";

        for (
            auto const& item :
            result.cases
        ) {

            std::cout
                << "\n--- BR="
                << std::fixed
                << std::setprecision(2)
                << item.bridge_budget_ratio
                << " | imperfectas="
                << item.imperfect_queries
                << " ---\n";

            for (
                auto const& record :
                item.imperfect_records
            ) {

                std::cout
                    << "q="
                    << record.query_index

                    << " tenant="
                    << record.tenant_required

                    << " eligible="
                    << record.selected_count

                    << " results="
                    << record.result_count

                    << " hits="
                    << record.hits
                    << "/"
                    << result.k

                    << " recall="
                    << std::fixed
                    << std::setprecision(1)
                    << record.recall_at_k

                    << " dist="
                    << record.computed_distances

                    << " visited="
                    << record.visited_members

                    << " pred="
                    << record.predicate_evaluations

                    << " fallback="
                    << record.fallback_count

                    << " type=";

                if (
                    record.empty
                ) {

                    std::cout
                        << "EMPTY";

                } else if (
                    record.underfilled
                ) {

                    std::cout
                        << "UNDERFILL";

                } else if (
                    record.full_but_inexact
                ) {

                    std::cout
                        << "FULL_BUT_INEXACT";

                } else {

                    std::cout
                        << "OTHER";
                }

                std::cout
                    << '\n';
            }
        }

        std::cout
            << "\n[Validez experimental]\n"
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
            << "Exact fallback              : NO\n"
            << "Underfill invalida harness  : NO\n"
            << "Ground truth                : exacto filtrado\n\n";

        if (
            !result.valid
        ) {

            std::cerr
                << "[G-DATA1A] FAIL\n";

            return 1;
        }

        std::cout
            << "[G-DATA1A] PASS\n";

        return 0;

    } catch (
        std::exception const& error
    ) {

        std::cerr
            << "\n[G-DATA1A] FAIL: "
            << error.what()
            << '\n';

        return 1;
    }
}

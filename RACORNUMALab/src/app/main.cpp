//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar G-DATA2A — Exact-Fallback Targeting Audit.
//
//  Este Gate evalúa RACORN-1+ como detector de queries difíciles.
//
//  No se optimiza rendimiento.
//  No se entrena ningún modelo.
//  No se utiliza Metal.
//
//  Directorio SIFT:
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
#include "core/RacornSiftTargetingAudit.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/SystemInfo.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

void print_ids(
    std::vector<std::size_t> const& ids,
    std::size_t maximum
) {

    const std::size_t count =
        std::min(
            ids.size(),
            maximum
        );

    for (
        std::size_t index = 0;
        index < count;
        ++index
    ) {

        if (
            index !=
            0U
        ) {

            std::cout
                << ",";
        }

        std::cout
            << ids[
                index
            ];
    }

    if (
        ids.size() >
        maximum
    ) {

        std::cout
            << ",...(+"
            << (
                ids.size() -
                maximum
            )
            << ")";
    }
}

void print_distribution(
    char const* label,
    racorn_uma::core::TargetingDistributionStats const& stats
) {

    std::cout
        << label
        << " | n="
        << stats.samples;

    if (
        stats.samples >
        0U
    ) {

        std::cout
            << " min="
            << std::fixed
            << std::setprecision(6)
            << stats.minimum

            << " p05="
            << stats.p05

            << " median="
            << stats.median

            << " mean="
            << stats.mean

            << " p95="
            << stats.p95

            << " max="
            << stats.maximum;
    }

    std::cout
        << '\n';
}

} // namespace

int main(
    int argc,
    char* argv[]
) {

    using racorn_uma::core::
        BuildInfo;

    using racorn_uma::core::
        RacornSiftTargetingAudit;

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
            RacornSiftTargetingAudit::run(
                data_directory
            );

        std::cout
            << "\n========================================\n"
            << " RACORNUMALab\n"
            << " Gate G-DATA2A - Exact-Fallback Targeting Audit\n"
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
            << "k                           : "
            << result.k
            << '\n'
            << "BR                          : "
            << std::fixed
            << std::setprecision(2)
            << result.bridge_budget_ratio
            << '\n'
            << "two_hop_threshold           : "
            << result.two_hop_threshold
            << "\n\n";

        //
        // Baseline.
        //

        std::cout
            << "[Baseline RACORN-1]\n"
            << "Mean Recall@10              : "
            << std::fixed
            << std::setprecision(4)
            << result.baseline_mean_recall
            << '\n'
            << "Min Recall@10               : "
            << result.baseline_min_recall
            << '\n'
            << "Queries perfectas           : "
            << result.baseline_perfect_queries
            << '\n'
            << "Queries dificiles           : "
            << result.baseline_hard_queries
            << "\nHard query IDs              : ";

        print_ids(
            result.baseline_hard_query_ids,
            100U
        );

        std::cout
            << "\n\n";

        //
        // Distribución de la señal baseline.
        //

        std::cout
            << "[fallback_selectivity final bajo RACORN-1]\n";

        print_distribution(
            "Hard",
            result
                .baseline_hard_fallback_selectivity
        );

        print_distribution(
            "Easy",
            result
                .baseline_easy_fallback_selectivity
        );

        //
        // Matriz de clasificación.
        //

        std::cout
            << "\n[Targeting Matrix]\n\n";

        std::cout
            << std::left
            << std::setw(9)
            << "Tau"

            << std::right
            << std::setw(7)
            << "Exact"

            << std::setw(7)
            << "TP"

            << std::setw(7)
            << "FP"

            << std::setw(7)
            << "FN"

            << std::setw(7)
            << "TN"

            << std::setw(11)
            << "Prec"

            << std::setw(11)
            << "Cover"

            << std::setw(11)
            << "Spec"

            << std::setw(11)
            << "F1"

            << std::setw(11)
            << "BalAcc"

            << '\n';

        for (
            auto const& item :
            result.cases
        ) {

            std::cout
                << std::left
                << std::fixed
                << std::setprecision(4)
                << std::setw(9)
                << item.threshold

                << std::right
                << std::setw(7)
                << item.exact_fallback_queries

                << std::setw(7)
                << item.true_positive

                << std::setw(7)
                << item.false_positive

                << std::setw(7)
                << item.false_negative

                << std::setw(7)
                << item.true_negative

                << std::fixed
                << std::setprecision(4)

                << std::setw(11)
                << item.precision

                << std::setw(11)
                << item.coverage

                << std::setw(11)
                << item.specificity

                << std::setw(11)
                << item.f1

                << std::setw(11)
                << item.balanced_accuracy

                << '\n';
        }

        //
        // Calidad final.
        //

        std::cout
            << "\n[Calidad final]\n";

        for (
            auto const& item :
            result.cases
        ) {

            std::cout
                << "tau="
                << std::fixed
                << std::setprecision(4)
                << item.threshold

                << " | Recall="
                << item.mean_recall_at_k

                << " | Min="
                << item.min_recall_at_k

                << " | P01="
                << item.recall_p01

                << " | perfect="
                << item.perfect_queries

                << " | hard_rescatadas="
                << item.rescued_hard_queries
                << "/"
                << result.baseline_hard_queries

                << " | hard_restantes="
                << item.remaining_hard_queries

                << " | easy_degradadas="
                << item.newly_imperfect_easy_queries

                << " | exact="
                << item.exact_fallback_queries
                << "/"
                << result.query_count

                << " | robust="
                << (
                    item.robust_quality_pass
                        ? "PASS"
                        : "FAIL"
                )

                << '\n';
        }

        //
        // Coste.
        //

        std::cout
            << "\n[Trabajo algorítmico]\n";

        for (
            auto const& item :
            result.cases
        ) {

            std::cout
                << "tau="
                << std::fixed
                << std::setprecision(4)
                << item.threshold

                << " | exact%="
                << std::setprecision(2)
                << (
                    item.exact_fallback_fraction *
                    100.0
                )

                << " | dist="
                << std::setprecision(3)
                << item.mean_computed_distances

                << " | visited="
                << item.mean_visited_members

                << " | pred="
                << item.mean_predicate_evaluations

                << " | fallback="
                << item.mean_fallback_count

                << " | fbSel="
                << std::setprecision(6)
                << item.mean_fallback_selectivity

                << '\n';
        }

        //
        // False negatives.
        //

        std::cout
            << "\n[Queries dificiles NO detectadas]\n";

        for (
            auto const& item :
            result.cases
        ) {

            std::cout
                << "tau="
                << std::fixed
                << std::setprecision(4)
                << item.threshold
                << " | FN="
                << item.false_negative
                << " | ids=";

            print_ids(
                item.missed_hard_query_ids,
                100U
            );

            std::cout
                << '\n';
        }

        //
        // False positives.
        //

        std::cout
            << "\n[False positives - muestra]\n";

        for (
            auto const& item :
            result.cases
        ) {

            std::cout
                << "tau="
                << std::fixed
                << std::setprecision(4)
                << item.threshold
                << " | FP="
                << item.false_positive
                << " | ids=";

            print_ids(
                item.false_positive_query_ids,
                20U
            );

            std::cout
                << '\n';
        }

        //
        // Mejor F1 observado.
        //

        std::cout
            << "\n[Mejor detector observado por F1]\n";

        if (
            result.best_f1_found
        ) {

            std::cout
                << "tau*="
                << std::fixed
                << std::setprecision(4)
                << result.best_f1_threshold

                << " | F1="
                << result.best_f1
                << '\n';

        } else {

            std::cout
                << "No encontrado.\n";
        }

        //
        // Validez.
        //

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
            << "Baseline valido             : "
            << yes_no(
                result.baseline_valid
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
            << "Modelo aprendido            : NO\n"
            << "BR                          : 1.0 fijo\n"
            << "Etiqueta hard               : RACORN-1 vs GT exacto\n"
            << "Prediccion hard             : fallback_to_exact\n"
            << "Ground truth                : exacto filtrado\n\n";

        if (
            !result.valid
        ) {

            std::cerr
                << "[G-DATA2A] FAIL\n";

            return 1;
        }

        std::cout
            << "[G-DATA2A] PASS\n";

        return 0;

    } catch (
        std::exception const& error
    ) {

        std::cerr
            << "\n[G-DATA2A] FAIL: "
            << error.what()
            << '\n';

        return 1;
    }
}

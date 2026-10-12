//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar el benchmark de punto de equilibrio CPU/GPU para la
//  materialización de bitsets de predicados.
//
//  Gate:
//  G-UMA5A — Predicate Engine Break-Even.
//
//  Este Gate no ejecuta RACORN.
//
//  Objetivo:
//  Identificar experimentalmente para qué combinación de:
//
//      N
//      +
//      complejidad del predicado
//
//  el uso síncrono de GPU Metal comienza a superar la CPU.
//
//  Distinción esencial:
//
//      GPU device time
//
//  mide el kernel.
//
//      GPU E2E time
//
//  mide el coste realmente observado por el consumidor CPU.
//
//  Para una política dinámica futura utilizaremos GPU E2E, no
//  únicamente el tiempo interno del kernel.
//

#include "core/BuildInfo.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/PredicateBreakEvenBenchmark.hpp"
#include "platform/apple/SystemInfo.hpp"

#include <iomanip>
#include <iostream>

namespace {

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
        PredicateBreakEvenBenchmark;

    using racorn_uma::platform::apple::
        SystemInfo;

    const auto result =
        PredicateBreakEvenBenchmark::run();

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G-UMA5A - CPU/GPU Break-Even\n"
        << "========================================\n\n";

    std::cout
        << "[Plataforma]\n"
        << "Compilador              : "
        << BuildInfo::compiler()
        << '\n'
        << "Estandar C++            : "
        << BuildInfo::cpp_standard()
        << '\n'
        << "Arquitectura            : "
        << BuildInfo::architecture()
        << '\n'
        << "Modelo hardware         : "
        << SystemInfo::hardware_model()
        << "\n\n";

    std::cout
        << "[RACORN congelado]\n"
        << "Repositorio              : "
        << RacornProbe::repository()
        << '\n'
        << "Version                  : "
        << RacornProbe::version()
        << '\n'
        << "Commit                   : "
        << RacornProbe::commit()
        << "\n\n";

    std::cout
        << "[Metal]\n"
        << "Metal disponible         : "
        << yes_no(
            result.metal_available
        )
        << '\n'
        << "Dispositivo              : "
        << result.device_name
        << '\n'
        << "Memoria unificada        : "
        << yes_no(
            result.unified_memory
        )
        << '\n'
        << "Kernel                   : "
        << result.kernel_name
        << '\n'
        << "Pipeline                 : "
        << yes_no(
            result.pipeline_available
        )
        << '\n'
        << "Execution width          : "
        << result.thread_execution_width
        << '\n'
        << "Threads/group max        : "
        << result
            .max_total_threads_per_threadgroup
        << '\n'
        << "Threads/group usados     : "
        << result.threads_per_threadgroup
        << "\n\n";

    std::cout
        << "[Metodologia]\n"
        << "N maximo                 : "
        << result.max_element_count
        << '\n'
        << "Columnas metadata        : "
        << result.metadata_column_count
        << '\n'
        << "Warmups/caso             : "
        << result.warmup_iterations
        << '\n'
        << "Mediciones/caso          : "
        << result.measured_iterations
        << '\n'
        << "Allocations en timing    : NO\n"
        << "Pipeline creation timing : NO\n"
        << "GPU dispatch/caso        : 1 por medicion\n"
        << "GPU sync/caso            : 1 por medicion\n\n";

    std::cout
        << "[Predicados]\n"
        << "P1 = tenant == 7\n"
        << "P3 = P1 AND year>=2022 AND language==2\n"
        << "P5 = P3 AND category IN {3,7,9}"
        << " AND security_level<=2\n\n";

    std::cout
        << "[Resultados]\n\n";

    std::cout
        << std::left
        << std::setw(6)
        << "Pred"

        << std::right
        << std::setw(10)
        << "N"

        << std::setw(10)
        << "Sel.%"

        << std::setw(12)
        << "CPU p50"

        << std::setw(12)
        << "CPU p95"

        << std::setw(12)
        << "GPU p50"

        << std::setw(12)
        << "GPU p95"

        << std::setw(12)
        << "Dev p50"

        << std::setw(10)
        << "Speedup"

        << std::setw(10)
        << "Winner"

        << '\n';

    for (
        auto const& benchmark_case :
        result.cases
    ) {

        std::cout
            << std::left
            << std::setw(6)
            << benchmark_case
                .predicate_name

            << std::right
            << std::setw(10)
            << benchmark_case
                .element_count

            << std::setw(10)
            << std::fixed
            << std::setprecision(3)
            << (
                benchmark_case
                    .observed_selectivity *
                100.0
            )

            << std::setw(12)
            << std::setprecision(4)
            << benchmark_case
                .cpu_p50_ms

            << std::setw(12)
            << benchmark_case
                .cpu_p95_ms

            << std::setw(12)
            << benchmark_case
                .gpu_wall_p50_ms

            << std::setw(12)
            << benchmark_case
                .gpu_wall_p95_ms;

        if (
            benchmark_case
                .gpu_device_timing_available
        ) {

            std::cout
                << std::setw(12)
                << benchmark_case
                    .gpu_device_p50_ms;

        } else {

            std::cout
                << std::setw(12)
                << "N/A";
        }

        std::cout
            << std::setw(9)
            << std::setprecision(2)
            << benchmark_case
                .gpu_wall_speedup_vs_cpu
            << "x"

            << std::setw(10)
            << (
                benchmark_case
                    .gpu_faster_at_p50
                    ? "GPU"
                    : "CPU"
            )

            << '\n';
    }

    std::cout
        << "\n[Exactitud]\n";

    for (
        auto const& benchmark_case :
        result.cases
    ) {

        std::cout
            << benchmark_case.predicate_name
            << " N="
            << benchmark_case.element_count
            << " -> selected="
            << benchmark_case.selected_count
            << ", mismatches="
            << benchmark_case.mismatch_word_count
            << ", exact="
            << yes_no(
                benchmark_case.exact_match
            )
            << '\n';
    }

    if (
        !result.metal_error.empty()
    ) {

        std::cout
            << "\n[Error Metal]\n"
            << result.metal_error
            << '\n';
    }

    std::cout
        << "\n[Invariantes]\n"
        << "RACORN modificado        : NO\n"
        << "RACORN ejecutado         : NO\n"
        << "Navegacion GPU           : NO\n"
        << "Predicate GPU            : SI\n"
        << "Bitset GPU               : SI\n"
        << "Atomics                  : NO\n"
        << "Copia explicita a VRAM   : NO\n"
        << "Casos exactos            : "
        << yes_no(
            result.all_cases_exact
        )
        << "\n\n";

    if (
        !result.valid
    ) {

        std::cerr
            << "[G-UMA5A] FAIL\n";

        return 1;
    }

    std::cout
        << "[G-UMA5A] PASS\n";

    return 0;
}

//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar G-UMA2 y verificar equivalencia exacta entre un predicado
//  de metadatos evaluado por CPU y el mismo predicado evaluado mediante
//  GPU Metal.
//
//  Gate:
//  G-UMA2 — Predicate Engine Metal.
//
//  Restricción arquitectónica principal:
//
//      RACORN CPU permanece completamente intacto.
//
//  Este Gate todavía no conecta la máscara GPU con RACORN.
//
//  Condición PASS:
//  - Metal disponible.
//  - UMA disponible.
//  - Buffers Shared válidos.
//  - Kernel Metal ejecutado.
//  - Todos los valores GPU pertenecen a {0,1}.
//  - CPU y GPU seleccionan exactamente los mismos IDs.
//  - Cero diferencias en 1.048.576 registros.
//
//  Nota:
//  La máscara utiliza un byte por vector.
//  La compactación a bitset pertenece a G-UMA3.
//

#include "core/BuildInfo.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/PredicateEngineProbe.hpp"
#include "platform/apple/SystemInfo.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {

double bytes_to_mib(
    std::uint64_t bytes
) noexcept {

    return
        static_cast<double>(
            bytes
        ) /
        (
            1024.0 *
            1024.0
        );
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
        PredicateEngineProbe;

    using racorn_uma::platform::apple::
        SystemInfo;

    const auto result =
        PredicateEngineProbe::run();

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G-UMA2 - Predicate Engine Metal\n"
        << "========================================\n\n";

    std::cout
        << "[Plataforma]\n";

    std::cout
        << "Compilador              : "
        << BuildInfo::compiler()
        << '\n';

    std::cout
        << "Estandar C++            : "
        << BuildInfo::cpp_standard()
        << '\n';

    std::cout
        << "Arquitectura            : "
        << BuildInfo::architecture()
        << '\n';

    std::cout
        << "Modelo hardware         : "
        << SystemInfo::hardware_model()
        << "\n\n";

    std::cout
        << "[RACORN congelado]\n";

    std::cout
        << "Repositorio              : "
        << RacornProbe::repository()
        << '\n';

    std::cout
        << "Version                  : "
        << RacornProbe::version()
        << '\n';

    std::cout
        << "Commit                   : "
        << RacornProbe::commit()
        << "\n\n";

    std::cout
        << "[Predicate Engine]\n";

    std::cout
        << "Metal disponible         : "
        << yes_no(
            result.metal_available
        )
        << '\n';

    std::cout
        << "Dispositivo              : "
        << result.device_name
        << '\n';

    std::cout
        << "Memoria unificada        : "
        << yes_no(
            result.unified_memory
        )
        << '\n';

    std::cout
        << "Biblioteca Metal         : "
        << yes_no(
            result.library_available
        )
        << '\n';

    std::cout
        << "Kernel                   : "
        << result.kernel_name
        << '\n';

    std::cout
        << "Kernel encontrado        : "
        << yes_no(
            result.kernel_available
        )
        << '\n';

    std::cout
        << "Pipeline                 : "
        << yes_no(
            result.pipeline_available
        )
        << '\n';

    std::cout
        << "Command Queue            : "
        << yes_no(
            result.command_queue_available
        )
        << '\n';

    std::cout
        << "Metadata Shared          : "
        << yes_no(
            result.metadata_buffer_available
        )
        << '\n';

    std::cout
        << "Mask Shared              : "
        << yes_no(
            result.mask_buffer_available
        )
        << '\n';

    std::cout
        << "GPU completada           : "
        << yes_no(
            result.command_completed
        )
        << "\n\n";

    std::cout
        << "[Metadata]\n";

    std::cout
        << "Registros                : "
        << result.element_count
        << '\n';

    std::cout
        << "Columnas                 : "
        << result.metadata_column_count
        << '\n';

    std::cout
        << "Metadata buffer          : "
        << std::fixed
        << std::setprecision(2)
        << bytes_to_mib(
            result.metadata_bytes
        )
        << " MiB\n";

    std::cout
        << "Eligibility mask         : "
        << std::fixed
        << std::setprecision(2)
        << bytes_to_mib(
            result.mask_bytes
        )
        << " MiB\n\n";

    std::cout
        << "[Filtro]\n";

    std::cout
        << "tenant_id                : == 7\n";

    std::cout
        << "year                     : >= 2022\n";

    std::cout
        << "language                 : == 2\n";

    std::cout
        << "category                 : IN {3,7,9}\n";

    std::cout
        << "security_level           : <= 2\n\n";

    std::cout
        << "[Dispatch]\n";

    std::cout
        << "Execution width          : "
        << result.thread_execution_width
        << '\n';

    std::cout
        << "Threads/group max        : "
        << result
            .max_total_threads_per_threadgroup
        << '\n';

    std::cout
        << "Threads/group usados     : "
        << result.threads_per_threadgroup
        << '\n';

    std::cout
        << "Hilos logicos GPU        : "
        << result.element_count
        << '\n';

    std::cout
        << "Dispatches GPU           : 1\n";

    std::cout
        << "Sincronizaciones CPU/GPU : 1\n";

    std::cout
        << "Atomics                  : 0\n";

    std::cout
        << "Barreras threadgroup     : 0\n\n";

    std::cout
        << "[Resultado del filtro]\n";

    std::cout
        << "Seleccionados CPU        : "
        << result.cpu_selected_count
        << '\n';

    std::cout
        << "Seleccionados GPU        : "
        << result.gpu_selected_count
        << '\n';

    std::cout
        << "Selectividad observada   : "
        << std::fixed
        << std::setprecision(6)
        << (
            result.observed_selectivity *
            100.0
        )
        << " %\n\n";

    std::cout
        << "[Verificacion semantica]\n";

    std::cout
        << "Registros verificados    : "
        << result.element_count
        << '\n';

    std::cout
        << "Valores mask invalidos   : "
        << result.invalid_mask_value_count
        << '\n';

    std::cout
        << "Diferencias CPU/GPU      : "
        << result.mismatch_count
        << '\n';

    std::cout
        << "Coincidencia exacta      : "
        << yes_no(
            result.exact_match
        )
        << '\n';

    if (
        result.mismatch_count >
            0 &&
        result.first_mismatch_index !=
            std::numeric_limits<
                std::size_t
            >::max()
    ) {

        std::cout
            << "Primer mismatch indice   : "
            << result.first_mismatch_index
            << '\n';

        std::cout
            << "Esperado                 : "
            << static_cast<unsigned>(
                result.first_mismatch_expected
            )
            << '\n';

        std::cout
            << "Observado                : "
            << static_cast<unsigned>(
                result.first_mismatch_observed
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
        << "\n[Invariantes]\n";

    std::cout
        << "RACORN modificado        : NO\n";

    std::cout
        << "HNSW ejecutado           : NO\n";

    std::cout
        << "Navegacion GPU           : NO\n";

    std::cout
        << "Predicate CPU            : SI\n";

    std::cout
        << "Predicate GPU            : SI\n";

    std::cout
        << "Metadata Shared          : SI\n";

    std::cout
        << "Resultado Shared         : SI\n";

    std::cout
        << "Copia explicita a VRAM   : NO\n\n";

    if (
        !result.valid
    ) {

        std::cerr
            << "[G-UMA2] FAIL\n";

        return 1;
    }

    std::cout
        << "[G-UMA2] PASS\n";

    return 0;
}

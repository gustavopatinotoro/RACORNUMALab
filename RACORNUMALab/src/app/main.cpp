//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar el primer flujo compute real CPU↔GPU utilizando memoria
//  unificada mediante MTLStorageModeShared.
//
//  Gate:
//  G-UMA1 — Shared Buffer Compute Probe.
//
//  Flujo validado:
//
//      CPU inicializa
//             ↓
//      MTLBuffer Shared
//             ↓
//      GPU Metal
//             ↓
//      mismo MTLBuffer Shared
//             ↓
//      CPU verifica
//
//  Condición PASS:
//  - Metal disponible.
//  - Memoria unificada.
//  - Biblioteca Metal disponible.
//  - Kernel encontrado.
//  - Pipeline creado.
//  - Command Queue creada.
//  - Buffer Shared válido.
//  - Command Buffer completado.
//  - Cero diferencias respecto del oráculo CPU.
//
//  RACORN:
//  Permanece completamente intacto.
//
//  Importante:
//  Este Gate no mide rendimiento.
//

#include "core/BuildInfo.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/SharedBufferProbe.hpp"
#include "platform/apple/SystemInfo.hpp"

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
        SharedBufferProbe;

    using racorn_uma::platform::apple::
        SystemInfo;

    const auto result =
        SharedBufferProbe::run();

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G-UMA1 - Shared Buffer Compute\n"
        << "========================================\n\n";

    //
    // Plataforma.
    //

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

    //
    // Provenance RACORN.
    //

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

    //
    // Compute Metal.
    //

    std::cout
        << "[Metal Compute]\n";

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
        << "Compute Pipeline         : "
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
        << "Shared Buffer            : "
        << yes_no(
            result.shared_buffer_available
        )
        << '\n';

    std::cout
        << "Command Buffer           : "
        << yes_no(
            result.command_buffer_available
        )
        << '\n';

    std::cout
        << "Compute Encoder          : "
        << yes_no(
            result.encoder_available
        )
        << '\n';

    std::cout
        << "GPU completada           : "
        << yes_no(
            result.command_completed
        )
        << "\n\n";

    //
    // Workload.
    //

    std::cout
        << "[Workload]\n";

    std::cout
        << "Elementos                : "
        << result.element_count
        << '\n';

    std::cout
        << "Buffer                    : "
        << std::fixed
        << std::setprecision(2)
        << bytes_to_mib(
            result.buffer_bytes
        )
        << " MiB\n";

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
        << "Dispatches GPU           : 1\n";

    std::cout
        << "Sincronizaciones CPU/GPU : 1\n\n";

    //
    // Verificación.
    //

    std::cout
        << "[Verificacion bit-exacta]\n";

    std::cout
        << "Elementos verificados    : "
        << result.element_count
        << '\n';

    std::cout
        << "Diferencias              : "
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
            << "Esperado                 : 0x"
            << std::hex
            << std::uppercase
            << result.first_mismatch_expected
            << std::dec
            << std::nouppercase
            << '\n';

        std::cout
            << "Observado                : 0x"
            << std::hex
            << std::uppercase
            << result.first_mismatch_observed
            << std::dec
            << std::nouppercase
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
        << "Predicate Engine         : NO\n";

    std::cout
        << "Copia CPU->VRAM explicita: NO\n";

    std::cout
        << "Copia VRAM->CPU explicita: NO\n";

    std::cout
        << "Kernel GPU ejecutado     : "
        << yes_no(
            result.command_completed
        )
        << "\n\n";

    if (
        !result.valid
    ) {

        std::cerr
            << "[G-UMA1] FAIL\n";

        return 1;
    }

    std::cout
        << "[G-UMA1] PASS\n";

    return 0;
}

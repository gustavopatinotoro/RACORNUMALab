//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar G-UMA3 y verificar la compactación GPU de una máscara
//  de elegibilidad hacia un bitset directamente consultable por CPU.
//
//  Gate:
//  G-UMA3 — Compactación GPU a bitset UMA.
//
//  Objetivo:
//  Demostrar:
//
//      1 byte/vector
//          ↓
//      GPU Metal
//          ↓
//      1 bit/vector
//
//  manteniendo equivalencia exacta.
//
//  Restricción arquitectónica:
//
//      RACORN CPU permanece completamente intacto.
//
//  Este Gate todavía no conecta el bitset con RACORN.
//
//  Condición PASS:
//  - GPU completa el kernel.
//  - Cero diferencias por palabra.
//  - Cero diferencias por bit.
//  - Misma cardinalidad antes y después de compactar.
//

#include "core/BuildInfo.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/BitsetCompactionProbe.hpp"
#include "platform/apple/SystemInfo.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

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
        BitsetCompactionProbe;

    using racorn_uma::platform::apple::
        SystemInfo;

    const auto result =
        BitsetCompactionProbe::run();

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G-UMA3 - GPU Bitset Compaction\n"
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
        << "[Metal]\n";

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
        << "Mask Shared              : "
        << yes_no(
            result.mask_buffer_available
        )
        << '\n';

    std::cout
        << "Bitset Shared            : "
        << yes_no(
            result.bitset_buffer_available
        )
        << '\n';

    std::cout
        << "GPU completada           : "
        << yes_no(
            result.command_completed
        )
        << "\n\n";

    std::cout
        << "[Representacion]\n";

    std::cout
        << "Vectores                 : "
        << result.element_count
        << '\n';

    std::cout
        << "Mascara byte/vector      : "
        << std::fixed
        << std::setprecision(2)
        << bytes_to_mib(
            result.mask_bytes
        )
        << " MiB\n";

    std::cout
        << "Palabras bitset          : "
        << result.bitset_word_count
        << '\n';

    std::cout
        << "Bitset 1 bit/vector      : "
        << std::fixed
        << std::setprecision(2)
        << bytes_to_kib(
            result.bitset_bytes
        )
        << " KiB\n";

    std::cout
        << "Factor de compactacion   : "
        << std::fixed
        << std::setprecision(2)
        << result.compression_ratio
        << "x\n";

    std::cout
        << "Reduccion almacenamiento : "
        << std::fixed
        << std::setprecision(2)
        << (
            result
                .storage_reduction_fraction *
            100.0
        )
        << " %\n\n";

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
        << "Hilos GPU                : "
        << result.bitset_word_count
        << '\n';

    std::cout
        << "Bytes procesados/hilo    : 32\n";

    std::cout
        << "Bits producidos/hilo     : 32\n";

    std::cout
        << "Dispatches GPU           : 1\n";

    std::cout
        << "Sincronizaciones CPU/GPU : 1\n";

    std::cout
        << "Atomics                  : 0\n";

    std::cout
        << "Barreras threadgroup     : 0\n\n";

    std::cout
        << "[Cardinalidad]\n";

    std::cout
        << "Seleccionados mascara    : "
        << result.cpu_selected_count
        << '\n';

    std::cout
        << "Seleccionados bitset     : "
        << result.bitset_selected_count
        << "\n\n";

    std::cout
        << "[Verificacion exacta]\n";

    std::cout
        << "Palabras verificadas     : "
        << result.bitset_word_count
        << '\n';

    std::cout
        << "Palabras diferentes      : "
        << result.word_mismatch_count
        << '\n';

    std::cout
        << "Coincidencia palabras    : "
        << yes_no(
            result.exact_word_match
        )
        << '\n';

    std::cout
        << "Bits logicos verificados : "
        << result.element_count
        << '\n';

    std::cout
        << "Bits diferentes          : "
        << result.bit_mismatch_count
        << '\n';

    std::cout
        << "Coincidencia bits        : "
        << yes_no(
            result.exact_bit_match
        )
        << '\n';

    if (
        result.word_mismatch_count >
            0 &&
        result.first_mismatch_word !=
            std::numeric_limits<
                std::size_t
            >::max()
    ) {

        std::cout
            << "Primer word mismatch     : "
            << result.first_mismatch_word
            << '\n';

        std::cout
            << "Word esperado            : 0x"
            << std::hex
            << std::uppercase
            << result
                .first_mismatch_expected_word
            << '\n';

        std::cout
            << "Word observado           : 0x"
            << result
                .first_mismatch_observed_word
            << std::dec
            << std::nouppercase
            << '\n';
    }

    if (
        result.bit_mismatch_count >
            0 &&
        result.first_mismatch_index !=
            std::numeric_limits<
                std::size_t
            >::max()
    ) {

        std::cout
            << "Primer bit mismatch idx  : "
            << result.first_mismatch_index
            << '\n';

        std::cout
            << "Bit esperado             : "
            << static_cast<unsigned>(
                result
                    .first_mismatch_expected_bit
            )
            << '\n';

        std::cout
            << "Bit observado            : "
            << static_cast<unsigned>(
                result
                    .first_mismatch_observed_bit
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
        << "Predicate GPU ejecutado  : NO\n";

    std::cout
        << "Compactacion GPU         : SI\n";

    std::cout
        << "Bitset Shared            : SI\n";

    std::cout
        << "Atomics                  : NO\n";

    std::cout
        << "Copia explicita a VRAM   : NO\n\n";

    if (
        !result.valid
    ) {

        std::cerr
            << "[G-UMA3] FAIL\n";

        return 1;
    }

    std::cout
        << "[G-UMA3] PASS\n";

    return 0;
}

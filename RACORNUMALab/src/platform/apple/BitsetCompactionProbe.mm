//
//  BitsetCompactionProbe.mm
//  RACORNUMALab
//
//  Propósito:
//  Implementar la compactación GPU de una máscara uint8_t hacia un
//  bitset uint32_t almacenado en memoria unificada.
//
//  Gate:
//  G-UMA3 — Compactación GPU a bitset UMA.
//
//  Workload:
//      1.048.576 decisiones de elegibilidad.
//
//  Entrada:
//      1 byte/vector = 1 MiB.
//
//  Salida:
//      1 bit/vector = 128 KiB.
//
//  Diseño:
//      32 entradas uint8_t
//              ↓
//         un hilo GPU
//              ↓
//         un uint32_t
//
//  Ejemplo:
//
//      eligibility[0]  -> bit 0
//      eligibility[1]  -> bit 1
//      ...
//      eligibility[31] -> bit 31
//
//  Después:
//
//      bitset[index >> 5]
//
//  contiene el estado de 32 vectores.
//
//  Validaciones:
//  1. CPU calcula cada palabra esperada.
//  2. Se compara palabra por palabra.
//  3. Se vuelve a consultar cada uno de los 1.048.576 bits.
//  4. La cardinalidad del bitset debe ser idéntica a la máscara.
//
//  Invariantes:
//  - RACORN no se modifica.
//  - HNSW no se ejecuta.
//  - Sin atomics.
//  - Sin barreras.
//  - Sin memoria threadgroup.
//  - Un único dispatch.
//  - Una única sincronización.
//  - Sin copias explícitas CPU↔VRAM.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Recursos administrados dentro de autoreleasepool.
//

#include "platform/apple/BitsetCompactionProbe.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace racorn_uma::platform::apple {

namespace {

constexpr char const* kKernelName =
    "compact_eligibility_to_bitset";

constexpr std::size_t kElementCount =
    1U << 20U;

constexpr std::size_t kBitsPerWord =
    32U;

constexpr std::uint64_t kMaskSeed =
    0xA0761D6478BD642FULL;

std::string string_from_nsstring(
    NSString* value
) {

    if (
        value ==
        nil
    ) {

        return {};
    }

    const char* utf8 =
        [value UTF8String];

    if (
        utf8 ==
        nullptr
    ) {

        return {};
    }

    return std::string(
        utf8
    );
}

std::uint64_t splitmix64(
    std::uint64_t value
) noexcept {

    value +=
        0x9E3779B97F4A7C15ULL;

    value =
        (
            value ^
            (
                value >>
                30U
            )
        ) *
        0xBF58476D1CE4E5B9ULL;

    value =
        (
            value ^
            (
                value >>
                27U
            )
        ) *
        0x94D049BB133111EBULL;

    return
        value ^
        (
            value >>
                31U
        );
}

bool deterministic_eligibility(
    std::size_t index
) noexcept {

    const std::uint64_t hash =
        splitmix64(
            kMaskSeed ^
            static_cast<std::uint64_t>(
                index
            )
        );

    //
    // Aproximadamente 0.1 % de selectividad.
    //
    // La finalidad no es modelar todavía un filtro concreto, sino
    // estresar una máscara dispersa similar a los escenarios FANNS
    // críticos estudiados anteriormente.
    //

    return
        (
            hash %
            1000ULL
        ) ==
        0ULL;
}

std::size_t bitset_word_count_for(
    std::size_t element_count
) noexcept {

    return
        (
            element_count +
            kBitsPerWord -
            1U
        ) /
        kBitsPerWord;
}

std::uint32_t expected_word(
    std::uint8_t const* mask,
    std::size_t element_count,
    std::size_t word_index
) noexcept {

    const std::size_t base =
        word_index *
        kBitsPerWord;

    std::uint32_t word =
        0U;

    for (
        std::size_t bit = 0;
        bit < kBitsPerWord;
        ++bit
    ) {

        const std::size_t element_index =
            base +
            bit;

        if (
            element_index >=
            element_count
        ) {

            break;
        }

        if (
            mask[element_index] !=
            0U
        ) {

            word |=
                (
                    std::uint32_t{1U}
                    <<
                    bit
                );
        }
    }

    return word;
}

bool bitset_lookup(
    std::uint32_t const* bitset,
    std::size_t index
) noexcept {

    const std::size_t word_index =
        index >>
        5U;

    const std::size_t bit_index =
        index &
        31U;

    return
        (
            (
                bitset[word_index] >>
                bit_index
            ) &
            1U
        ) !=
        0U;
}

std::size_t choose_threads_per_threadgroup(
    std::size_t execution_width,
    std::size_t maximum
) noexcept {

    if (
        maximum ==
        0
    ) {

        return 0;
    }

    constexpr std::size_t preferred =
        256U;

    std::size_t selected =
        std::min(
            preferred,
            maximum
        );

    if (
        execution_width ==
        0
    ) {

        return selected;
    }

    selected =
        (
            selected /
            execution_width
        ) *
        execution_width;

    if (
        selected ==
        0
    ) {

        selected =
            std::min(
                execution_width,
                maximum
            );
    }

    return selected;
}

} // namespace

BitsetCompactionProbeResult
BitsetCompactionProbe::run() {

    BitsetCompactionProbeResult result;

    result.kernel_name =
        kKernelName;

    result.element_count =
        kElementCount;

    result.bitset_word_count =
        bitset_word_count_for(
            kElementCount
        );

    result.mask_bytes =
        static_cast<std::uint64_t>(
            kElementCount *
            sizeof(std::uint8_t)
        );

    result.bitset_bytes =
        static_cast<std::uint64_t>(
            result.bitset_word_count *
            sizeof(std::uint32_t)
        );

    result.compression_ratio =
        static_cast<double>(
            result.mask_bytes
        ) /
        static_cast<double>(
            result.bitset_bytes
        );

    result.storage_reduction_fraction =
        1.0 -
        (
            static_cast<double>(
                result.bitset_bytes
            ) /
            static_cast<double>(
                result.mask_bytes
            )
        );

    result.first_mismatch_word =
        std::numeric_limits<
            std::size_t
        >::max();

    result.first_mismatch_index =
        std::numeric_limits<
            std::size_t
        >::max();

    @autoreleasepool {

        //
        // 1. Dispositivo Metal.
        //

        id<MTLDevice> device =
            MTLCreateSystemDefaultDevice();

        if (
            device ==
            nil
        ) {

            return result;
        }

        result.metal_available =
            true;

        result.device_name =
            string_from_nsstring(
                device.name
            );

        result.unified_memory =
            device.hasUnifiedMemory;

        //
        // 2. Biblioteca Metal.
        //

        id<MTLLibrary> library =
            [device newDefaultLibrary];

        if (
            library ==
            nil
        ) {

            result.metal_error =
                "No se pudo cargar la biblioteca Metal predeterminada.";

            return result;
        }

        result.library_available =
            true;

        //
        // 3. Kernel.
        //

        NSString* kernel_name =
            [NSString
                stringWithUTF8String:
                    kKernelName];

        id<MTLFunction> function =
            [
                library
                newFunctionWithName:
                    kernel_name
            ];

        if (
            function ==
            nil
        ) {

            result.metal_error =
                "No se encontro el kernel compact_eligibility_to_bitset.";

            return result;
        }

        result.kernel_available =
            true;

        //
        // 4. Pipeline.
        //

        NSError* pipeline_error =
            nil;

        id<MTLComputePipelineState> pipeline =
            [
                device
                newComputePipelineStateWithFunction:
                    function
                error:
                    &pipeline_error
            ];

        if (
            pipeline ==
            nil
        ) {

            if (
                pipeline_error !=
                nil
            ) {

                result.metal_error =
                    string_from_nsstring(
                        pipeline_error
                            .localizedDescription
                    );

            } else {

                result.metal_error =
                    "No se pudo crear MTLComputePipelineState.";
            }

            return result;
        }

        result.pipeline_available =
            true;

        result.thread_execution_width =
            static_cast<std::size_t>(
                pipeline.threadExecutionWidth
            );

        result.max_total_threads_per_threadgroup =
            static_cast<std::size_t>(
                pipeline
                    .maxTotalThreadsPerThreadgroup
            );

        result.threads_per_threadgroup =
            choose_threads_per_threadgroup(
                result.thread_execution_width,
                result
                    .max_total_threads_per_threadgroup
            );

        if (
            result.threads_per_threadgroup ==
            0
        ) {

            result.metal_error =
                "No existe una configuracion valida de threadgroup.";

            return result;
        }

        //
        // 5. Command Queue.
        //

        id<MTLCommandQueue> command_queue =
            [device newCommandQueue];

        if (
            command_queue ==
            nil
        ) {

            result.metal_error =
                "No se pudo crear MTLCommandQueue.";

            return result;
        }

        result.command_queue_available =
            true;

        //
        // 6. Máscara Shared.
        //

        id<MTLBuffer> mask_buffer =
            [
                device
                newBufferWithLength:
                    static_cast<NSUInteger>(
                        result.mask_bytes
                    )
                options:
                    MTLResourceStorageModeShared
            ];

        if (
            mask_buffer ==
                nil ||
            mask_buffer.storageMode !=
                MTLStorageModeShared ||
            mask_buffer.contents ==
                nullptr
        ) {

            result.metal_error =
                "No se pudo crear la mascara Shared.";

            return result;
        }

        result.mask_buffer_available =
            true;

        //
        // 7. Bitset Shared.
        //

        id<MTLBuffer> bitset_buffer =
            [
                device
                newBufferWithLength:
                    static_cast<NSUInteger>(
                        result.bitset_bytes
                    )
                options:
                    MTLResourceStorageModeShared
            ];

        if (
            bitset_buffer ==
                nil ||
            bitset_buffer.storageMode !=
                MTLStorageModeShared ||
            bitset_buffer.contents ==
                nullptr
        ) {

            result.metal_error =
                "No se pudo crear el bitset Shared.";

            return result;
        }

        result.bitset_buffer_available =
            true;

        auto* mask =
            static_cast<std::uint8_t*>(
                mask_buffer.contents
            );

        auto* bitset =
            static_cast<std::uint32_t*>(
                bitset_buffer.contents
            );

        //
        // 8. Inicialización CPU de una máscara dispersa.
        //

        std::size_t cpu_selected_count =
            0;

        for (
            std::size_t index = 0;
            index < kElementCount;
            ++index
        ) {

            const bool eligible =
                deterministic_eligibility(
                    index
                );

            mask[index] =
                eligible
                    ? 1U
                    : 0U;

            if (
                eligible
            ) {

                ++cpu_selected_count;
            }
        }

        result.cpu_selected_count =
            cpu_selected_count;

        //
        // Sentinel para comprobar que GPU escribe cada palabra.
        //

        std::memset(
            bitset,
            0xCD,
            static_cast<std::size_t>(
                result.bitset_bytes
            )
        );

        //
        // 9. Command Buffer.
        //

        id<MTLCommandBuffer> command_buffer =
            [command_queue commandBuffer];

        if (
            command_buffer ==
                nil
        ) {

            result.metal_error =
                "No se pudo crear MTLCommandBuffer.";

            return result;
        }

        result.command_buffer_available =
            true;

        //
        // 10. Encoder.
        //

        id<MTLComputeCommandEncoder> encoder =
            [
                command_buffer
                computeCommandEncoder
            ];

        if (
            encoder ==
                nil
        ) {

            result.metal_error =
                "No se pudo crear MTLComputeCommandEncoder.";

            return result;
        }

        result.encoder_available =
            true;

        [
            encoder
            setComputePipelineState:
                pipeline
        ];

        [
            encoder
            setBuffer:
                mask_buffer
            offset:
                0
            atIndex:
                0
        ];

        [
            encoder
            setBuffer:
                bitset_buffer
            offset:
                0
            atIndex:
                1
        ];

        const std::uint32_t element_count =
            static_cast<std::uint32_t>(
                kElementCount
            );

        [
            encoder
            setBytes:
                &element_count
            length:
                sizeof(element_count)
            atIndex:
                2
        ];

        //
        // Cada hilo produce una palabra uint32_t.
        //

        const MTLSize grid_size =
            MTLSizeMake(
                result.bitset_word_count,
                1,
                1
            );

        const MTLSize threadgroup_size =
            MTLSizeMake(
                result.threads_per_threadgroup,
                1,
                1
            );

        [
            encoder
            dispatchThreads:
                grid_size
            threadsPerThreadgroup:
                threadgroup_size
        ];

        [
            encoder
            endEncoding
        ];

        [
            command_buffer
            commit
        ];

        //
        // Única sincronización GPU → CPU.
        //

        [
            command_buffer
            waitUntilCompleted
        ];

        if (
            command_buffer.status !=
                MTLCommandBufferStatusCompleted
        ) {

            if (
                command_buffer.error !=
                    nil
            ) {

                result.metal_error =
                    string_from_nsstring(
                        command_buffer
                            .error
                            .localizedDescription
                    );

            } else {

                result.metal_error =
                    "El Command Buffer no finalizo correctamente.";
            }

            return result;
        }

        result.command_completed =
            true;

        //
        // 11. Verificación palabra por palabra.
        //

        std::size_t word_mismatch_count =
            0;

        for (
            std::size_t word_index = 0;
            word_index <
                result.bitset_word_count;
            ++word_index
        ) {

            const std::uint32_t expected =
                expected_word(
                    mask,
                    kElementCount,
                    word_index
                );

            const std::uint32_t observed =
                bitset[word_index];

            if (
                expected !=
                observed
            ) {

                if (
                    word_mismatch_count ==
                    0
                ) {

                    result.first_mismatch_word =
                        word_index;

                    result.first_mismatch_expected_word =
                        expected;

                    result.first_mismatch_observed_word =
                        observed;
                }

                ++word_mismatch_count;
            }
        }

        result.word_mismatch_count =
            word_mismatch_count;

        result.exact_word_match =
            word_mismatch_count ==
            0;

        //
        // 12. Verificación semántica bit por bit.
        //
        // Este es exactamente el tipo de lookup O(1) que posteriormente
        // utilizará el predicado CPU consumido por RACORN.
        //

        std::size_t bit_mismatch_count =
            0;

        std::size_t bitset_selected_count =
            0;

        for (
            std::size_t index = 0;
            index < kElementCount;
            ++index
        ) {

            const bool expected =
                mask[index] !=
                0U;

            const bool observed =
                bitset_lookup(
                    bitset,
                    index
                );

            if (
                observed
            ) {

                ++bitset_selected_count;
            }

            if (
                expected !=
                observed
            ) {

                if (
                    bit_mismatch_count ==
                    0
                ) {

                    result.first_mismatch_index =
                        index;

                    result.first_mismatch_expected_bit =
                        expected
                            ? 1U
                            : 0U;

                    result.first_mismatch_observed_bit =
                        observed
                            ? 1U
                            : 0U;
                }

                ++bit_mismatch_count;
            }
        }

        result.bitset_selected_count =
            bitset_selected_count;

        result.bit_mismatch_count =
            bit_mismatch_count;

        result.exact_bit_match =
            bit_mismatch_count ==
                0 &&
            bitset_selected_count ==
                cpu_selected_count;

        result.valid =
            result.metal_available &&
            result.unified_memory &&
            result.library_available &&
            result.kernel_available &&
            result.pipeline_available &&
            result.command_queue_available &&
            result.mask_buffer_available &&
            result.bitset_buffer_available &&
            result.command_buffer_available &&
            result.encoder_available &&
            result.command_completed &&
            result.exact_word_match &&
            result.exact_bit_match;
    }

    return result;
}

} // namespace racorn_uma::platform::apple

//
//  PredicateEngineProbe.mm
//  RACORNUMALab
//
//  Propósito:
//  Implementar el primer Predicate Engine Metal real de RACORNUMALab.
//
//  Gate:
//  G-UMA2 — Predicate Engine Metal.
//
//  Dataset de metadatos:
//  1.048.576 registros con cinco columnas uint32_t:
//
//      tenant_id
//      category
//      year
//      language
//      security_level
//
//  Representación:
//  Structure of Arrays (SoA).
//
//  Las cinco columnas se almacenan dentro de un único MTLBuffer Shared
//  contiguo. Cada columna ocupa element_count * sizeof(uint32_t).
//
//  Predicado:
//      tenant_id == 7
//      AND year >= 2022
//      AND language == 2
//      AND category IN {3, 7, 9}
//      AND security_level <= 2
//
//  CPU:
//  Evalúa el mismo predicado como oráculo.
//
//  GPU:
//  Un hilo evalúa exactamente un registro.
//
//  Salida:
//      uint8_t eligibility[element_count]
//
//  Invariantes:
//  - RACORN no se modifica.
//  - HNSW no se ejecuta.
//  - Sin atomics.
//  - Sin memoria threadgroup.
//  - Un único kernel.
//  - Un único dispatch.
//  - Una única sincronización CPU/GPU.
//  - Igualdad exacta CPU/GPU obligatoria.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Recursos administrados mediante autoreleasepool.
//

#include "platform/apple/PredicateEngineProbe.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace racorn_uma::platform::apple {

namespace {

constexpr char const* kKernelName =
    "evaluate_metadata_predicate";

constexpr std::size_t kElementCount =
    1U << 20U;

constexpr std::size_t kMetadataColumnCount =
    5;

constexpr std::uint64_t kMetadataSeed =
    0xD1B54A32D192ED03ULL;

constexpr std::uint32_t kTenantRequired =
    7U;

constexpr std::uint32_t kMinimumYear =
    2022U;

constexpr std::uint32_t kLanguageRequired =
    2U;

constexpr std::uint32_t kCategoryA =
    3U;

constexpr std::uint32_t kCategoryB =
    7U;

constexpr std::uint32_t kCategoryC =
    9U;

constexpr std::uint32_t kMaximumSecurityLevel =
    2U;

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

std::uint64_t metadata_hash(
    std::size_t index
) noexcept {

    return
        splitmix64(
            kMetadataSeed ^
            static_cast<std::uint64_t>(
                index
            )
        );
}

bool category_allowed(
    std::uint32_t category
) noexcept {

    return
        category ==
            kCategoryA ||
        category ==
            kCategoryB ||
        category ==
            kCategoryC;
}

bool evaluate_cpu_predicate(
    std::uint32_t tenant,
    std::uint32_t category,
    std::uint32_t year,
    std::uint32_t language,
    std::uint32_t security_level
) noexcept {

    return
        tenant ==
            kTenantRequired &&
        year >=
            kMinimumYear &&
        language ==
            kLanguageRequired &&
        category_allowed(
            category
        ) &&
        security_level <=
            kMaximumSecurityLevel;
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
        256;

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

PredicateEngineProbeResult
PredicateEngineProbe::run() {

    PredicateEngineProbeResult result;

    result.kernel_name =
        kKernelName;

    result.element_count =
        kElementCount;

    result.metadata_column_count =
        kMetadataColumnCount;

    const std::size_t column_bytes =
        kElementCount *
        sizeof(std::uint32_t);

    const std::size_t metadata_bytes =
        column_bytes *
        kMetadataColumnCount;

    const std::size_t mask_bytes =
        kElementCount *
        sizeof(std::uint8_t);

    result.metadata_bytes =
        static_cast<std::uint64_t>(
            metadata_bytes
        );

    result.mask_bytes =
        static_cast<std::uint64_t>(
            mask_bytes
        );

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
                "No se encontro el kernel evaluate_metadata_predicate.";

            return result;
        }

        result.kernel_available =
            true;

        //
        // 4. Compute Pipeline.
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
                    "No se pudo crear el Compute Pipeline.";
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
                "El pipeline no admite una configuracion valida de threadgroup.";

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
        // 6. Buffer Shared de metadatos.
        //
        // Un único recurso físico contiene cinco columnas SoA.
        //

        id<MTLBuffer> metadata_buffer =
            [
                device
                newBufferWithLength:
                    metadata_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        if (
            metadata_buffer ==
                nil ||
            metadata_buffer.storageMode !=
                MTLStorageModeShared ||
            metadata_buffer.contents ==
                nullptr
        ) {

            result.metal_error =
                "No se pudo crear el buffer Shared de metadatos.";

            return result;
        }

        result.metadata_buffer_available =
            true;

        //
        // 7. Buffer Shared de salida.
        //

        id<MTLBuffer> mask_buffer =
            [
                device
                newBufferWithLength:
                    mask_bytes
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
                "No se pudo crear el buffer Shared de elegibilidad.";

            return result;
        }

        result.mask_buffer_available =
            true;

        //
        // 8. Layout SoA dentro del buffer.
        //

        auto* metadata =
            static_cast<std::uint32_t*>(
                metadata_buffer.contents
            );

        std::uint32_t* tenant =
            metadata +
            (
                0 *
                kElementCount
            );

        std::uint32_t* category =
            metadata +
            (
                1 *
                kElementCount
            );

        std::uint32_t* year =
            metadata +
            (
                2 *
                kElementCount
            );

        std::uint32_t* language =
            metadata +
            (
                3 *
                kElementCount
            );

        std::uint32_t* security_level =
            metadata +
            (
                4 *
                kElementCount
            );

        //
        // 9. Inicialización CPU de metadatos.
        //
        // Las distribuciones son deliberadamente simples y reproducibles.
        //

        std::size_t cpu_selected_count =
            0;

        for (
            std::size_t index = 0;
            index < kElementCount;
            ++index
        ) {

            const std::uint64_t hash =
                metadata_hash(
                    index
                );

            const std::uint32_t tenant_value =
                static_cast<std::uint32_t>(
                    hash &
                    0x0FU
                );

            const std::uint32_t category_value =
                static_cast<std::uint32_t>(
                    (
                        hash >>
                        8U
                    ) &
                    0x0FU
                );

            const std::uint32_t year_value =
                2019U +
                static_cast<std::uint32_t>(
                    (
                        hash >>
                        16U
                    ) %
                    8U
                );

            const std::uint32_t language_value =
                static_cast<std::uint32_t>(
                    (
                        hash >>
                        24U
                    ) &
                    0x03U
                );

            const std::uint32_t security_value =
                static_cast<std::uint32_t>(
                    (
                        hash >>
                        32U
                    ) %
                    6U
                );

            tenant[index] =
                tenant_value;

            category[index] =
                category_value;

            year[index] =
                year_value;

            language[index] =
                language_value;

            security_level[index] =
                security_value;

            if (
                evaluate_cpu_predicate(
                    tenant_value,
                    category_value,
                    year_value,
                    language_value,
                    security_value
                )
            ) {

                ++cpu_selected_count;
            }
        }

        result.cpu_selected_count =
            cpu_selected_count;

        //
        // Sentinel deliberado.
        //
        // Un resultado válido de GPU solo puede ser 0 o 1.
        //

        auto* gpu_mask =
            static_cast<std::uint8_t*>(
                mask_buffer.contents
            );

        std::memset(
            gpu_mask,
            0xCD,
            mask_bytes
        );

        //
        // 10. Command Buffer.
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
        // 11. Compute Encoder.
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

        //
        // Cada columna es un rango diferente del mismo MTLBuffer Shared.
        //

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                0 * column_bytes
            atIndex:
                0
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                1 * column_bytes
            atIndex:
                1
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                2 * column_bytes
            atIndex:
                2
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                3 * column_bytes
            atIndex:
                3
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                4 * column_bytes
            atIndex:
                4
        ];

        [
            encoder
            setBuffer:
                mask_buffer
            offset:
                0
            atIndex:
                5
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
                6
        ];

        //
        // 12. Dispatch.
        //
        // Un hilo Metal por registro de metadatos.
        //

        const MTLSize grid_size =
            MTLSizeMake(
                kElementCount,
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
        // Única frontera explícita GPU -> CPU.
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
        // 13. Verificación exacta CPU vs GPU.
        //

        std::size_t gpu_selected_count =
            0;

        std::size_t mismatch_count =
            0;

        std::size_t invalid_value_count =
            0;

        for (
            std::size_t index = 0;
            index < kElementCount;
            ++index
        ) {

            const bool cpu_accepts =
                evaluate_cpu_predicate(
                    tenant[index],
                    category[index],
                    year[index],
                    language[index],
                    security_level[index]
                );

            const std::uint8_t expected =
                cpu_accepts
                    ? 1U
                    : 0U;

            const std::uint8_t observed =
                gpu_mask[index];

            if (
                observed >
                1U
            ) {

                ++invalid_value_count;
            }

            if (
                observed ==
                1U
            ) {

                ++gpu_selected_count;
            }

            if (
                observed !=
                expected
            ) {

                if (
                    mismatch_count ==
                    0
                ) {

                    result.first_mismatch_index =
                        index;

                    result.first_mismatch_expected =
                        expected;

                    result.first_mismatch_observed =
                        observed;
                }

                ++mismatch_count;
            }
        }

        result.gpu_selected_count =
            gpu_selected_count;

        result.mismatch_count =
            mismatch_count;

        result.invalid_mask_value_count =
            invalid_value_count;

        result.observed_selectivity =
            static_cast<double>(
                cpu_selected_count
            ) /
            static_cast<double>(
                kElementCount
            );

        result.exact_match =
            mismatch_count ==
                0 &&
            invalid_value_count ==
                0 &&
            cpu_selected_count ==
                gpu_selected_count;

        result.valid =
            result.metal_available &&
            result.unified_memory &&
            result.library_available &&
            result.kernel_available &&
            result.pipeline_available &&
            result.command_queue_available &&
            result.metadata_buffer_available &&
            result.mask_buffer_available &&
            result.command_buffer_available &&
            result.encoder_available &&
            result.command_completed &&
            result.exact_match;
    }

    return result;
}

} // namespace racorn_uma::platform::apple

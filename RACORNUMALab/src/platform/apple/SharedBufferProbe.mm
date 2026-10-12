//
//  SharedBufferProbe.mm
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar el primer kernel Metal real de RACORNUMALab utilizando
//  un único MTLBuffer Shared visible tanto para CPU como para GPU.
//
//  Gate:
//  G-UMA1 — CPU → UMA Shared → GPU → UMA Shared → CPU.
//
//  Diseño:
//  1. CPU crea un MTLBuffer Shared.
//  2. CPU inicializa 1.048.576 valores uint32_t.
//  3. GPU lee y modifica esos mismos valores.
//  4. CPU espera exactamente una vez la finalización del command buffer.
//  5. CPU verifica todos los elementos contra un oráculo determinista.
//
//  Transformación:
//      output = (input XOR 0xA5A5A5A5) + 0x12345678
//
//  La aritmética uint32_t utiliza wrap-around deliberadamente.
//
//  Justificación:
//  La transformación no tiene valor algorítmico para RACORN.
//  Su función es demostrar:
//
//      - ejecución compute Metal real;
//      - lectura GPU de memoria inicializada por CPU;
//      - escritura GPU sobre almacenamiento Shared;
//      - observación CPU del resultado;
//      - equivalencia bit-exacta;
//      - ausencia de copias explícitas CPU↔VRAM.
//
//  Invariantes:
//  - No se modifica RACORN.
//  - No se ejecuta HNSW.
//  - No se utilizan heaps.
//  - No se ejecutan predicados.
//  - No hay benchmark.
//  - No se utilizan buffers Private.
//  - Un único dispatch.
//  - Una única espera CPU tras commit.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Recursos locales administrados dentro de autoreleasepool.
//

#include "platform/apple/SharedBufferProbe.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace racorn_uma::platform::apple {

namespace {

constexpr char const* kKernelName =
    "uma_transform_in_place";

constexpr std::size_t kElementCount =
    1U << 20U;

constexpr std::uint32_t kInputMultiplier =
    0x9E3779B9U;

constexpr std::uint32_t kInputXor =
    0xC001D00DU;

constexpr std::uint32_t kTransformXor =
    0xA5A5A5A5U;

constexpr std::uint32_t kTransformAdd =
    0x12345678U;

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

std::uint32_t initial_value(
    std::size_t index
) noexcept {

    const auto index32 =
        static_cast<std::uint32_t>(
            index
        );

    return
        (
            index32 *
            kInputMultiplier
        ) ^
        kInputXor;
}

std::uint32_t expected_value(
    std::size_t index
) noexcept {

    const std::uint32_t input =
        initial_value(
            index
        );

    return
        (
            input ^
            kTransformXor
        ) +
        kTransformAdd;
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

SharedBufferProbeResult
SharedBufferProbe::run() {

    SharedBufferProbeResult result;

    result.kernel_name =
        kKernelName;

    result.element_count =
        kElementCount;

    result.buffer_bytes =
        static_cast<std::uint64_t>(
            kElementCount
        ) *
        sizeof(std::uint32_t);

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
        // 2. Biblioteca Metal compilada por Xcode.
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
        // 3. Función compute.
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
                "No se encontro el kernel uma_transform_in_place.";

            return result;
        }

        result.kernel_available =
            true;

        //
        // 4. Compute Pipeline State.
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
                "El pipeline reporto cero hilos utilizables por threadgroup.";

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
        // 6. Buffer UMA Shared.
        //
        // Es el único almacenamiento del vector de prueba.
        // No existe una copia CPU y otra GPU.
        //

        id<MTLBuffer> buffer =
            [
                device
                newBufferWithLength:
                    static_cast<NSUInteger>(
                        result.buffer_bytes
                    )
                options:
                    MTLResourceStorageModeShared
            ];

        if (
            buffer ==
            nil ||
            buffer.storageMode !=
                MTLStorageModeShared ||
            buffer.contents ==
                nullptr
        ) {

            result.metal_error =
                "No se pudo crear un MTLBuffer Shared mapeado en CPU.";

            return result;
        }

        result.shared_buffer_available =
            true;

        //
        // 7. Inicialización CPU.
        //

        auto* values =
            static_cast<std::uint32_t*>(
                buffer.contents
            );

        for (
            std::size_t index = 0;
            index < kElementCount;
            ++index
        ) {

            values[index] =
                initial_value(
                    index
                );
        }

        //
        // 8. Command Buffer.
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
        // 9. Compute Encoder.
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
                buffer
            offset:
                0
            atIndex:
                0
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
                1
        ];

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

        //
        // Un único dispatch grande y regular.
        //

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

        //
        // 10. Submit GPU.
        //

        [
            command_buffer
            commit
        ];

        //
        // Única sincronización explícita del Gate.
        //
        // La CPU no toca el buffer entre commit y completion.
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
                    "El command buffer no finalizo con estado Completed.";
            }

            return result;
        }

        result.command_completed =
            true;

        //
        // 11. Verificación CPU bit-exacta.
        //
        // La CPU vuelve a leer exactamente el mismo MTLBuffer Shared
        // que acaba de modificar la GPU.
        //

        std::size_t mismatch_count =
            0;

        for (
            std::size_t index = 0;
            index < kElementCount;
            ++index
        ) {

            const std::uint32_t expected =
                expected_value(
                    index
                );

            const std::uint32_t observed =
                values[index];

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

        result.mismatch_count =
            mismatch_count;

        result.exact_match =
            mismatch_count ==
            0;

        result.valid =
            result.metal_available &&
            result.unified_memory &&
            result.library_available &&
            result.kernel_available &&
            result.pipeline_available &&
            result.command_queue_available &&
            result.shared_buffer_available &&
            result.command_buffer_available &&
            result.encoder_available &&
            result.command_completed &&
            result.exact_match;
    }

    return result;
}

} // namespace racorn_uma::platform::apple

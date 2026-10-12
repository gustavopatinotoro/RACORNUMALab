//
//  MetalInfo.mm
//  RACORNUMALab
//
//  Propósito:
//  Implementar la frontera Objective-C++ entre RACORNUMALab y Metal.
//
//  Gate:
//  G-UMA0 — Caracterización Metal / memoria unificada.
//
//  Responsabilidades:
//  - Obtener el MTLDevice predeterminado.
//  - Consultar propiedades relevantes para ejecución heterogénea.
//  - Verificar creación de una MTLCommandQueue.
//  - Verificar asignación de un MTLBuffer Shared.
//  - Copiar los resultados a estructuras C++ sin exponer objetos Metal.
//
//  Diseño:
//  El resto del proyecto permanece en C++20. Objective-C++ se limita
//  deliberadamente a la capa:
//
//      platform/apple
//
//  Esto evita introducir tipos Objective-C dentro del motor RACORN.
//
//  Invariantes:
//  - No se ejecutan compute kernels.
//  - No se modifica RACORN.
//  - No se accede al grafo HNSW.
//  - No se realizan benchmarks.
//  - El buffer de prueba tiene únicamente 4 KiB.
//
//  Contrato G-UMA0:
//  Para un Apple Silicon M1 válido esperamos:
//
//      metal_available           = true
//      unified_memory            = true
//      command_queue_available   = true
//      shared_buffer_available   = true
//
//  Nota:
//  recommendedMaxWorkingSetSize es una recomendación del dispositivo,
//  no una autorización para consumir esa cantidad completa de memoria.
//  El laboratorio continuará imponiendo sus propias restricciones
//  conservadoras para el MacBook Air M1 de 8 GiB.
//
//  Seguridad:
//  - Sin red.
//  - Sin sistema de archivos.
//  - Recursos Metal gestionados por ARC/autoreleasepool.
//

#include "platform/apple/MetalInfo.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace racorn_uma::platform::apple {

namespace {

constexpr std::size_t kSharedProbeBufferLength =
    4U * 1024U;

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

} // namespace

MetalDeviceSnapshot
MetalInfo::capture() {

    MetalDeviceSnapshot snapshot;

    @autoreleasepool {

        id<MTLDevice> device =
            MTLCreateSystemDefaultDevice();

        if (
            device ==
            nil
        ) {

            return snapshot;
        }

        snapshot.metal_available =
            true;

        snapshot.device_name =
            string_from_nsstring(
                device.name
            );

        snapshot.registry_id =
            static_cast<std::uint64_t>(
                device.registryID
            );

        snapshot.unified_memory =
            device.hasUnifiedMemory;

        snapshot.low_power =
            device.lowPower;

        snapshot.removable =
            device.removable;

        snapshot.headless =
            device.headless;

        snapshot.recommended_max_working_set_size =
            static_cast<std::uint64_t>(
                device.recommendedMaxWorkingSetSize
            );

        snapshot.current_allocated_size_before_probe =
            static_cast<std::uint64_t>(
                device.currentAllocatedSize
            );

        snapshot.max_buffer_length =
            static_cast<std::uint64_t>(
                device.maxBufferLength
            );

        snapshot.max_threadgroup_memory_length =
            static_cast<std::uint64_t>(
                device.maxThreadgroupMemoryLength
            );

        const MTLSize max_threads =
            device.maxThreadsPerThreadgroup;

        snapshot.max_threads_per_threadgroup_width =
            static_cast<std::uint64_t>(
                max_threads.width
            );

        snapshot.max_threads_per_threadgroup_height =
            static_cast<std::uint64_t>(
                max_threads.height
            );

        snapshot.max_threads_per_threadgroup_depth =
            static_cast<std::uint64_t>(
                max_threads.depth
            );

        //
        // Verificación de infraestructura de comandos.
        //
        // No se envía trabajo a GPU en G-UMA0.
        //

        id<MTLCommandQueue> command_queue =
            [device newCommandQueue];

        snapshot.command_queue_available =
            command_queue != nil;

        //
        // Verificación explícita de almacenamiento Shared.
        //
        // El buffer es deliberadamente pequeño. Aquí solo comprobamos:
        //
        // 1. que la asignación es válida;
        // 2. que Metal informa MTLStorageModeShared;
        // 3. que existe mapeo CPU mediante contents.
        //
        // La escritura GPU→CPU se probará posteriormente en G-UMA1.
        //

        snapshot.shared_probe_buffer_length =
            static_cast<std::uint64_t>(
                kSharedProbeBufferLength
            );

        id<MTLBuffer> shared_buffer =
            [
                device
                newBufferWithLength:
                    kSharedProbeBufferLength
                options:
                    MTLResourceStorageModeShared
            ];

        if (
            shared_buffer != nil
        ) {

            const bool storage_is_shared =
                shared_buffer.storageMode ==
                MTLStorageModeShared;

            const bool cpu_mapping_exists =
                shared_buffer.contents !=
                nullptr;

            snapshot.shared_buffer_available =
                storage_is_shared &&
                cpu_mapping_exists;
        }

        snapshot.current_allocated_size_after_probe =
            static_cast<std::uint64_t>(
                device.currentAllocatedSize
            );

        //
        // Condición estricta del Gate para Apple Silicon.
        //

        snapshot.valid =
            snapshot.metal_available &&
            snapshot.unified_memory &&
            snapshot.command_queue_available &&
            snapshot.shared_buffer_available &&
            snapshot.max_buffer_length >=
                kSharedProbeBufferLength &&
            snapshot.recommended_max_working_set_size >
                0;
    }

    return snapshot;
}

} // namespace racorn_uma::platform::apple

//
//  SharedBufferProbe.metal
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar la primera transformación compute real sobre un MTLBuffer
//  Shared utilizado directamente por CPU y GPU.
//
//  Gate:
//  G-UMA1 — CPU → UMA Shared → GPU → UMA Shared → CPU.
//
//  Transformación:
//      output = (input XOR 0xA5A5A5A5) + 0x12345678
//
//  Características:
//  - Un hilo procesa exactamente un elemento.
//  - Sin atomics.
//  - Sin memoria threadgroup.
//  - Sin divergencia salvo la protección de límites.
//  - Acceso lineal y coalescente.
//  - Un único buffer de datos.
//
//  Este kernel NO implementa ninguna parte de RACORN.
//

#include <metal_stdlib>

using namespace metal;

kernel void uma_transform_in_place(
    device uint* values [[buffer(0)]],
    constant uint& element_count [[buffer(1)]],
    uint global_id [[thread_position_in_grid]]
) {

    if (
        global_id >=
        element_count
    ) {

        return;
    }

    const uint input =
        values[global_id];

    values[global_id] =
        (
            input ^
            0xA5A5A5A5u
        ) +
        0x12345678u;
}

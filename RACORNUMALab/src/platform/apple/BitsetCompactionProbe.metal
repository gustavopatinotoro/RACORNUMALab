//
//  BitsetCompactionProbe.metal
//  RACORNUMALab
//
//  Propósito:
//  Compactar una máscara de elegibilidad de 1 byte por vector en un
//  bitset de 1 bit por vector.
//
//  Gate:
//  G-UMA3 — Compactación GPU a bitset UMA.
//
//  Mapeo:
//
//      1 hilo GPU
//          ↓
//      lee hasta 32 bytes
//          ↓
//      produce 1 uint32_t
//
//  Convención:
//
//      bitset[index / 32]
//      bit = index % 32
//
//  Diseño:
//  - Sin atomics.
//  - Sin memoria threadgroup.
//  - Sin barreras.
//  - Escritura exclusiva: cada hilo escribe una palabra diferente.
//  - Accesos de entrada contiguos.
//  - Salida compacta y directamente consultable por CPU.
//
//  RACORN:
//  Este kernel no contiene lógica HNSW ni RACORN.
//

#include <metal_stdlib>

using namespace metal;

kernel void compact_eligibility_to_bitset(
    device const uchar* eligibility [[buffer(0)]],
    device uint* bitset [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint word_index [[thread_position_in_grid]]
) {

    const uint base_index =
        word_index *
        32u;

    if (
        base_index >=
        element_count
    ) {

        return;
    }

    uint packed_word =
        0u;

    for (
        uint bit_index = 0u;
        bit_index < 32u;
        ++bit_index
    ) {

        const uint element_index =
            base_index +
            bit_index;

        if (
            element_index >=
            element_count
        ) {

            break;
        }

        if (
            eligibility[element_index] !=
            uchar(0)
        ) {

            packed_word |=
                (
                    1u <<
                    bit_index
                );
        }
    }

    bitset[word_index] =
        packed_word;
}

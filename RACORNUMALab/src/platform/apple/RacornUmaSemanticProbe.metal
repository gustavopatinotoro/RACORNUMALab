//
//  RacornUmaSemanticProbe.metal
//  RACORNUMALab
//
//  Propósito:
//  Evaluar directamente un predicado compuesto de metadatos y producir
//  el bitset final que será consumido posteriormente por RACORN CPU.
//
//  Gate:
//  G-UMA4 — RACORN CPU consumiendo bitset GPU/UMA.
//
//  Diseño final:
//      32 registros
//          ↓
//      1 hilo GPU
//          ↓
//      1 palabra uint32_t
//
//  Esto fusiona:
//
//      evaluación del predicado
//              +
//      compactación de elegibilidad
//
//  evitando:
//
//  - máscara intermedia de 1 byte/vector;
//  - segundo dispatch;
//  - atomics;
//  - barreras;
//  - memoria threadgroup.
//
//  Predicado:
//      tenant_id == 7
//      AND year >= 2022
//      AND language == 2
//      AND category IN {3,7,9}
//      AND security_level <= 2
//
//  La GPU NO implementa navegación RACORN.
//
//  Convención:
//      bitset[id >> 5]
//      bit = id & 31
//

#include <metal_stdlib>

using namespace metal;

kernel void evaluate_metadata_predicate_to_bitset(
    device const uint* tenant_id [[buffer(0)]],
    device const uint* category [[buffer(1)]],
    device const uint* year [[buffer(2)]],
    device const uint* language [[buffer(3)]],
    device const uint* security_level [[buffer(4)]],
    device uint* eligibility_bitset [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    uint word_index [[thread_position_in_grid]]
) {

    constexpr uint bits_per_word =
        32u;

    const uint base_index =
        word_index *
        bits_per_word;

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
        bit_index < bits_per_word;
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

        const uint tenant_value =
            tenant_id[element_index];

        const uint category_value =
            category[element_index];

        const uint year_value =
            year[element_index];

        const uint language_value =
            language[element_index];

        const uint security_value =
            security_level[element_index];

        const bool category_allowed =
            category_value == 3u ||
            category_value == 7u ||
            category_value == 9u;

        const bool accepted =
            tenant_value == 7u &&
            year_value >= 2022u &&
            language_value == 2u &&
            category_allowed &&
            security_value <= 2u;

        if (
            accepted
        ) {

            packed_word |=
                (
                    1u <<
                    bit_index
                );
        }
    }

    eligibility_bitset[word_index] =
        packed_word;
}

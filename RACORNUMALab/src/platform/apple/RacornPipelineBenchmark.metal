//
//  RacornPipelineBenchmark.metal
//  RACORNUMALab
//
//  Propósito:
//  Materializar asíncronamente el bitset de la siguiente consulta
//  mientras RACORN navega la consulta actual.
//
//  Gate:
//  G-UMA5C — Pipeline CPU/GPU.
//
//  Predicado:
//
//      tenant_id == tenant_required
//      AND year >= 2022
//      AND language >= 2
//      AND category IN {3,7,9}
//      AND security_level <= 2
//
//  Mapeo:
//
//      32 registros
//          ↓
//      1 hilo GPU
//          ↓
//      1 uint32_t
//
//  Sin atomics.
//  Sin memoria threadgroup.
//  Sin barreras.
//  Sin navegación RACORN.
//

#include <metal_stdlib>

using namespace metal;

kernel void racorn_pipeline_predicate_to_bitset(
    device const uint* tenant_id [[buffer(0)]],
    device const uint* category [[buffer(1)]],
    device const uint* year [[buffer(2)]],
    device const uint* language [[buffer(3)]],
    device const uint* security_level [[buffer(4)]],
    device uint* eligibility_bitset [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    constant uint& tenant_required [[buffer(7)]],
    uint word_index [[thread_position_in_grid]]
) {

    constexpr uint bits_per_word = 32u;

    const uint base_index =
        word_index * bits_per_word;

    if (
        base_index >= element_count
    ) {

        return;
    }

    uint packed_word = 0u;

    for (
        uint bit_index = 0u;
        bit_index < bits_per_word;
        ++bit_index
    ) {

        const uint element_index =
            base_index + bit_index;

        if (
            element_index >= element_count
        ) {

            break;
        }

        const uint category_value =
            category[element_index];

        const bool category_allowed =
            category_value == 3u ||
            category_value == 7u ||
            category_value == 9u;

        const bool accepted =
            tenant_id[element_index] ==
                tenant_required &&

            year[element_index] >=
                2022u &&

            language[element_index] >=
                2u &&

            category_allowed &&

            security_level[element_index] <=
                2u;

        if (
            accepted
        ) {

            packed_word |=
                (
                    1u << bit_index
                );
        }
    }

    eligibility_bitset[word_index] =
        packed_word;
}

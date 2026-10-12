//
//  PredicateBreakEvenBenchmark.metal
//  RACORNUMALab
//
//  Propósito:
//  Implementar el kernel Metal utilizado para medir el punto de
//  equilibrio CPU/GPU en la generación de bitsets de elegibilidad.
//
//  Gate:
//  G-UMA5A — Punto de equilibrio CPU/GPU.
//
//  Diseño:
//
//      32 registros
//          ↓
//      un hilo GPU
//          ↓
//      una palabra uint32
//
//  Esto evita:
//
//  - atomics;
//  - máscara byte/vector;
//  - segundo kernel;
//  - segundo dispatch;
//  - barreras;
//  - memoria threadgroup.
//
//  Modos:
//
//      mode = 1:
//
//          tenant == 7
//
//      mode = 3:
//
//          tenant == 7
//          AND year >= 2022
//          AND language == 2
//
//      mode = 5:
//
//          mode 3
//          AND category IN {3,7,9}
//          AND security_level <= 2
//
//  El modo es constante para todo el dispatch.
//
//  RACORN:
//  Este kernel no contiene ninguna lógica de navegación.
//

#include <metal_stdlib>

using namespace metal;

kernel void benchmark_predicate_to_bitset(
    device const uint* tenant_id [[buffer(0)]],
    device const uint* category [[buffer(1)]],
    device const uint* year [[buffer(2)]],
    device const uint* language [[buffer(3)]],
    device const uint* security_level [[buffer(4)]],
    device uint* eligibility_bitset [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    constant uint& predicate_mode [[buffer(7)]],
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

        bool accepted =
            tenant_value == 7u;

        if (
            predicate_mode >= 3u
        ) {

            accepted =
                accepted &&
                year[element_index] >= 2022u &&
                language[element_index] == 2u;
        }

        if (
            predicate_mode >= 5u
        ) {

            const uint category_value =
                category[element_index];

            const bool category_allowed =
                category_value == 3u ||
                category_value == 7u ||
                category_value == 9u;

            accepted =
                accepted &&
                category_allowed &&
                security_level[element_index] <= 2u;
        }

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

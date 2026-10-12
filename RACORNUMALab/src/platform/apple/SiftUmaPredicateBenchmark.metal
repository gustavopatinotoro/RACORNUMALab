//
//  SiftUmaPredicateBenchmark.metal
//  RACORNUMALab
//
//  Propósito:
//  Materializar en GPU un bitset de elegibilidad para G-DATA3A.
//
//  Cada thread produce exactamente una palabra uint32_t del bitset:
//
//      thread i -> IDs [32*i, 32*i + 31]
//
//  Predicado:
//
//      tenant == tenant_required
//      year >= 2022
//      language >= language_min
//      category IN {3, 7, 9}
//      security_level <= 2
//
//  Invariantes:
//  - No calcula distancias.
//  - No navega HNSW.
//  - No modifica RACORN.
//  - Es una transformación pura metadata -> bitset.
//  - Entrada y salida se alojan en memoria Shared.
//

#include <metal_stdlib>

using namespace metal;

kernel void racorn_sift_predicate_to_bitset(
    device const uint* tenant          [[buffer(0)]],
    device const uint* category        [[buffer(1)]],
    device const uint* year            [[buffer(2)]],
    device const uint* language        [[buffer(3)]],
    device const uint* security_level  [[buffer(4)]],
    device uint* output_bitset         [[buffer(5)]],

    constant uint& element_count       [[buffer(6)]],
    constant uint& tenant_required     [[buffer(7)]],
    constant uint& language_min        [[buffer(8)]],

    uint word_index                    [[thread_position_in_grid]]
) {

    const uint base =
        word_index * 32U;

    if (
        base >=
        element_count
    ) {
        return;
    }

    uint word =
        0U;

    for (
        uint bit = 0U;
        bit < 32U;
        ++bit
    ) {

        const uint key =
            base +
            bit;

        if (
            key >=
            element_count
        ) {
            break;
        }

        const uint current_category =
            category[key];

        const bool category_allowed =
            current_category == 3U ||
            current_category == 7U ||
            current_category == 9U;

        const bool accepted =
            tenant[key] ==
                tenant_required &&

            year[key] >=
                2022U &&

            language[key] >=
                language_min &&

            category_allowed &&

            security_level[key] <=
                2U;

        if (
            accepted
        ) {

            word |=
                (
                    1U <<
                    bit
                );
        }
    }

    output_bitset[
        word_index
    ] =
        word;
}

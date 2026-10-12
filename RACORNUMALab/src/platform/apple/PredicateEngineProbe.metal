//
//  PredicateEngineProbe.metal
//  RACORNUMALab
//
//  Propósito:
//  Evaluar en GPU un predicado compuesto de metadatos sobre una
//  representación columnar SoA.
//
//  Gate:
//  G-UMA2 — Predicate Engine Metal.
//
//  Mapeo:
//      un hilo GPU <-> un registro de metadatos.
//
//  Predicado:
//      tenant_id == 7
//      AND year >= 2022
//      AND language == 2
//      AND category IN {3, 7, 9}
//      AND security_level <= 2
//
//  Salida:
//      eligibility[i] = 0 o 1.
//
//  Diseño GPU:
//  - Acceso lineal.
//  - Trabajo independiente.
//  - Sin atomics.
//  - Sin memoria threadgroup.
//  - Sin barreras.
//  - Sin pointer chasing.
//  - Sin navegación HNSW.
//  - Sin lógica RACORN.
//
//  Este kernel es deliberadamente una operación regular y paralela.
//

#include <metal_stdlib>

using namespace metal;

kernel void evaluate_metadata_predicate(
    device const uint* tenant_id [[buffer(0)]],
    device const uint* category [[buffer(1)]],
    device const uint* year [[buffer(2)]],
    device const uint* language [[buffer(3)]],
    device const uint* security_level [[buffer(4)]],
    device uchar* eligibility [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    uint global_id [[thread_position_in_grid]]
) {

    if (
        global_id >=
        element_count
    ) {

        return;
    }

    const uint tenant_value =
        tenant_id[global_id];

    const uint category_value =
        category[global_id];

    const uint year_value =
        year[global_id];

    const uint language_value =
        language[global_id];

    const uint security_value =
        security_level[global_id];

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

    eligibility[global_id] =
        accepted
            ? uchar(1)
            : uchar(0);
}

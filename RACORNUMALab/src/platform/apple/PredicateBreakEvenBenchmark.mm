//
//  PredicateBreakEvenBenchmark.mm
//  RACORNUMALab
//
//  Propósito:
//  Medir el punto de equilibrio CPU/GPU para la generación de bitsets
//  de elegibilidad sobre Apple Silicon con memoria unificada.
//
//  Gate:
//  G-UMA5A — Punto de equilibrio CPU/GPU.
//
//  Metodología:
//
//  1. Se asigna metadata Shared una sola vez.
//  2. Se inicializan 1.048.576 registros.
//  3. Se crea el pipeline Metal una sola vez.
//  4. Se evalúan cinco tamaños y tres complejidades.
//  5. Para cada caso se verifica primero CPU == GPU.
//  6. Se realizan warmups.
//  7. Se realizan mediciones alternando el orden CPU/GPU.
//
//  Timing CPU:
//      steady_clock alrededor exclusivamente de la materialización.
//
//  Timing GPU end-to-end:
//      commandBuffer
//          + encoder
//          + set arguments
//          + dispatch
//          + commit
//          + waitUntilCompleted
//
//  Timing GPU device:
//      GPUEndTime - GPUStartTime
//
//  Este último se consulta exclusivamente después de que el command
//  buffer termina.
//
//  Razón para medir ambos tiempos:
//
//      GPU device rápido
//
//  no implica necesariamente:
//
//      query GPU rápida
//
//  porque encoding, scheduling y sincronización forman parte del coste
//  real cuando RACORN necesita consumir inmediatamente el bitset.
//
//  Invariantes:
//  - RACORN no se ejecuta.
//  - No se asignan buffers dentro del benchmark.
//  - No se recompila pipeline dentro del benchmark.
//  - Un único dispatch GPU por materialización.
//  - Una única sincronización GPU→CPU por materialización.
//  - Cero atomics.
//  - Cero barreras.
//  - Igualdad bit-exacta obligatoria.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Sin persistencia.
//

#include "platform/apple/PredicateBreakEvenBenchmark.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace racorn_uma::platform::apple {

namespace {

using Clock =
    std::chrono::steady_clock;

constexpr char const* kKernelName =
    "benchmark_predicate_to_bitset";

constexpr std::size_t kMaxElementCount =
    1U << 20U;

constexpr std::size_t kMetadataColumnCount =
    5U;

constexpr std::size_t kBitsPerWord =
    32U;

constexpr std::size_t kWarmupIterations =
    5U;

constexpr std::size_t kMeasuredIterations =
    50U;

constexpr std::array<std::size_t, 5>
    kElementCounts{
        20'000U,
        100'000U,
        250'000U,
        500'000U,
        1U << 20U
    };

constexpr std::array<std::uint32_t, 3>
    kPredicateModes{
        1U,
        3U,
        5U
    };

constexpr std::uint64_t kMetadataSeed =
    0xD1B54A32D192ED03ULL;

struct MetadataView final {

    std::uint32_t const* tenant{};
    std::uint32_t const* category{};
    std::uint32_t const* year{};
    std::uint32_t const* language{};
    std::uint32_t const* security_level{};
};

struct GpuTimingSample final {

    bool completed{};

    bool device_timing_available{};

    double wall_ms{};

    double device_ms{};

    std::string error;
};

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

std::size_t bitset_word_count_for(
    std::size_t element_count
) noexcept {

    return
        (
            element_count +
            kBitsPerWord -
            1U
        ) /
        kBitsPerWord;
}

bool evaluate_predicate(
    MetadataView const& metadata,
    std::size_t index,
    std::uint32_t mode
) noexcept {

    bool accepted =
        metadata.tenant[index] ==
        7U;

    if (
        mode >=
        3U
    ) {

        accepted =
            accepted &&
            metadata.year[index] >=
                2022U &&
            metadata.language[index] ==
                2U;
    }

    if (
        mode >=
        5U
    ) {

        const std::uint32_t category =
            metadata.category[index];

        const bool category_allowed =
            category ==
                3U ||
            category ==
                7U ||
            category ==
                9U;

        accepted =
            accepted &&
            category_allowed &&
            metadata.security_level[index] <=
                2U;
    }

    return accepted;
}

#if defined(__clang__)
__attribute__((noinline))
#endif
void materialize_cpu_bitset(
    MetadataView const& metadata,
    std::size_t element_count,
    std::uint32_t predicate_mode,
    std::uint32_t* output
) noexcept {

    const std::size_t word_count =
        bitset_word_count_for(
            element_count
        );

    for (
        std::size_t word_index = 0;
        word_index < word_count;
        ++word_index
    ) {

        const std::size_t base_index =
            word_index *
            kBitsPerWord;

        std::uint32_t word =
            0U;

        for (
            std::size_t bit_index = 0;
            bit_index <
                kBitsPerWord;
            ++bit_index
        ) {

            const std::size_t element_index =
                base_index +
                bit_index;

            if (
                element_index >=
                element_count
            ) {

                break;
            }

            if (
                evaluate_predicate(
                    metadata,
                    element_index,
                    predicate_mode
                )
            ) {

                word |=
                    (
                        std::uint32_t{1U}
                        <<
                        bit_index
                    );
            }
        }

        output[word_index] =
            word;
    }
}

std::size_t selected_count_from_bitset(
    std::uint32_t const* bitset,
    std::size_t word_count
) noexcept {

    std::size_t count =
        0;

    for (
        std::size_t index = 0;
        index < word_count;
        ++index
    ) {

        count +=
            static_cast<std::size_t>(
                __builtin_popcount(
                    bitset[index]
                )
            );
    }

    return count;
}

std::size_t mismatch_word_count(
    std::uint32_t const* first,
    std::uint32_t const* second,
    std::size_t word_count
) noexcept {

    std::size_t mismatches =
        0;

    for (
        std::size_t index = 0;
        index < word_count;
        ++index
    ) {

        if (
            first[index] !=
            second[index]
        ) {

            ++mismatches;
        }
    }

    return mismatches;
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
        256U;

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

double milliseconds_between(
    Clock::time_point start,
    Clock::time_point end
) noexcept {

    return
        std::chrono::duration<
            double,
            std::milli
        >(
            end -
            start
        ).count();
}

double percentile(
    std::vector<double> values,
    double probability
) {

    if (
        values.empty()
    ) {

        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end()
    );

    if (
        values.size() ==
        1
    ) {

        return values.front();
    }

    const double position =
        probability *
        static_cast<double>(
            values.size() -
            1U
        );

    const std::size_t lower =
        static_cast<std::size_t>(
            std::floor(
                position
            )
        );

    const std::size_t upper =
        static_cast<std::size_t>(
            std::ceil(
                position
            )
        );

    if (
        lower ==
        upper
    ) {

        return
            values[lower];
    }

    const double fraction =
        position -
        static_cast<double>(
            lower
        );

    return
        values[lower] *
            (
                1.0 -
                fraction
            ) +
        values[upper] *
            fraction;
}

std::string predicate_name(
    std::uint32_t mode
) {

    switch (
        mode
    ) {

    case 1U:
        return
            "P1";

    case 3U:
        return
            "P3";

    case 5U:
        return
            "P5";

    default:
        return
            "UNKNOWN";
    }
}

GpuTimingSample run_gpu_once(
    id<MTLCommandQueue> command_queue,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> metadata_buffer,
    id<MTLBuffer> bitset_buffer,
    std::size_t column_bytes,
    std::size_t element_count,
    std::uint32_t predicate_mode,
    std::size_t threads_per_group
) {

    GpuTimingSample sample;

    @autoreleasepool {

        const auto wall_start =
            Clock::now();

        id<MTLCommandBuffer> command_buffer =
            [command_queue commandBuffer];

        if (
            command_buffer ==
            nil
        ) {

            sample.error =
                "No se pudo crear MTLCommandBuffer.";

            return sample;
        }

        id<MTLComputeCommandEncoder> encoder =
            [
                command_buffer
                computeCommandEncoder
            ];

        if (
            encoder ==
            nil
        ) {

            sample.error =
                "No se pudo crear MTLComputeCommandEncoder.";

            return sample;
        }

        [
            encoder
            setComputePipelineState:
                pipeline
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                0 *
                column_bytes
            atIndex:
                0
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                1 *
                column_bytes
            atIndex:
                1
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                2 *
                column_bytes
            atIndex:
                2
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                3 *
                column_bytes
            atIndex:
                3
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                4 *
                column_bytes
            atIndex:
                4
        ];

        [
            encoder
            setBuffer:
                bitset_buffer
            offset:
                0
            atIndex:
                5
        ];

        const std::uint32_t count32 =
            static_cast<std::uint32_t>(
                element_count
            );

        [
            encoder
            setBytes:
                &count32
            length:
                sizeof(
                    count32
                )
            atIndex:
                6
        ];

        [
            encoder
            setBytes:
                &predicate_mode
            length:
                sizeof(
                    predicate_mode
                )
            atIndex:
                7
        ];

        const MTLSize grid_size =
            MTLSizeMake(
                bitset_word_count_for(
                    element_count
                ),
                1,
                1
            );

        const MTLSize threadgroup_size =
            MTLSizeMake(
                threads_per_group,
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

        [
            command_buffer
            waitUntilCompleted
        ];

        const auto wall_end =
            Clock::now();

        sample.wall_ms =
            milliseconds_between(
                wall_start,
                wall_end
            );

        if (
            command_buffer.status !=
                MTLCommandBufferStatusCompleted
        ) {

            if (
                command_buffer.error !=
                    nil
            ) {

                sample.error =
                    string_from_nsstring(
                        command_buffer
                            .error
                            .localizedDescription
                    );

            } else {

                sample.error =
                    "El Command Buffer no finalizo correctamente.";
            }

            return sample;
        }

        sample.completed =
            true;

        const double gpu_start =
            command_buffer.GPUStartTime;

        const double gpu_end =
            command_buffer.GPUEndTime;

        if (
            std::isfinite(
                gpu_start
            ) &&
            std::isfinite(
                gpu_end
            ) &&
            gpu_start >
                0.0 &&
            gpu_end >=
                gpu_start
        ) {

            sample.device_timing_available =
                true;

            sample.device_ms =
                (
                    gpu_end -
                    gpu_start
                ) *
                1000.0;
        }
    }

    return sample;
}

} // namespace

PredicateBreakEvenBenchmarkResult
PredicateBreakEvenBenchmark::run() {

    PredicateBreakEvenBenchmarkResult output;

    output.kernel_name =
        kKernelName;

    output.max_element_count =
        kMaxElementCount;

    output.metadata_column_count =
        kMetadataColumnCount;

    output.warmup_iterations =
        kWarmupIterations;

    output.measured_iterations =
        kMeasuredIterations;

    @autoreleasepool {

        //
        // ============================================================
        // 1. Metal
        // ============================================================
        //

        id<MTLDevice> device =
            MTLCreateSystemDefaultDevice();

        if (
            device ==
            nil
        ) {

            return output;
        }

        output.metal_available =
            true;

        output.device_name =
            string_from_nsstring(
                device.name
            );

        output.unified_memory =
            device.hasUnifiedMemory;

        id<MTLLibrary> library =
            [device newDefaultLibrary];

        if (
            library ==
            nil
        ) {

            output.metal_error =
                "No se pudo cargar la biblioteca Metal.";

            return output;
        }

        output.library_available =
            true;

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

            output.metal_error =
                "No se encontro benchmark_predicate_to_bitset.";

            return output;
        }

        output.kernel_available =
            true;

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

                output.metal_error =
                    string_from_nsstring(
                        pipeline_error
                            .localizedDescription
                    );

            } else {

                output.metal_error =
                    "No se pudo crear MTLComputePipelineState.";
            }

            return output;
        }

        output.pipeline_available =
            true;

        output.thread_execution_width =
            static_cast<std::size_t>(
                pipeline.threadExecutionWidth
            );

        output.max_total_threads_per_threadgroup =
            static_cast<std::size_t>(
                pipeline
                    .maxTotalThreadsPerThreadgroup
            );

        output.threads_per_threadgroup =
            choose_threads_per_threadgroup(
                output.thread_execution_width,
                output
                    .max_total_threads_per_threadgroup
            );

        if (
            output.threads_per_threadgroup ==
            0
        ) {

            output.metal_error =
                "No existe un threadgroup valido.";

            return output;
        }

        id<MTLCommandQueue> command_queue =
            [device newCommandQueue];

        if (
            command_queue ==
            nil
        ) {

            output.metal_error =
                "No se pudo crear MTLCommandQueue.";

            return output;
        }

        output.command_queue_available =
            true;

        //
        // ============================================================
        // 2. Metadata máxima Shared
        // ============================================================
        //

        const std::size_t column_bytes =
            kMaxElementCount *
            sizeof(
                std::uint32_t
            );

        const std::size_t metadata_bytes =
            column_bytes *
            kMetadataColumnCount;

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
            metadata_buffer.contents ==
                nullptr
        ) {

            output.metal_error =
                "No se pudo crear metadata Shared.";

            return output;
        }

        auto* metadata =
            static_cast<std::uint32_t*>(
                metadata_buffer.contents
            );

        std::uint32_t* tenant =
            metadata +
            0 *
            kMaxElementCount;

        std::uint32_t* category =
            metadata +
            1 *
            kMaxElementCount;

        std::uint32_t* year =
            metadata +
            2 *
            kMaxElementCount;

        std::uint32_t* language =
            metadata +
            3 *
            kMaxElementCount;

        std::uint32_t* security_level =
            metadata +
            4 *
            kMaxElementCount;

        for (
            std::size_t index = 0;
            index <
                kMaxElementCount;
            ++index
        ) {

            const std::uint64_t hash =
                metadata_hash(
                    index
                );

            tenant[index] =
                static_cast<std::uint32_t>(
                    hash &
                    0x0FU
                );

            category[index] =
                static_cast<std::uint32_t>(
                    (
                        hash >>
                        8U
                    ) &
                    0x0FU
                );

            year[index] =
                2019U +
                static_cast<std::uint32_t>(
                    (
                        hash >>
                        16U
                    ) %
                    8U
                );

            language[index] =
                static_cast<std::uint32_t>(
                    (
                        hash >>
                        24U
                    ) &
                    0x03U
                );

            security_level[index] =
                static_cast<std::uint32_t>(
                    (
                        hash >>
                        32U
                    ) %
                    6U
                );
        }

        const MetadataView metadata_view{
            .tenant =
                tenant,

            .category =
                category,

            .year =
                year,

            .language =
                language,

            .security_level =
                security_level
        };

        //
        // ============================================================
        // 3. Output buffers
        // ============================================================
        //

        const std::size_t max_word_count =
            bitset_word_count_for(
                kMaxElementCount
            );

        const std::size_t max_bitset_bytes =
            max_word_count *
            sizeof(
                std::uint32_t
            );

        id<MTLBuffer> gpu_bitset_buffer =
            [
                device
                newBufferWithLength:
                    max_bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        if (
            gpu_bitset_buffer ==
                nil ||
            gpu_bitset_buffer.contents ==
                nullptr
        ) {

            output.metal_error =
                "No se pudo crear bitset GPU Shared.";

            return output;
        }

        auto* gpu_bitset =
            static_cast<std::uint32_t*>(
                gpu_bitset_buffer.contents
            );

        std::vector<std::uint32_t>
            cpu_bitset(
                max_word_count,
                0U
            );

        //
        // ============================================================
        // 4. Casos
        // ============================================================
        //

        bool all_exact =
            true;

        for (
            std::uint32_t mode :
            kPredicateModes
        ) {

            for (
                std::size_t element_count :
                kElementCounts
            ) {

                PredicateBreakEvenCase
                    benchmark_case;

                benchmark_case.element_count =
                    element_count;

                benchmark_case.predicate_mode =
                    mode;

                benchmark_case.predicate_name =
                    predicate_name(
                        mode
                    );

                benchmark_case.bitset_word_count =
                    bitset_word_count_for(
                        element_count
                    );

                //
                // ----------------------------------------------------
                // 4.1 Verificación semántica previa
                // ----------------------------------------------------
                //

                materialize_cpu_bitset(
                    metadata_view,
                    element_count,
                    mode,
                    cpu_bitset.data()
                );

                const GpuTimingSample
                    verification_gpu =
                    run_gpu_once(
                        command_queue,
                        pipeline,
                        metadata_buffer,
                        gpu_bitset_buffer,
                        column_bytes,
                        element_count,
                        mode,
                        output
                            .threads_per_threadgroup
                    );

                if (
                    !verification_gpu.completed
                ) {

                    output.metal_error =
                        verification_gpu.error;

                    return output;
                }

                benchmark_case.mismatch_word_count =
                    mismatch_word_count(
                        cpu_bitset.data(),
                        gpu_bitset,
                        benchmark_case
                            .bitset_word_count
                    );

                benchmark_case.exact_match =
                    benchmark_case
                        .mismatch_word_count ==
                    0;

                all_exact =
                    all_exact &&
                    benchmark_case.exact_match;

                benchmark_case.selected_count =
                    selected_count_from_bitset(
                        cpu_bitset.data(),
                        benchmark_case
                            .bitset_word_count
                    );

                benchmark_case.observed_selectivity =
                    static_cast<double>(
                        benchmark_case
                            .selected_count
                    ) /
                    static_cast<double>(
                        element_count
                    );

                //
                // ----------------------------------------------------
                // 4.2 Warmup
                // ----------------------------------------------------
                //

                for (
                    std::size_t iteration = 0;
                    iteration <
                        kWarmupIterations;
                    ++iteration
                ) {

                    materialize_cpu_bitset(
                        metadata_view,
                        element_count,
                        mode,
                        cpu_bitset.data()
                    );

                    const GpuTimingSample warm_gpu =
                        run_gpu_once(
                            command_queue,
                            pipeline,
                            metadata_buffer,
                            gpu_bitset_buffer,
                            column_bytes,
                            element_count,
                            mode,
                            output
                                .threads_per_threadgroup
                        );

                    if (
                        !warm_gpu.completed
                    ) {

                        output.metal_error =
                            warm_gpu.error;

                        return output;
                    }
                }

                //
                // ----------------------------------------------------
                // 4.3 Medición
                // ----------------------------------------------------
                //

                std::vector<double>
                    cpu_times;

                std::vector<double>
                    gpu_wall_times;

                std::vector<double>
                    gpu_device_times;

                cpu_times.reserve(
                    kMeasuredIterations
                );

                gpu_wall_times.reserve(
                    kMeasuredIterations
                );

                gpu_device_times.reserve(
                    kMeasuredIterations
                );

                bool all_gpu_device_timings =
                    true;

                for (
                    std::size_t iteration = 0;
                    iteration <
                        kMeasuredIterations;
                    ++iteration
                ) {

                    //
                    // Alternamos el orden para reducir sesgo sistemático
                    // CPU-first / GPU-first.
                    //

                    const bool cpu_first =
                        (
                            iteration &
                            1U
                        ) ==
                        0U;

                    auto measure_cpu =
                        [&]() {

                            const auto start =
                                Clock::now();

                            materialize_cpu_bitset(
                                metadata_view,
                                element_count,
                                mode,
                                cpu_bitset.data()
                            );

                            const auto end =
                                Clock::now();

                            cpu_times.push_back(
                                milliseconds_between(
                                    start,
                                    end
                                )
                            );
                        };

                    auto measure_gpu =
                        [&]() -> bool {

                            const GpuTimingSample sample =
                                run_gpu_once(
                                    command_queue,
                                    pipeline,
                                    metadata_buffer,
                                    gpu_bitset_buffer,
                                    column_bytes,
                                    element_count,
                                    mode,
                                    output
                                        .threads_per_threadgroup
                                );

                            if (
                                !sample.completed
                            ) {

                                output.metal_error =
                                    sample.error;

                                return false;
                            }

                            gpu_wall_times.push_back(
                                sample.wall_ms
                            );

                            if (
                                sample
                                    .device_timing_available
                            ) {

                                gpu_device_times.push_back(
                                    sample.device_ms
                                );

                            } else {

                                all_gpu_device_timings =
                                    false;
                            }

                            return true;
                        };

                    if (
                        cpu_first
                    ) {

                        measure_cpu();

                        if (
                            !measure_gpu()
                        ) {

                            return output;
                        }

                    } else {

                        if (
                            !measure_gpu()
                        ) {

                            return output;
                        }

                        measure_cpu();
                    }
                }

                //
                // ----------------------------------------------------
                // 4.4 Resumen estadístico
                // ----------------------------------------------------
                //

                benchmark_case.cpu_p50_ms =
                    percentile(
                        cpu_times,
                        0.50
                    );

                benchmark_case.cpu_p95_ms =
                    percentile(
                        cpu_times,
                        0.95
                    );

                benchmark_case.gpu_wall_p50_ms =
                    percentile(
                        gpu_wall_times,
                        0.50
                    );

                benchmark_case.gpu_wall_p95_ms =
                    percentile(
                        gpu_wall_times,
                        0.95
                    );

                benchmark_case
                    .gpu_device_timing_available =
                    all_gpu_device_timings &&
                    gpu_device_times.size() ==
                        kMeasuredIterations;

                if (
                    benchmark_case
                        .gpu_device_timing_available
                ) {

                    benchmark_case
                        .gpu_device_p50_ms =
                        percentile(
                            gpu_device_times,
                            0.50
                        );

                    benchmark_case
                        .gpu_device_p95_ms =
                        percentile(
                            gpu_device_times,
                            0.95
                        );
                }

                if (
                    benchmark_case
                        .gpu_wall_p50_ms >
                    0.0
                ) {

                    benchmark_case
                        .gpu_wall_speedup_vs_cpu =
                        benchmark_case
                            .cpu_p50_ms /
                        benchmark_case
                            .gpu_wall_p50_ms;
                }

                benchmark_case.gpu_faster_at_p50 =
                    benchmark_case
                        .gpu_wall_p50_ms <
                    benchmark_case
                        .cpu_p50_ms;

                output.cases.push_back(
                    benchmark_case
                );
            }
        }

        output.all_cases_exact =
            all_exact;

        output.valid =
            output.metal_available &&
            output.unified_memory &&
            output.library_available &&
            output.kernel_available &&
            output.pipeline_available &&
            output.command_queue_available &&
            output.all_cases_exact &&
            output.cases.size() ==
                (
                    kPredicateModes.size() *
                    kElementCounts.size()
                );
    }

    return output;
}

} // namespace racorn_uma::platform::apple

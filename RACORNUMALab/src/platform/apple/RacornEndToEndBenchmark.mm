//
//  RacornEndToEndBenchmark.mm
//  RACORNUMALab
//
//  Propósito:
//  Medir end-to-end tres estrategias de filtrado para RACORN:
//
//      CPU-Lazy
//      CPU-Bitset
//      GPU-Bitset
//
//  Gate:
//  G-UMA5B — RACORN End-to-End CPU/GPU.
//
//  Revisión G-UMA5B.1:
//  La versión inicial realizaba un volumen experimental innecesariamente
//  alto antes de producir salida visible. Esta revisión:
//
//  - reduce Q de 32 a 16;
//  - reduce warm-up de 2 a 1 ronda;
//  - reduce mediciones de 10 a 5 rondas;
//  - sustituye index.search(exact=true) por un oráculo exacto filtrado
//    independiente;
//  - imprime progreso durante construcción, validación y benchmark.
//
//  Configuración:
//
//      N ∈ {20k, 100k, 250k}
//      D = 32
//      M = 16
//      efConstruction = 100
//      efSearch = 200
//      k = 10
//      Q = 16
//      bridge_budget_ratio = 0.50
//      two_hop_threshold = 0.30
//
//  Cada query utiliza:
//
//      tenant = query_index ∈ [0,15]
//
//  y:
//
//      year >= 2022
//      language >= 2
//      category IN {3,7,9}
//      security_level <= 2
//
//  Estrategias:
//
//  CPU-Lazy:
//      RACORN
//          ↓
//      predicate(metadata[id])
//
//  CPU-Bitset:
//      CPU materializa bitset Shared
//          ↓
//      RACORN CPU consume bitset
//
//  GPU-Bitset:
//      Metal materializa bitset Shared
//          ↓
//      una sincronización
//          ↓
//      RACORN CPU consume bitset
//
//  Oráculo:
//  El ground truth exacto es independiente de USEARCH:
//
//      recorrer IDs
//          ↓
//      evaluar filtro exacto
//          ↓
//      calcular L2 solo si el ID es elegible
//          ↓
//      ordenar candidatos elegibles
//          ↓
//      seleccionar top-k
//
//  Esto evita obligar al camino exacto de USEARCH a recorrer de forma
//  costosa el espacio vectorial para cada consulta.
//
//  Condición PASS:
//  Las tres estrategias deben producir:
//
//      mismo top-k
//      mismo Recall@k
//      mismo computed_distances
//      mismo visited_members
//      mismo predicate_evaluations
//      mismo fallback_count
//
//  Benchmark:
//  - Una ronda de warmup.
//  - Cinco rondas medidas.
//  - Orden de estrategias rotado.
//
//  Invariantes:
//  - RACORN no se modifica.
//  - HNSW/RACORN nunca se ejecuta en GPU.
//  - CPU-Bitset y GPU-Bitset usan almacenamiento Shared.
//  - Sin copias explícitas CPU↔VRAM.
//  - Índice construido fuera del timing.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Sin persistencia.
//

#include "platform/apple/RacornEndToEndBenchmark.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace racorn_uma::platform::apple {

namespace {

using namespace unum::usearch;

using Clock =
    std::chrono::steady_clock;

constexpr char const* kKernelName =
    "racorn_e2e_predicate_to_bitset";

constexpr std::array<std::size_t, 3>
    kVectorCounts{
        20'000U,
        100'000U,
        250'000U
    };

constexpr std::size_t kMaxVectorCount =
    250'000U;

constexpr std::size_t kDimensions =
    32U;

constexpr std::size_t kConnectivity =
    16U;

constexpr std::size_t kEfConstruction =
    100U;

constexpr std::size_t kEfSearch =
    200U;

constexpr std::size_t kNeighbors =
    10U;

constexpr std::size_t kQueryCount =
    16U;

constexpr std::size_t kWarmupRounds =
    1U;

constexpr std::size_t kMeasuredRounds =
    5U;

constexpr double kTwoHopThreshold =
    0.30;

constexpr double kBridgeBudgetRatio =
    0.50;

constexpr std::size_t kMetadataColumnCount =
    5U;

constexpr std::size_t kBitsPerWord =
    32U;

constexpr std::uint64_t kBaseSeed =
    0x13579BDF2468ACE0ULL;

constexpr std::uint64_t kQuerySeed =
    0xCAFEBABE12345678ULL;

constexpr std::uint64_t kMetadataSeed =
    0xD1B54A32D192ED03ULL;

using VectorKey =
    std::size_t;

using Slot =
    std::size_t;

using Index =
    index_gt<
        float,
        VectorKey,
        Slot
    >;

struct FilterSpec final {

    std::uint32_t tenant_required{};

    std::uint32_t language_min{};
};

struct MetadataView final {

    std::uint32_t const* tenant{};
    std::uint32_t const* category{};
    std::uint32_t const* year{};
    std::uint32_t const* language{};
    std::uint32_t const* security_level{};
};

struct SearchSnapshot final {

    std::vector<std::size_t> keys;

    std::size_t computed_distances{};
    std::size_t visited_members{};
    std::size_t predicate_evaluations{};
    std::size_t fallback_count{};
};

struct ExactCandidate final {

    float distance{};

    std::size_t key{};
};

struct GpuPrepareSample final {

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
        value == nil
    ) {
        return {};
    }

    const char* utf8 =
        [value UTF8String];

    if (
        utf8 == nullptr
    ) {
        return {};
    }

    return std::string(
        utf8
    );
}

std::uint64_t splitmix64_step(
    std::uint64_t& state
) noexcept {

    state +=
        0x9E3779B97F4A7C15ULL;

    std::uint64_t value =
        state;

    value =
        (
            value ^
            (
                value >> 30U
            )
        ) *
        0xBF58476D1CE4E5B9ULL;

    value =
        (
            value ^
            (
                value >> 27U
            )
        ) *
        0x94D049BB133111EBULL;

    return
        value ^
        (
            value >> 31U
        );
}

std::uint64_t splitmix64_value(
    std::uint64_t value
) noexcept {

    value +=
        0x9E3779B97F4A7C15ULL;

    value =
        (
            value ^
            (
                value >> 30U
            )
        ) *
        0xBF58476D1CE4E5B9ULL;

    value =
        (
            value ^
            (
                value >> 27U
            )
        ) *
        0x94D049BB133111EBULL;

    return
        value ^
        (
            value >> 31U
        );
}

float random_signed_float(
    std::uint64_t& state
) noexcept {

    constexpr double denominator =
        16777215.0;

    const std::uint32_t bits =
        static_cast<std::uint32_t>(
            splitmix64_step(
                state
            ) >>
            40U
        );

    const double unit =
        static_cast<double>(
            bits
        ) /
        denominator;

    return
        static_cast<float>(
            unit *
                2.0 -
            1.0
        );
}

std::uint64_t metadata_hash(
    std::size_t index
) noexcept {

    return
        splitmix64_value(
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

FilterSpec filter_for_query(
    std::size_t query_index
) noexcept {

    return FilterSpec{
        .tenant_required =
            static_cast<std::uint32_t>(
                query_index %
                16U
            ),

        .language_min =
            2U
    };
}

bool evaluate_filter(
    MetadataView const& metadata,
    std::size_t key,
    FilterSpec const& filter
) noexcept {

    const std::uint32_t category =
        metadata.category[key];

    const bool category_allowed =
        category ==
            3U ||
        category ==
            7U ||
        category ==
            9U;

    return
        metadata.tenant[key] ==
            filter.tenant_required &&

        metadata.year[key] >=
            2022U &&

        metadata.language[key] >=
            filter.language_min &&

        category_allowed &&

        metadata.security_level[key] <=
            2U;
}

class VectorStore final {

public:

    VectorStore(
        std::size_t count,
        std::size_t dimensions,
        std::uint64_t seed
    )
        : dimensions_(
              dimensions
          ),
          data_(
              count *
              dimensions
          ) {

        std::uint64_t state =
            seed;

        for (
            float& value :
            data_
        ) {

            value =
                random_signed_float(
                    state
                );
        }
    }

    [[nodiscard]]
    float const* row(
        std::size_t index
    ) const noexcept {

        return
            data_.data() +
            index *
                dimensions_;
    }

private:

    std::size_t dimensions_{};

    std::vector<float> data_;
};

class QueryStore final {

public:

    QueryStore(
        std::size_t count,
        std::size_t dimensions,
        std::uint64_t seed
    )
        : dimensions_(
              dimensions
          ),
          data_(
              count *
              dimensions
          ) {

        std::uint64_t state =
            seed;

        for (
            float& value :
            data_
        ) {

            value =
                random_signed_float(
                    state
                );
        }
    }

    [[nodiscard]]
    float const* row(
        std::size_t index
    ) const noexcept {

        return
            data_.data() +
            index *
                dimensions_;
    }

private:

    std::size_t dimensions_{};

    std::vector<float> data_;
};

float l2_squared(
    float const* first,
    float const* second,
    std::size_t dimensions
) noexcept {

    float sum =
        0.0F;

    for (
        std::size_t dimension = 0;
        dimension < dimensions;
        ++dimension
    ) {

        const float difference =
            first[dimension] -
            second[dimension];

        sum +=
            difference *
            difference;
    }

    return sum;
}

struct L2Metric final {

    using IndexType =
        Index;

    using MemberRef =
        typename IndexType::member_cref_t;

    using MemberIterator =
        typename IndexType::member_citerator_t;

    VectorStore const* store{};

    std::size_t dimensions{};

    [[nodiscard]]
    float const* row(
        std::size_t key
    ) const noexcept {

        return
            store->row(
                key
            );
    }

    float operator()(
        MemberRef const& first,
        MemberRef const& second
    ) const {

        return
            metric_l2sq_gt<
                float,
                float
            >{}(
                row(
                    get_key(
                        second
                    )
                ),
                row(
                    get_key(
                        first
                    )
                ),
                dimensions
            );
    }

    float operator()(
        float const* query,
        MemberRef const& member
    ) const {

        return
            metric_l2sq_gt<
                float,
                float
            >{}(
                query,
                row(
                    get_key(
                        member
                    )
                ),
                dimensions
            );
    }

    float operator()(
        MemberIterator const& first,
        MemberIterator const& second
    ) const {

        return
            metric_l2sq_gt<
                float,
                float
            >{}(
                row(
                    get_key(
                        *second
                    )
                ),
                row(
                    get_key(
                        *first
                    )
                ),
                dimensions
            );
    }

    float operator()(
        float const* query,
        MemberIterator const& member
    ) const {

        return
            metric_l2sq_gt<
                float,
                float
            >{}(
                query,
                row(
                    get_key(
                        *member
                    )
                ),
                dimensions
            );
    }
};

template <typename IndexType>
class AlignedIndex final {

public:

    using Allocator =
        aligned_allocator_gt<
            IndexType,
            64
        >;

    explicit AlignedIndex(
        index_config_t const& configuration
    ) {

        pointer_ =
            Allocator{}.allocate(
                1
            );

        new (pointer_)
            IndexType(
                configuration
            );
    }

    ~AlignedIndex() {

        if (
            pointer_ == nullptr
        ) {
            return;
        }

        pointer_->~IndexType();

        Allocator{}.deallocate(
            pointer_,
            1
        );
    }

    AlignedIndex(
        AlignedIndex const&
    ) = delete;

    AlignedIndex& operator=(
        AlignedIndex const&
    ) = delete;

    [[nodiscard]]
    IndexType& get() noexcept {

        return *pointer_;
    }

private:

    IndexType* pointer_{
        nullptr
    };
};

struct LazyPredicate final {

    MetadataView metadata{};

    FilterSpec filter{};

    [[nodiscard]]
    bool accepts_key(
        std::size_t key
    ) const noexcept {

        return
            evaluate_filter(
                metadata,
                key,
                filter
            );
    }

    template <typename MemberLike>
    bool operator()(
        MemberLike&& member
    ) const noexcept {

        return
            accepts_key(
                static_cast<std::size_t>(
                    get_key(
                        member
                    )
                )
            );
    }
};

struct BitsetPredicate final {

    std::uint32_t const* bitset{};

    [[nodiscard]]
    bool accepts_key(
        std::size_t key
    ) const noexcept {

        const std::size_t word =
            key >> 5U;

        const std::size_t bit =
            key & 31U;

        return
            (
                (
                    bitset[word] >>
                    bit
                ) &
                1U
            ) !=
            0U;
    }

    template <typename MemberLike>
    bool operator()(
        MemberLike&& member
    ) const noexcept {

        return
            accepts_key(
                static_cast<std::size_t>(
                    get_key(
                        member
                    )
                )
            );
    }
};

void materialize_cpu_bitset(
    MetadataView const& metadata,
    std::size_t element_count,
    FilterSpec const& filter,
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

        const std::size_t base =
            word_index *
            kBitsPerWord;

        std::uint32_t word =
            0U;

        for (
            std::size_t bit_index = 0;
            bit_index < kBitsPerWord;
            ++bit_index
        ) {

            const std::size_t key =
                base +
                bit_index;

            if (
                key >= element_count
            ) {
                break;
            }

            if (
                evaluate_filter(
                    metadata,
                    key,
                    filter
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

std::vector<std::size_t>
exact_filtered_top_k(
    VectorStore const& base,
    float const* query,
    MetadataView const& metadata,
    FilterSpec const& filter,
    std::size_t vector_count,
    std::size_t k
) {

    std::vector<ExactCandidate>
        candidates;

    candidates.reserve(
        512U
    );

    for (
        std::size_t key = 0;
        key < vector_count;
        ++key
    ) {

        if (
            !evaluate_filter(
                metadata,
                key,
                filter
            )
        ) {
            continue;
        }

        candidates.push_back(
            ExactCandidate{
                .distance =
                    l2_squared(
                        query,
                        base.row(
                            key
                        ),
                        kDimensions
                    ),

                .key =
                    key
            }
        );
    }

    if (
        candidates.size() < k
    ) {
        return {};
    }

    const auto comparator =
        [](
            ExactCandidate const& first,
            ExactCandidate const& second
        ) {

            if (
                first.distance <
                second.distance
            ) {
                return true;
            }

            if (
                first.distance >
                second.distance
            ) {
                return false;
            }

            return
                first.key <
                second.key;
        };

    std::partial_sort(
        candidates.begin(),
        candidates.begin() +
            static_cast<std::ptrdiff_t>(
                k
            ),
        candidates.end(),
        comparator
    );

    std::vector<std::size_t>
        result;

    result.reserve(
        k
    );

    for (
        std::size_t index = 0;
        index < k;
        ++index
    ) {

        result.push_back(
            candidates[index].key
        );
    }

    return result;
}

std::size_t choose_threads_per_threadgroup(
    std::size_t execution_width,
    std::size_t maximum
) noexcept {

    if (
        maximum == 0
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
        execution_width != 0
    ) {

        selected =
            (
                selected /
                execution_width
            ) *
            execution_width;

        if (
            selected == 0
        ) {

            selected =
                std::min(
                    execution_width,
                    maximum
                );
        }
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
        values.size() == 1
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
        lower == upper
    ) {
        return values[lower];
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

index_search_config_t
make_racorn_search_config() {

    index_search_config_t config;

    config.expansion =
        kEfSearch;

    config.thread =
        0;

    config.two_hop_threshold =
        kTwoHopThreshold;

    config.racorn =
        true;

    config.bridge_budget_ratio =
        kBridgeBudgetRatio;

    config.three_hop_threshold =
        -1.0;

    config.fallback_exact_threshold =
        0.0;

    config.racorn_stride_sampling =
        true;

    return config;
}

SearchSnapshot snapshot_search_result(
    Index::search_result_t const& result
) {

    SearchSnapshot snapshot;

    snapshot.keys.reserve(
        result.size()
    );

    for (
        std::size_t index = 0;
        index < result.size();
        ++index
    ) {

        snapshot.keys.push_back(
            static_cast<std::size_t>(
                result[index]
                    .member
                    .key
            )
        );
    }

    snapshot.computed_distances =
        result.computed_distances;

    snapshot.visited_members =
        result.visited_members;

    snapshot.predicate_evaluations =
        result.predicate_evaluations;

    snapshot.fallback_count =
        result.fallback_count;

    return snapshot;
}

double recall_at_k(
    std::vector<std::size_t> const& approximate,
    std::vector<std::size_t> const& exact
) {

    if (
        exact.empty()
    ) {
        return 0.0;
    }

    std::unordered_set<std::size_t>
        truth(
            exact.begin(),
            exact.end()
        );

    std::size_t hits =
        0;

    for (
        std::size_t key :
        approximate
    ) {

        if (
            truth.contains(
                key
            )
        ) {
            ++hits;
        }
    }

    return
        static_cast<double>(
            hits
        ) /
        static_cast<double>(
            exact.size()
        );
}

template <typename Predicate>
bool result_satisfies_filter(
    std::vector<std::size_t> const& keys,
    Predicate const& predicate
) noexcept {

    for (
        std::size_t key :
        keys
    ) {

        if (
            !predicate.accepts_key(
                key
            )
        ) {
            return false;
        }
    }

    return true;
}

bool same_double(
    double first,
    double second
) noexcept {

    return
        std::abs(
            first -
            second
        ) <=
        1e-12;
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

GpuPrepareSample run_gpu_prepare(
    id<MTLCommandQueue> command_queue,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> metadata_buffer,
    id<MTLBuffer> bitset_buffer,
    std::size_t metadata_column_stride_bytes,
    std::size_t element_count,
    FilterSpec const& filter,
    std::size_t threads_per_group
) {

    GpuPrepareSample sample;

    @autoreleasepool {

        const auto wall_start =
            Clock::now();

        id<MTLCommandBuffer> command_buffer =
            [command_queue commandBuffer];

        if (
            command_buffer == nil
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
            encoder == nil
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

        for (
            NSUInteger column = 0;
            column < 5;
            ++column
        ) {

            [
                encoder
                setBuffer:
                    metadata_buffer
                offset:
                    column *
                    metadata_column_stride_bytes
                atIndex:
                    column
            ];
        }

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
                &filter.tenant_required
            length:
                sizeof(
                    filter.tenant_required
                )
            atIndex:
                7
        ];

        [
            encoder
            setBytes:
                &filter.language_min
            length:
                sizeof(
                    filter.language_min
                )
            atIndex:
                8
        ];

        const MTLSize grid =
            MTLSizeMake(
                bitset_word_count_for(
                    element_count
                ),
                1,
                1
            );

        const MTLSize group =
            MTLSizeMake(
                threads_per_group,
                1,
                1
            );

        [
            encoder
            dispatchThreads:
                grid
            threadsPerThreadgroup:
                group
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
                command_buffer.error != nil
            ) {

                sample.error =
                    string_from_nsstring(
                        command_buffer
                            .error
                            .localizedDescription
                    );

            } else {

                sample.error =
                    "Command Buffer GPU incompleto.";
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

template <typename Predicate>
double timed_racorn_search(
    Index& index,
    float const* query,
    L2Metric const& metric,
    Predicate const& predicate,
    std::size_t& benchmark_sink
) {

    auto configuration =
        make_racorn_search_config();

    const auto start =
        Clock::now();

    auto result =
        index.search(
            query,
            kNeighbors,
            metric,
            configuration,
            predicate
        );

    const auto end =
        Clock::now();

    benchmark_sink ^=
        result.size() +
        result.computed_distances +
        result.visited_members +
        result.fallback_count;

    return
        milliseconds_between(
            start,
            end
        );
}

void fill_latency_stats(
    EndToEndLatencyStats& output,
    std::vector<double> const& total,
    std::vector<double> const& prepare,
    std::vector<double> const& search,
    std::vector<double> const& device
) {

    output.total_p50_ms =
        percentile(
            total,
            0.50
        );

    output.total_p95_ms =
        percentile(
            total,
            0.95
        );

    output.prepare_p50_ms =
        percentile(
            prepare,
            0.50
        );

    output.prepare_p95_ms =
        percentile(
            prepare,
            0.95
        );

    output.search_p50_ms =
        percentile(
            search,
            0.50
        );

    output.search_p95_ms =
        percentile(
            search,
            0.95
        );

    output.gpu_device_timing_available =
        !device.empty() &&
        device.size() ==
            total.size();

    if (
        output.gpu_device_timing_available
    ) {

        output.gpu_device_p50_ms =
            percentile(
                device,
                0.50
            );

        output.gpu_device_p95_ms =
            percentile(
                device,
                0.95
            );
    }
}

void print_progress(
    std::string const& message
) {

    std::cout
        << message
        << '\n'
        << std::flush;
}

} // namespace

RacornEndToEndBenchmarkResult
RacornEndToEndBenchmark::run() {

    RacornEndToEndBenchmarkResult output;

    output.kernel_name =
        kKernelName;

    output.dimensions =
        kDimensions;

    output.connectivity =
        kConnectivity;

    output.ef_construction =
        kEfConstruction;

    output.ef_search =
        kEfSearch;

    output.k =
        kNeighbors;

    output.query_count =
        kQueryCount;

    output.two_hop_threshold =
        kTwoHopThreshold;

    output.bridge_budget_ratio =
        kBridgeBudgetRatio;

    output.warmup_rounds =
        kWarmupRounds;

    output.measured_rounds =
        kMeasuredRounds;

    output.racorn_modified =
        false;

    print_progress(
        "[G-UMA5B] Inicializando Metal..."
    );

    @autoreleasepool {

        id<MTLDevice> device =
            MTLCreateSystemDefaultDevice();

        if (
            device == nil
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
            library == nil
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
            function == nil
        ) {

            output.metal_error =
                "No se encontro racorn_e2e_predicate_to_bitset.";

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
            pipeline == nil
        ) {

            if (
                pipeline_error != nil
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
            output.threads_per_threadgroup == 0
        ) {

            output.metal_error =
                "Configuracion threadgroup invalida.";

            return output;
        }

        id<MTLCommandQueue> command_queue =
            [device newCommandQueue];

        if (
            command_queue == nil
        ) {

            output.metal_error =
                "No se pudo crear MTLCommandQueue.";

            return output;
        }

        output.command_queue_available =
            true;

        print_progress(
            "[G-UMA5B] Inicializando metadata Shared..."
        );

        const std::size_t column_stride_bytes =
            kMaxVectorCount *
            sizeof(
                std::uint32_t
            );

        const std::size_t metadata_bytes =
            column_stride_bytes *
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
            metadata_buffer == nil ||
            metadata_buffer.contents == nullptr
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
            kMaxVectorCount;

        std::uint32_t* category =
            metadata +
            1 *
            kMaxVectorCount;

        std::uint32_t* year =
            metadata +
            2 *
            kMaxVectorCount;

        std::uint32_t* language =
            metadata +
            3 *
            kMaxVectorCount;

        std::uint32_t* security_level =
            metadata +
            4 *
            kMaxVectorCount;

        for (
            std::size_t index = 0;
            index < kMaxVectorCount;
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

        const std::size_t max_word_count =
            bitset_word_count_for(
                kMaxVectorCount
            );

        const std::size_t max_bitset_bytes =
            max_word_count *
            sizeof(
                std::uint32_t
            );

        id<MTLBuffer> cpu_bitset_buffer =
            [
                device
                newBufferWithLength:
                    max_bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        id<MTLBuffer> gpu_bitset_buffer =
            [
                device
                newBufferWithLength:
                    max_bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        if (
            cpu_bitset_buffer == nil ||
            gpu_bitset_buffer == nil ||
            cpu_bitset_buffer.contents == nullptr ||
            gpu_bitset_buffer.contents == nullptr
        ) {

            output.metal_error =
                "No se pudieron crear los bitsets Shared.";

            return output;
        }

        auto* cpu_bitset =
            static_cast<std::uint32_t*>(
                cpu_bitset_buffer.contents
            );

        auto* gpu_bitset =
            static_cast<std::uint32_t*>(
                gpu_bitset_buffer.contents
            );

        bool all_cases_identical =
            true;

        for (
            std::size_t vector_count :
            kVectorCounts
        ) {

            std::cout
                << "\n[G-UMA5B] N="
                << vector_count
                << " - preparando dataset e indice...\n"
                << std::flush;

            RacornEndToEndCase
                case_result;

            case_result.vector_count =
                vector_count;

            case_result.selected_min =
                std::numeric_limits<
                    std::size_t
                >::max();

            VectorStore base(
                vector_count,
                kDimensions,
                kBaseSeed
            );

            QueryStore queries(
                kQueryCount,
                kDimensions,
                kQuerySeed
            );

            L2Metric metric{
                .store =
                    &base,

                .dimensions =
                    kDimensions
            };

            index_config_t
                index_configuration(
                    kConnectivity
                );

            AlignedIndex<Index>
                aligned_index(
                    index_configuration
                );

            Index& index =
                aligned_index.get();

            index.reserve(
                index_limits_t(
                    vector_count,
                    1
                )
            );

            index_update_config_t
                update_configuration;

            update_configuration.expansion =
                kEfConstruction;

            update_configuration.thread =
                0;

            const std::size_t progress_step =
                std::max<std::size_t>(
                    vector_count /
                        10U,
                    1U
                );

            for (
                std::size_t key = 0;
                key < vector_count;
                ++key
            ) {

                index.add(
                    static_cast<VectorKey>(
                        key
                    ),
                    base.row(
                        key
                    ),
                    metric,
                    update_configuration
                );

                if (
                    (
                        key +
                        1U
                    ) %
                        progress_step ==
                        0U ||
                    key +
                        1U ==
                        vector_count
                ) {

                    const std::size_t percent =
                        (
                            (
                                key +
                                1U
                            ) *
                            100U
                        ) /
                        vector_count;

                    std::cout
                        << "[G-UMA5B] N="
                        << vector_count
                        << " HNSW "
                        << percent
                        << "%\n"
                        << std::flush;
                }
            }

            print_progress(
                "[G-UMA5B] Validando bitsets, ground truth y semantica..."
            );

            const BitsetPredicate
                cpu_bitset_predicate{
                    .bitset =
                        cpu_bitset
                };

            const BitsetPredicate
                gpu_bitset_predicate{
                    .bitset =
                        gpu_bitset
                };

            std::size_t total_selected =
                0;

            double recall_sum =
                0.0;

            double distances_sum =
                0.0;

            double visited_sum =
                0.0;

            double predicate_evaluations_sum =
                0.0;

            double fallback_sum =
                0.0;

            bool exact_complete =
                true;

            bool filter_results_valid =
                true;

            for (
                std::size_t query_index = 0;
                query_index < kQueryCount;
                ++query_index
            ) {

                const FilterSpec filter =
                    filter_for_query(
                        query_index
                    );

                const LazyPredicate
                    lazy_predicate{
                        .metadata =
                            metadata_view,

                        .filter =
                            filter
                    };

                materialize_cpu_bitset(
                    metadata_view,
                    vector_count,
                    filter,
                    cpu_bitset
                );

                const GpuPrepareSample gpu_prepare =
                    run_gpu_prepare(
                        command_queue,
                        pipeline,
                        metadata_buffer,
                        gpu_bitset_buffer,
                        column_stride_bytes,
                        vector_count,
                        filter,
                        output
                            .threads_per_threadgroup
                    );

                if (
                    !gpu_prepare.completed
                ) {

                    output.metal_error =
                        gpu_prepare.error;

                    return output;
                }

                const std::size_t word_count =
                    bitset_word_count_for(
                        vector_count
                    );

                case_result.bitset_word_mismatches +=
                    mismatch_word_count(
                        cpu_bitset,
                        gpu_bitset,
                        word_count
                    );

                std::size_t selected =
                    0;

                for (
                    std::size_t key = 0;
                    key < vector_count;
                    ++key
                ) {

                    const bool direct =
                        lazy_predicate.accepts_key(
                            key
                        );

                    const bool cpu_bit =
                        cpu_bitset_predicate.accepts_key(
                            key
                        );

                    const bool gpu_bit =
                        gpu_bitset_predicate.accepts_key(
                            key
                        );

                    if (
                        direct
                    ) {
                        ++selected;
                    }

                    if (
                        direct != cpu_bit ||
                        direct != gpu_bit
                    ) {

                        ++case_result
                            .predicate_representation_mismatches;
                    }
                }

                case_result.selected_min =
                    std::min(
                        case_result.selected_min,
                        selected
                    );

                case_result.selected_max =
                    std::max(
                        case_result.selected_max,
                        selected
                    );

                total_selected +=
                    selected;

                if (
                    selected <
                    kNeighbors
                ) {

                    exact_complete =
                        false;
                }

                float const* query =
                    queries.row(
                        query_index
                    );

                const std::vector<std::size_t>
                    exact_keys =
                    exact_filtered_top_k(
                        base,
                        query,
                        metadata_view,
                        filter,
                        vector_count,
                        kNeighbors
                    );

                if (
                    exact_keys.size() !=
                    kNeighbors
                ) {

                    exact_complete =
                        false;
                }

                auto lazy_configuration =
                    make_racorn_search_config();

                auto lazy_result =
                    index.search(
                        query,
                        kNeighbors,
                        metric,
                        lazy_configuration,
                        lazy_predicate
                    );

                const SearchSnapshot
                    lazy_snapshot =
                    snapshot_search_result(
                        lazy_result
                    );

                auto cpu_configuration =
                    make_racorn_search_config();

                auto cpu_result =
                    index.search(
                        query,
                        kNeighbors,
                        metric,
                        cpu_configuration,
                        cpu_bitset_predicate
                    );

                const SearchSnapshot
                    cpu_snapshot =
                    snapshot_search_result(
                        cpu_result
                    );

                auto gpu_configuration =
                    make_racorn_search_config();

                auto gpu_result =
                    index.search(
                        query,
                        kNeighbors,
                        metric,
                        gpu_configuration,
                        gpu_bitset_predicate
                    );

                const SearchSnapshot
                    gpu_snapshot =
                    snapshot_search_result(
                        gpu_result
                    );

                const double lazy_recall =
                    recall_at_k(
                        lazy_snapshot.keys,
                        exact_keys
                    );

                const double cpu_recall =
                    recall_at_k(
                        cpu_snapshot.keys,
                        exact_keys
                    );

                const double gpu_recall =
                    recall_at_k(
                        gpu_snapshot.keys,
                        exact_keys
                    );

                recall_sum +=
                    lazy_recall;

                distances_sum +=
                    static_cast<double>(
                        lazy_snapshot
                            .computed_distances
                    );

                visited_sum +=
                    static_cast<double>(
                        lazy_snapshot
                            .visited_members
                    );

                predicate_evaluations_sum +=
                    static_cast<double>(
                        lazy_snapshot
                            .predicate_evaluations
                    );

                fallback_sum +=
                    static_cast<double>(
                        lazy_snapshot
                            .fallback_count
                    );

                filter_results_valid =
                    filter_results_valid &&

                    result_satisfies_filter(
                        lazy_snapshot.keys,
                        lazy_predicate
                    ) &&

                    result_satisfies_filter(
                        cpu_snapshot.keys,
                        lazy_predicate
                    ) &&

                    result_satisfies_filter(
                        gpu_snapshot.keys,
                        lazy_predicate
                    );

                if (
                    lazy_snapshot.keys !=
                        cpu_snapshot.keys ||
                    lazy_snapshot.keys !=
                        gpu_snapshot.keys
                ) {

                    ++case_result
                        .topk_mismatch_queries;
                }

                if (
                    !same_double(
                        lazy_recall,
                        cpu_recall
                    ) ||
                    !same_double(
                        lazy_recall,
                        gpu_recall
                    )
                ) {

                    ++case_result
                        .recall_mismatch_queries;
                }

                if (
                    lazy_snapshot.computed_distances !=
                        cpu_snapshot.computed_distances ||
                    lazy_snapshot.computed_distances !=
                        gpu_snapshot.computed_distances
                ) {

                    ++case_result
                        .computed_distances_mismatch_queries;
                }

                if (
                    lazy_snapshot.visited_members !=
                        cpu_snapshot.visited_members ||
                    lazy_snapshot.visited_members !=
                        gpu_snapshot.visited_members
                ) {

                    ++case_result
                        .visited_members_mismatch_queries;
                }

                if (
                    lazy_snapshot.predicate_evaluations !=
                        cpu_snapshot.predicate_evaluations ||
                    lazy_snapshot.predicate_evaluations !=
                        gpu_snapshot.predicate_evaluations
                ) {

                    ++case_result
                        .predicate_evaluations_mismatch_queries;
                }

                if (
                    lazy_snapshot.fallback_count !=
                        cpu_snapshot.fallback_count ||
                    lazy_snapshot.fallback_count !=
                        gpu_snapshot.fallback_count
                ) {

                    ++case_result
                        .fallback_count_mismatch_queries;
                }
            }

            const double query_count =
                static_cast<double>(
                    kQueryCount
                );

            case_result.mean_selectivity =
                static_cast<double>(
                    total_selected
                ) /
                (
                    query_count *
                    static_cast<double>(
                        vector_count
                    )
                );

            case_result.mean_recall_at_k =
                recall_sum /
                query_count;

            case_result.mean_computed_distances =
                distances_sum /
                query_count;

            case_result.mean_visited_members =
                visited_sum /
                query_count;

            case_result.mean_predicate_evaluations =
                predicate_evaluations_sum /
                query_count;

            case_result.mean_fallback_count =
                fallback_sum /
                query_count;

            case_result.exact_ground_truth_complete =
                exact_complete;

            case_result.filter_results_valid =
                filter_results_valid;

            case_result.navigation_semantics_identical =
                case_result.bitset_word_mismatches ==
                    0 &&

                case_result.predicate_representation_mismatches ==
                    0 &&

                case_result.topk_mismatch_queries ==
                    0 &&

                case_result.recall_mismatch_queries ==
                    0 &&

                case_result.computed_distances_mismatch_queries ==
                    0 &&

                case_result.visited_members_mismatch_queries ==
                    0 &&

                case_result.predicate_evaluations_mismatch_queries ==
                    0 &&

                case_result.fallback_count_mismatch_queries ==
                    0;

            if (
                !case_result.navigation_semantics_identical ||
                !case_result.exact_ground_truth_complete ||
                !case_result.filter_results_valid
            ) {

                all_cases_identical =
                    false;

                output.cases.push_back(
                    case_result
                );

                continue;
            }

            print_progress(
                "[G-UMA5B] Semantica PASS. Warm-up..."
            );

            for (
                std::size_t round = 0;
                round < kWarmupRounds;
                ++round
            ) {

                for (
                    std::size_t query_index = 0;
                    query_index < kQueryCount;
                    ++query_index
                ) {

                    const FilterSpec filter =
                        filter_for_query(
                            query_index
                        );

                    const LazyPredicate
                        lazy_predicate{
                            .metadata =
                                metadata_view,

                            .filter =
                                filter
                        };

                    float const* query =
                        queries.row(
                            query_index
                        );

                    timed_racorn_search(
                        index,
                        query,
                        metric,
                        lazy_predicate,
                        output.benchmark_sink
                    );

                    materialize_cpu_bitset(
                        metadata_view,
                        vector_count,
                        filter,
                        cpu_bitset
                    );

                    timed_racorn_search(
                        index,
                        query,
                        metric,
                        cpu_bitset_predicate,
                        output.benchmark_sink
                    );

                    const GpuPrepareSample sample =
                        run_gpu_prepare(
                            command_queue,
                            pipeline,
                            metadata_buffer,
                            gpu_bitset_buffer,
                            column_stride_bytes,
                            vector_count,
                            filter,
                            output
                                .threads_per_threadgroup
                        );

                    if (
                        !sample.completed
                    ) {

                        output.metal_error =
                            sample.error;

                        return output;
                    }

                    timed_racorn_search(
                        index,
                        query,
                        metric,
                        gpu_bitset_predicate,
                        output.benchmark_sink
                    );
                }
            }

            print_progress(
                "[G-UMA5B] Midiendo latencia end-to-end..."
            );

            const std::size_t sample_count =
                kMeasuredRounds *
                kQueryCount;

            std::vector<double> lazy_total;
            std::vector<double> lazy_search;

            std::vector<double> cpu_total;
            std::vector<double> cpu_prepare;
            std::vector<double> cpu_search;

            std::vector<double> gpu_total;
            std::vector<double> gpu_prepare_wall;
            std::vector<double> gpu_prepare_device;
            std::vector<double> gpu_search;

            lazy_total.reserve(
                sample_count
            );

            lazy_search.reserve(
                sample_count
            );

            cpu_total.reserve(
                sample_count
            );

            cpu_prepare.reserve(
                sample_count
            );

            cpu_search.reserve(
                sample_count
            );

            gpu_total.reserve(
                sample_count
            );

            gpu_prepare_wall.reserve(
                sample_count
            );

            gpu_prepare_device.reserve(
                sample_count
            );

            gpu_search.reserve(
                sample_count
            );

            bool all_gpu_device_timings =
                true;

            for (
                std::size_t round = 0;
                round < kMeasuredRounds;
                ++round
            ) {

                std::cout
                    << "[G-UMA5B] N="
                    << vector_count
                    << " medicion "
                    << (
                        round +
                        1U
                    )
                    << "/"
                    << kMeasuredRounds
                    << '\n'
                    << std::flush;

                for (
                    std::size_t query_index = 0;
                    query_index < kQueryCount;
                    ++query_index
                ) {

                    const FilterSpec filter =
                        filter_for_query(
                            query_index
                        );

                    const LazyPredicate
                        lazy_predicate{
                            .metadata =
                                metadata_view,

                            .filter =
                                filter
                        };

                    float const* query =
                        queries.row(
                            query_index
                        );

                    auto measure_lazy =
                        [&]() {

                            const double search_ms =
                                timed_racorn_search(
                                    index,
                                    query,
                                    metric,
                                    lazy_predicate,
                                    output
                                        .benchmark_sink
                                );

                            lazy_search.push_back(
                                search_ms
                            );

                            lazy_total.push_back(
                                search_ms
                            );
                        };

                    auto measure_cpu_bitset =
                        [&]() {

                            const auto total_start =
                                Clock::now();

                            const auto prepare_start =
                                Clock::now();

                            materialize_cpu_bitset(
                                metadata_view,
                                vector_count,
                                filter,
                                cpu_bitset
                            );

                            const auto prepare_end =
                                Clock::now();

                            const double search_ms =
                                timed_racorn_search(
                                    index,
                                    query,
                                    metric,
                                    cpu_bitset_predicate,
                                    output
                                        .benchmark_sink
                                );

                            const auto total_end =
                                Clock::now();

                            cpu_prepare.push_back(
                                milliseconds_between(
                                    prepare_start,
                                    prepare_end
                                )
                            );

                            cpu_search.push_back(
                                search_ms
                            );

                            cpu_total.push_back(
                                milliseconds_between(
                                    total_start,
                                    total_end
                                )
                            );
                        };

                    auto measure_gpu_bitset =
                        [&]() -> bool {

                            const auto total_start =
                                Clock::now();

                            const GpuPrepareSample prepare =
                                run_gpu_prepare(
                                    command_queue,
                                    pipeline,
                                    metadata_buffer,
                                    gpu_bitset_buffer,
                                    column_stride_bytes,
                                    vector_count,
                                    filter,
                                    output
                                        .threads_per_threadgroup
                                );

                            if (
                                !prepare.completed
                            ) {

                                output.metal_error =
                                    prepare.error;

                                return false;
                            }

                            const double search_ms =
                                timed_racorn_search(
                                    index,
                                    query,
                                    metric,
                                    gpu_bitset_predicate,
                                    output
                                        .benchmark_sink
                                );

                            const auto total_end =
                                Clock::now();

                            gpu_prepare_wall.push_back(
                                prepare.wall_ms
                            );

                            gpu_search.push_back(
                                search_ms
                            );

                            gpu_total.push_back(
                                milliseconds_between(
                                    total_start,
                                    total_end
                                )
                            );

                            if (
                                prepare.device_timing_available
                            ) {

                                gpu_prepare_device.push_back(
                                    prepare.device_ms
                                );

                            } else {

                                all_gpu_device_timings =
                                    false;
                            }

                            return true;
                        };

                    const std::size_t order =
                        (
                            round +
                            query_index
                        ) %
                        3U;

                    if (
                        order == 0U
                    ) {

                        measure_lazy();
                        measure_cpu_bitset();

                        if (
                            !measure_gpu_bitset()
                        ) {
                            return output;
                        }

                    } else if (
                        order == 1U
                    ) {

                        measure_cpu_bitset();

                        if (
                            !measure_gpu_bitset()
                        ) {
                            return output;
                        }

                        measure_lazy();

                    } else {

                        if (
                            !measure_gpu_bitset()
                        ) {
                            return output;
                        }

                        measure_lazy();
                        measure_cpu_bitset();
                    }
                }
            }

            const std::vector<double>
                empty_samples;

            fill_latency_stats(
                case_result.cpu_lazy,
                lazy_total,
                empty_samples,
                lazy_search,
                empty_samples
            );

            fill_latency_stats(
                case_result.cpu_bitset,
                cpu_total,
                cpu_prepare,
                cpu_search,
                empty_samples
            );

            if (
                !all_gpu_device_timings
            ) {
                gpu_prepare_device.clear();
            }

            fill_latency_stats(
                case_result.gpu_bitset,
                gpu_total,
                gpu_prepare_wall,
                gpu_search,
                gpu_prepare_device
            );

            if (
                case_result.cpu_bitset.total_p50_ms >
                0.0
            ) {

                case_result.cpu_bitset_speedup_vs_lazy =
                    case_result.cpu_lazy.total_p50_ms /
                    case_result.cpu_bitset.total_p50_ms;
            }

            if (
                case_result.gpu_bitset.total_p50_ms >
                0.0
            ) {

                case_result.gpu_bitset_speedup_vs_lazy =
                    case_result.cpu_lazy.total_p50_ms /
                    case_result.gpu_bitset.total_p50_ms;
            }

            const double lazy_p50 =
                case_result.cpu_lazy.total_p50_ms;

            const double cpu_p50 =
                case_result.cpu_bitset.total_p50_ms;

            const double gpu_p50 =
                case_result.gpu_bitset.total_p50_ms;

            if (
                lazy_p50 <= cpu_p50 &&
                lazy_p50 <= gpu_p50
            ) {

                case_result.p50_winner =
                    "CPU-Lazy";

            } else if (
                cpu_p50 <= gpu_p50
            ) {

                case_result.p50_winner =
                    "CPU-Bitset";

            } else {

                case_result.p50_winner =
                    "GPU-Bitset";
            }

            case_result.valid =
                case_result.navigation_semantics_identical &&
                case_result.exact_ground_truth_complete &&
                case_result.filter_results_valid &&
                lazy_total.size() ==
                    sample_count &&
                cpu_total.size() ==
                    sample_count &&
                gpu_total.size() ==
                    sample_count;

            all_cases_identical =
                all_cases_identical &&
                case_result.valid;

            output.cases.push_back(
                case_result
            );

            std::cout
                << "[G-UMA5B] N="
                << vector_count
                << " completado.\n"
                << std::flush;
        }

        output.all_cases_semantically_identical =
            all_cases_identical;

        output.valid =
            output.metal_available &&
            output.unified_memory &&
            output.library_available &&
            output.kernel_available &&
            output.pipeline_available &&
            output.command_queue_available &&
            output.all_cases_semantically_identical &&
            output.cases.size() ==
                kVectorCounts.size() &&
            !output.racorn_modified;
    }

    print_progress(
        "[G-UMA5B] Benchmark finalizado. Generando resumen..."
    );

    return output;
}

} // namespace racorn_uma::platform::apple

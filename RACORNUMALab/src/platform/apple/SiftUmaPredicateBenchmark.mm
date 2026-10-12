//
//  SiftUmaPredicateBenchmark.mm
//  RACORNUMALab
//
//  Propósito:
//  Implementar G-DATA3A sobre SIFT100K real.
//
//  Comparación end-to-end:
//
//      CPU-Lazy
//          RACORN -> predicado SoA
//
//      CPU-Bitset
//          CPU -> bitset Shared -> RACORN
//
//      GPU-Bitset
//          Metal -> bitset Shared -> RACORN
//
//  Semántica:
//  Las tres rutas deben producir exactamente:
//
//      - mismo result_count;
//      - mismo top-k ordenado;
//      - mismo Recall@k;
//      - mismo computed_distances;
//      - mismo visited_members;
//      - mismo predicate_evaluations;
//      - mismo fallback_count.
//
//  Validación:
//      1000 queries.
//
//  Benchmark:
//      primeras 256 queries;
//      1 warm-up;
//      3 rondas medidas;
//      orden de estrategias rotado.
//
//  Configuración:
//      N   = 100000
//      D   = 128
//      M   = 16
//      efC = 100
//      efS = 200
//      k   = 10
//      BR  = 1.0
//      TT  = 0.30
//
//  RACORN-1+:
//      desactivado:
//          fallback_exact_threshold = 0.
//
//  Invariantes:
//  - RACORN upstream intacto.
//  - HNSW/RACORN siempre en CPU.
//  - GPU solo materializa el bitset.
//  - Metadata y bitsets usan MTLResourceStorageModeShared.
//  - Índice fuera del timing.
//  - Dataset fuera del repositorio.
//

#include "platform/apple/SiftUmaPredicateBenchmark.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
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
    "racorn_sift_predicate_to_bitset";

constexpr char const* kBaseFilename =
    "sift100k_base_f32le.bin";

constexpr char const* kQueryFilename =
    "sift10k_query_f32le.bin";

constexpr char const* kMetadataFilename =
    "sift100k_metadata_u32le_soa.bin";

constexpr std::size_t kVectorCount =
    100'000U;

constexpr std::size_t kDimensions =
    128U;

constexpr std::size_t kStoredQueryCount =
    10'000U;

constexpr std::size_t kSemanticQueryCount =
    1'000U;

constexpr std::size_t kBenchmarkQueryCount =
    256U;

constexpr std::size_t kTenantCount =
    16U;

constexpr std::size_t kMetadataColumnCount =
    5U;

constexpr std::size_t kConnectivity =
    16U;

constexpr std::size_t kEfConstruction =
    100U;

constexpr std::size_t kEfSearch =
    200U;

constexpr std::size_t kNeighbors =
    10U;

constexpr std::size_t kWarmupRounds =
    1U;

constexpr std::size_t kMeasuredRounds =
    3U;

constexpr std::size_t kBitsPerWord =
    32U;

constexpr double kTwoHopThreshold =
    0.30;

constexpr double kBridgeBudgetRatio =
    1.00;

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

struct ExactCandidate final {

    float distance{};
    std::size_t key{};
};

struct SearchSnapshot final {

    std::vector<std::size_t> keys;

    std::size_t computed_distances{};
    std::size_t visited_members{};
    std::size_t predicate_evaluations{};
    std::size_t fallback_count{};
};

struct GpuPrepareSample final {

    bool completed{};
    bool device_timing_available{};

    double wall_ms{};
    double device_ms{};

    std::string error;
};

class DenseVectors final {

public:

    DenseVectors(
        std::vector<float> values,
        std::size_t dimensions
    )
        : values_(
              std::move(
                  values
              )
          ),
          dimensions_(
              dimensions
          ) {
    }

    [[nodiscard]]
    float const* row(
        std::size_t index
    ) const noexcept {

        return
            values_.data() +
            index *
                dimensions_;
    }

private:

    std::vector<float> values_;
    std::size_t dimensions_{};
};

struct L2Metric final {

    using IndexType =
        Index;

    using MemberRef =
        typename IndexType::member_cref_t;

    using MemberIterator =
        typename IndexType::member_citerator_t;

    DenseVectors const* vectors{};
    std::size_t dimensions{};

    [[nodiscard]]
    float const* row(
        std::size_t key
    ) const noexcept {

        return
            vectors->row(
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
            pointer_ ==
            nullptr
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

bool evaluate_filter(
    MetadataView const& metadata,
    std::size_t key,
    FilterSpec const& filter
) noexcept {

    const std::uint32_t category =
        metadata.category[key];

    const bool category_allowed =
        category == 3U ||
        category == 7U ||
        category == 9U;

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
            key >>
            5U;

        const std::size_t bit =
            key &
            31U;

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
        [
            value
            UTF8String
        ];

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

void ensure_little_endian() {

    if constexpr (
        std::endian::native !=
        std::endian::little
    ) {

        throw std::runtime_error(
            "G-DATA3A requiere little-endian."
        );
    }
}

void validate_file_size(
    std::filesystem::path const& path,
    std::uintmax_t expected
) {

    if (
        !std::filesystem::exists(
            path
        ) ||
        !std::filesystem::is_regular_file(
            path
        )
    ) {

        throw std::runtime_error(
            "Archivo inexistente o invalido: " +
            path.string()
        );
    }

    const std::uintmax_t actual =
        std::filesystem::file_size(
            path
        );

    if (
        actual !=
        expected
    ) {

        throw std::runtime_error(
            "Tamaño inesperado en " +
            path.filename().string() +
            ": observado=" +
            std::to_string(
                actual
            ) +
            ", esperado=" +
            std::to_string(
                expected
            )
        );
    }
}

template <typename T>
std::vector<T> read_binary_exact(
    std::filesystem::path const& path,
    std::size_t element_count
) {

    std::ifstream input(
        path,
        std::ios::binary
    );

    if (
        !input
    ) {

        throw std::runtime_error(
            "No se pudo abrir: " +
            path.string()
        );
    }

    std::vector<T> output(
        element_count
    );

    const std::size_t bytes =
        element_count *
        sizeof(
            T
        );

    input.read(
        reinterpret_cast<char*>(
            output.data()
        ),
        static_cast<std::streamsize>(
            bytes
        )
    );

    if (
        input.gcount() !=
        static_cast<std::streamsize>(
            bytes
        )
    ) {

        throw std::runtime_error(
            "Lectura incompleta: " +
            path.string()
        );
    }

    return output;
}

void read_binary_into(
    std::filesystem::path const& path,
    void* destination,
    std::size_t bytes
) {

    if (
        destination ==
        nullptr
    ) {

        throw std::runtime_error(
            "Destino nulo al cargar archivo."
        );
    }

    std::ifstream input(
        path,
        std::ios::binary
    );

    if (
        !input
    ) {

        throw std::runtime_error(
            "No se pudo abrir: " +
            path.string()
        );
    }

    input.read(
        static_cast<char*>(
            destination
        ),
        static_cast<std::streamsize>(
            bytes
        )
    );

    if (
        input.gcount() !=
        static_cast<std::streamsize>(
            bytes
        )
    ) {

        throw std::runtime_error(
            "Lectura incompleta: " +
            path.string()
        );
    }
}

std::vector<float> read_query_prefix(
    std::filesystem::path const& path
) {

    const std::size_t element_count =
        kSemanticQueryCount *
        kDimensions;

    std::ifstream input(
        path,
        std::ios::binary
    );

    if (
        !input
    ) {

        throw std::runtime_error(
            "No se pudo abrir queries."
        );
    }

    std::vector<float> output(
        element_count
    );

    const std::size_t bytes =
        element_count *
        sizeof(
            float
        );

    input.read(
        reinterpret_cast<char*>(
            output.data()
        ),
        static_cast<std::streamsize>(
            bytes
        )
    );

    if (
        input.gcount() !=
        static_cast<std::streamsize>(
            bytes
        )
    ) {

        throw std::runtime_error(
            "Lectura incompleta de queries."
        );
    }

    return output;
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
                kTenantCount
            ),

        .language_min =
            2U
    };
}

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
                key >=
                element_count
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

        output[
            word_index
        ] =
            word;
    }
}

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

std::vector<std::size_t>
exact_filtered_top_k(
    DenseVectors const& base,
    float const* query,
    std::vector<std::size_t> const& eligible
) {

    if (
        eligible.size() <
        kNeighbors
    ) {

        return {};
    }

    std::vector<ExactCandidate>
        candidates;

    candidates.reserve(
        eligible.size()
    );

    for (
        std::size_t key :
        eligible
    ) {

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
                kNeighbors
            ),
        candidates.end(),
        comparator
    );

    std::vector<std::size_t>
        result;

    result.reserve(
        kNeighbors
    );

    for (
        std::size_t index = 0;
        index < kNeighbors;
        ++index
    ) {

        result.push_back(
            candidates[index]
                .key
        );
    }

    return result;
}

index_search_config_t
make_search_configuration() {

    index_search_config_t configuration;

    configuration.expansion =
        kEfSearch;

    configuration.thread =
        0;

    configuration.two_hop_threshold =
        kTwoHopThreshold;

    configuration.racorn =
        true;

    configuration.bridge_budget_ratio =
        kBridgeBudgetRatio;

    configuration.three_hop_threshold =
        -1.0;

    configuration.fallback_exact_threshold =
        0.0;

    configuration.racorn_stride_sampling =
        true;

    return configuration;
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
        0U;

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
        0U;

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
        0U
    ) {

        return 0U;
    }

    constexpr std::size_t preferred =
        256U;

    std::size_t selected =
        std::min(
            preferred,
            maximum
        );

    if (
        execution_width >
        0U
    ) {

        selected =
            (
                selected /
                execution_width
            ) *
            execution_width;

        if (
            selected ==
            0U
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
        values.size() ==
        1U
    ) {

        return
            values.front();
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
            values[
                lower
            ];
    }

    const double fraction =
        position -
        static_cast<double>(
            lower
        );

    return
        values[
            lower
        ] *
            (
                1.0 -
                fraction
            ) +
        values[
            upper
        ] *
            fraction;
}

GpuPrepareSample run_gpu_prepare(
    id<MTLCommandQueue> command_queue,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> metadata_buffer,
    id<MTLBuffer> bitset_buffer,
    std::size_t column_stride_bytes,
    std::size_t element_count,
    FilterSpec const& filter,
    std::size_t threads_per_group
) {

    GpuPrepareSample sample;

    @autoreleasepool {

        const auto wall_start =
            Clock::now();

        id<MTLCommandBuffer> command_buffer =
            [
                command_queue
                commandBuffer
            ];

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

        for (
            NSUInteger column = 0;
            column < 5U;
            ++column
        ) {

            [
                encoder
                setBuffer:
                    metadata_buffer
                offset:
                    column *
                    column_stride_bytes
                atIndex:
                    column
            ];
        }

        [
            encoder
            setBuffer:
                bitset_buffer
            offset:
                0U
            atIndex:
                5U
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
                6U
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
                7U
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
                8U
        ];

        const MTLSize grid =
            MTLSizeMake(
                bitset_word_count_for(
                    element_count
                ),
                1U,
                1U
            );

        const MTLSize group =
            MTLSizeMake(
                threads_per_group,
                1U,
                1U
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
                    "CommandBuffer GPU incompleto.";
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
        make_search_configuration();

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
        result.predicate_evaluations +
        result.fallback_count;

    return
        milliseconds_between(
            start,
            end
        );
}

void fill_latency_stats(
    SiftUmaLatencyStats& output,
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

    output.total_p99_ms =
        percentile(
            total,
            0.99
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

    output.prepare_p99_ms =
        percentile(
            prepare,
            0.99
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

    output.search_p99_ms =
        percentile(
            search,
            0.99
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

        output.gpu_device_p99_ms =
            percentile(
                device,
                0.99
            );
    }

    if (
        output.total_p50_ms >
        0.0
    ) {

        output.qps_from_p50 =
            1000.0 /
            output.total_p50_ms;
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

SiftUmaPredicateBenchmarkResult
SiftUmaPredicateBenchmark::run(
    std::filesystem::path const& data_directory
) {

    ensure_little_endian();

    SiftUmaPredicateBenchmarkResult output;

    output.data_directory =
        data_directory;

    output.vector_count =
        kVectorCount;

    output.dimensions =
        kDimensions;

    output.semantic_query_count =
        kSemanticQueryCount;

    output.benchmark_query_count =
        kBenchmarkQueryCount;

    output.connectivity =
        kConnectivity;

    output.ef_construction =
        kEfConstruction;

    output.ef_search =
        kEfSearch;

    output.k =
        kNeighbors;

    output.bridge_budget_ratio =
        kBridgeBudgetRatio;

    output.two_hop_threshold =
        kTwoHopThreshold;

    output.warmup_rounds =
        kWarmupRounds;

    output.measured_rounds =
        kMeasuredRounds;

    output.kernel_name =
        kKernelName;

    output.racorn_modified =
        false;

    const auto base_path =
        data_directory /
        kBaseFilename;

    const auto query_path =
        data_directory /
        kQueryFilename;

    const auto metadata_path =
        data_directory /
        kMetadataFilename;

    print_progress(
        "[G-DATA3A] Validando archivos..."
    );

    validate_file_size(
        base_path,
        static_cast<std::uintmax_t>(
            kVectorCount
        ) *
            kDimensions *
            sizeof(
                float
            )
    );

    output.base_file_valid =
        true;

    validate_file_size(
        query_path,
        static_cast<std::uintmax_t>(
            kStoredQueryCount
        ) *
            kDimensions *
            sizeof(
                float
            )
    );

    output.query_file_valid =
        true;

    validate_file_size(
        metadata_path,
        static_cast<std::uintmax_t>(
            kMetadataColumnCount
        ) *
            kVectorCount *
            sizeof(
                std::uint32_t
            )
    );

    output.metadata_file_valid =
        true;

    print_progress(
        "[G-DATA3A] Cargando SIFT100K..."
    );

    DenseVectors base(
        read_binary_exact<float>(
            base_path,
            kVectorCount *
                kDimensions
        ),
        kDimensions
    );

    print_progress(
        "[G-DATA3A] Cargando 1000 queries..."
    );

    DenseVectors queries(
        read_query_prefix(
            query_path
        ),
        kDimensions
    );

    print_progress(
        "[G-DATA3A] Inicializando Metal..."
    );

    @autoreleasepool {

        id<MTLDevice> device =
            MTLCreateSystemDefaultDevice();

        if (
            device ==
            nil
        ) {

            output.metal_error =
                "Metal no disponible.";

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
            [
                device
                newDefaultLibrary
            ];

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
            [
                NSString
                stringWithUTF8String:
                    kKernelName
            ];

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
                "No se encontro racorn_sift_predicate_to_bitset.";

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
                    "No se pudo crear pipeline Metal.";
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
            0U
        ) {

            output.metal_error =
                "Threadgroup Metal invalido.";

            return output;
        }

        id<MTLCommandQueue> command_queue =
            [
                device
                newCommandQueue
            ];

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
        // Metadata directamente en memoria UMA Shared.
        //

        const std::size_t column_stride_bytes =
            kVectorCount *
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
            metadata_buffer ==
            nil ||
            metadata_buffer.contents ==
            nullptr
        ) {

            output.metal_error =
                "No se pudo crear metadata Shared.";

            return output;
        }

        print_progress(
            "[G-DATA3A] Cargando metadata directamente en Shared..."
        );

        read_binary_into(
            metadata_path,
            metadata_buffer.contents,
            metadata_bytes
        );

        auto* metadata_base =
            static_cast<std::uint32_t*>(
                metadata_buffer.contents
            );

        const MetadataView metadata{
            .tenant =
                metadata_base +
                0U *
                    kVectorCount,

            .category =
                metadata_base +
                1U *
                    kVectorCount,

            .year =
                metadata_base +
                2U *
                    kVectorCount,

            .language =
                metadata_base +
                3U *
                    kVectorCount,

            .security_level =
                metadata_base +
                4U *
                    kVectorCount
        };

        const std::size_t word_count =
            bitset_word_count_for(
                kVectorCount
            );

        const std::size_t bitset_bytes =
            word_count *
            sizeof(
                std::uint32_t
            );

        id<MTLBuffer> cpu_bitset_buffer =
            [
                device
                newBufferWithLength:
                    bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        id<MTLBuffer> gpu_bitset_buffer =
            [
                device
                newBufferWithLength:
                    bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        if (
            cpu_bitset_buffer ==
                nil ||
            gpu_bitset_buffer ==
                nil ||
            cpu_bitset_buffer.contents ==
                nullptr ||
            gpu_bitset_buffer.contents ==
                nullptr
        ) {

            output.metal_error =
                "No se pudieron crear bitsets Shared.";

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

        //
        // Precomputar conjuntos elegibles para GT.
        //

        print_progress(
            "[G-DATA3A] Precomputando candidatos filtrados..."
        );

        std::array<
            std::vector<std::size_t>,
            kTenantCount
        >
            eligible_by_tenant;

        for (
            std::size_t tenant_index = 0;
            tenant_index < kTenantCount;
            ++tenant_index
        ) {

            const LazyPredicate predicate{
                .metadata =
                    metadata,

                .filter =
                    FilterSpec{
                        .tenant_required =
                            static_cast<std::uint32_t>(
                                tenant_index
                            ),

                        .language_min =
                            2U
                    }
            };

            auto& eligible =
                eligible_by_tenant[
                    tenant_index
                ];

            for (
                std::size_t key = 0;
                key < kVectorCount;
                ++key
            ) {

                if (
                    predicate.accepts_key(
                        key
                    )
                ) {

                    eligible.push_back(
                        key
                    );
                }
            }
        }

        output.selected_min =
            std::numeric_limits<
                std::size_t
            >::max();

        output.selected_max =
            0U;

        std::size_t selected_sum =
            0U;

        for (
            std::size_t query_index = 0;
            query_index < kSemanticQueryCount;
            ++query_index
        ) {

            const auto& eligible =
                eligible_by_tenant[
                    query_index %
                    kTenantCount
                ];

            output.selected_min =
                std::min(
                    output.selected_min,
                    eligible.size()
                );

            output.selected_max =
                std::max(
                    output.selected_max,
                    eligible.size()
                );

            selected_sum +=
                eligible.size();
        }

        output.mean_selectivity =
            static_cast<double>(
                selected_sum
            ) /
            (
                static_cast<double>(
                    kSemanticQueryCount
                ) *
                static_cast<double>(
                    kVectorCount
                )
            );

        //
        // Validar representación de los 16 filtros únicos.
        //

        print_progress(
            "[G-DATA3A] Validando CPU/GPU bitset para 16 filtros unicos..."
        );

        for (
            std::size_t tenant_index = 0;
            tenant_index < kTenantCount;
            ++tenant_index
        ) {

            const FilterSpec filter{
                .tenant_required =
                    static_cast<std::uint32_t>(
                        tenant_index
                    ),

                .language_min =
                    2U
            };

            const LazyPredicate direct{
                .metadata =
                    metadata,

                .filter =
                    filter
            };

            materialize_cpu_bitset(
                metadata,
                kVectorCount,
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
                    kVectorCount,
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

            output.bitset_word_mismatches +=
                mismatch_word_count(
                    cpu_bitset,
                    gpu_bitset,
                    word_count
                );

            for (
                std::size_t key = 0;
                key < kVectorCount;
                ++key
            ) {

                const bool expected =
                    direct.accepts_key(
                        key
                    );

                if (
                    expected !=
                        cpu_bitset_predicate
                            .accepts_key(
                                key
                            ) ||
                    expected !=
                        gpu_bitset_predicate
                            .accepts_key(
                                key
                            )
                ) {

                    ++output
                        .predicate_representation_mismatches;
                }
            }
        }

        //
        // Construir HNSW.
        //

        L2Metric metric{
            .vectors =
                &base,

            .dimensions =
                kDimensions
        };

        print_progress(
            "[G-DATA3A] Construyendo HNSW SIFT100K..."
        );

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
                kVectorCount,
                1
            )
        );

        index_update_config_t
            update_configuration;

        update_configuration.expansion =
            kEfConstruction;

        update_configuration.thread =
            0;

        constexpr std::size_t progress_step =
            kVectorCount /
            10U;

        for (
            std::size_t key = 0;
            key < kVectorCount;
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
                    0U
            ) {

                std::cout
                    << "[G-DATA3A] HNSW "
                    << (
                        (
                            key +
                            1U
                        ) *
                        100U /
                        kVectorCount
                    )
                    << "%\n"
                    << std::flush;
            }
        }

        //
        // Ground truth exacto filtrado.
        //

        print_progress(
            "[G-DATA3A] Construyendo ground truth exacto..."
        );

        std::vector<
            std::vector<std::size_t>
        >
            exact_by_query;

        exact_by_query.reserve(
            kSemanticQueryCount
        );

        bool exact_complete =
            true;

        for (
            std::size_t query_index = 0;
            query_index < kSemanticQueryCount;
            ++query_index
        ) {

            const auto& eligible =
                eligible_by_tenant[
                    query_index %
                    kTenantCount
                ];

            auto exact =
                exact_filtered_top_k(
                    base,
                    queries.row(
                        query_index
                    ),
                    eligible
                );

            if (
                exact.size() !=
                kNeighbors
            ) {

                exact_complete =
                    false;
            }

            exact_by_query.push_back(
                std::move(
                    exact
                )
            );

            if (
                (
                    query_index +
                    1U
                ) %
                    100U ==
                    0U
            ) {

                std::cout
                    << "[G-DATA3A] GT "
                    << (
                        query_index +
                        1U
                    )
                    << "/"
                    << kSemanticQueryCount
                    << '\n'
                    << std::flush;
            }
        }

        output.exact_ground_truth_complete =
            exact_complete;

        //
        // Validación semántica 1000 queries.
        //

        print_progress(
            "[G-DATA3A] Validando semantica sobre 1000 queries..."
        );

        bool filter_valid =
            true;

        double recall_sum =
            0.0;

        double minimum_recall =
            1.0;

        double distances_sum =
            0.0;

        double visited_sum =
            0.0;

        double predicate_sum =
            0.0;

        double fallback_sum =
            0.0;

        for (
            std::size_t query_index = 0;
            query_index < kSemanticQueryCount;
            ++query_index
        ) {

            const FilterSpec filter =
                filter_for_query(
                    query_index
                );

            const LazyPredicate
                lazy_predicate{
                    .metadata =
                        metadata,

                    .filter =
                        filter
                };

            materialize_cpu_bitset(
                metadata,
                kVectorCount,
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
                    kVectorCount,
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

            output.bitset_word_mismatches +=
                mismatch_word_count(
                    cpu_bitset,
                    gpu_bitset,
                    word_count
                );

            float const* query =
                queries.row(
                    query_index
                );

            auto lazy_configuration =
                make_search_configuration();

            auto lazy_result =
                index.search(
                    query,
                    kNeighbors,
                    metric,
                    lazy_configuration,
                    lazy_predicate
                );

            const SearchSnapshot lazy =
                snapshot_search_result(
                    lazy_result
                );

            auto cpu_configuration =
                make_search_configuration();

            auto cpu_result =
                index.search(
                    query,
                    kNeighbors,
                    metric,
                    cpu_configuration,
                    cpu_bitset_predicate
                );

            const SearchSnapshot cpu =
                snapshot_search_result(
                    cpu_result
                );

            auto gpu_configuration =
                make_search_configuration();

            auto gpu_result =
                index.search(
                    query,
                    kNeighbors,
                    metric,
                    gpu_configuration,
                    gpu_bitset_predicate
                );

            const SearchSnapshot gpu =
                snapshot_search_result(
                    gpu_result
                );

            const double lazy_recall =
                recall_at_k(
                    lazy.keys,
                    exact_by_query[
                        query_index
                    ]
                );

            const double cpu_recall =
                recall_at_k(
                    cpu.keys,
                    exact_by_query[
                        query_index
                    ]
                );

            const double gpu_recall =
                recall_at_k(
                    gpu.keys,
                    exact_by_query[
                        query_index
                    ]
                );

            recall_sum +=
                lazy_recall;

            minimum_recall =
                std::min(
                    minimum_recall,
                    lazy_recall
                );

            distances_sum +=
                static_cast<double>(
                    lazy
                        .computed_distances
                );

            visited_sum +=
                static_cast<double>(
                    lazy
                        .visited_members
                );

            predicate_sum +=
                static_cast<double>(
                    lazy
                        .predicate_evaluations
                );

            fallback_sum +=
                static_cast<double>(
                    lazy
                        .fallback_count
                );

            if (
                lazy.keys.size() ==
                    kNeighbors &&
                same_double(
                    lazy_recall,
                    1.0
                )
            ) {

                ++output
                    .perfect_queries;

            } else {

                ++output
                    .imperfect_queries;
            }

            if (
                lazy.keys.size() <
                kNeighbors
            ) {

                ++output
                    .underfilled_queries;
            }

            filter_valid =
                filter_valid &&

                result_satisfies_filter(
                    lazy.keys,
                    lazy_predicate
                ) &&

                result_satisfies_filter(
                    cpu.keys,
                    lazy_predicate
                ) &&

                result_satisfies_filter(
                    gpu.keys,
                    lazy_predicate
                );

            if (
                lazy.keys !=
                    cpu.keys ||
                lazy.keys !=
                    gpu.keys
            ) {

                ++output
                    .topk_mismatch_queries;
            }

            if (
                lazy.keys.size() !=
                    cpu.keys.size() ||
                lazy.keys.size() !=
                    gpu.keys.size()
            ) {

                ++output
                    .result_count_mismatch_queries;
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

                ++output
                    .recall_mismatch_queries;
            }

            if (
                lazy.computed_distances !=
                    cpu.computed_distances ||
                lazy.computed_distances !=
                    gpu.computed_distances
            ) {

                ++output
                    .computed_distances_mismatch_queries;
            }

            if (
                lazy.visited_members !=
                    cpu.visited_members ||
                lazy.visited_members !=
                    gpu.visited_members
            ) {

                ++output
                    .visited_members_mismatch_queries;
            }

            if (
                lazy.predicate_evaluations !=
                    cpu.predicate_evaluations ||
                lazy.predicate_evaluations !=
                    gpu.predicate_evaluations
            ) {

                ++output
                    .predicate_evaluations_mismatch_queries;
            }

            if (
                lazy.fallback_count !=
                    cpu.fallback_count ||
                lazy.fallback_count !=
                    gpu.fallback_count
            ) {

                ++output
                    .fallback_count_mismatch_queries;
            }

            if (
                (
                    query_index +
                    1U
                ) %
                    100U ==
                    0U
            ) {

                std::cout
                    << "[G-DATA3A] Semantica "
                    << (
                        query_index +
                        1U
                    )
                    << "/"
                    << kSemanticQueryCount
                    << '\n'
                    << std::flush;
            }
        }

        const double semantic_query_count =
            static_cast<double>(
                kSemanticQueryCount
            );

        output.mean_recall_at_k =
            recall_sum /
            semantic_query_count;

        output.min_recall_at_k =
            minimum_recall;

        output.mean_computed_distances =
            distances_sum /
            semantic_query_count;

        output.mean_visited_members =
            visited_sum /
            semantic_query_count;

        output.mean_predicate_evaluations =
            predicate_sum /
            semantic_query_count;

        output.mean_fallback_count =
            fallback_sum /
            semantic_query_count;

        output.filter_results_valid =
            filter_valid;

        output.navigation_semantics_identical =
            output.bitset_word_mismatches ==
                0U &&

            output.predicate_representation_mismatches ==
                0U &&

            output.topk_mismatch_queries ==
                0U &&

            output.result_count_mismatch_queries ==
                0U &&

            output.recall_mismatch_queries ==
                0U &&

            output.computed_distances_mismatch_queries ==
                0U &&

            output.visited_members_mismatch_queries ==
                0U &&

            output.predicate_evaluations_mismatch_queries ==
                0U &&

            output.fallback_count_mismatch_queries ==
                0U;

        if (
            !output.navigation_semantics_identical ||
            !output.exact_ground_truth_complete ||
            !output.filter_results_valid
        ) {

            return output;
        }

        //
        // Benchmark.
        //

        print_progress(
            "[G-DATA3A] Semantica PASS. Warm-up..."
        );

        for (
            std::size_t round = 0;
            round < kWarmupRounds;
            ++round
        ) {

            for (
                std::size_t query_index = 0;
                query_index < kBenchmarkQueryCount;
                ++query_index
            ) {

                const FilterSpec filter =
                    filter_for_query(
                        query_index
                    );

                const LazyPredicate lazy{
                    .metadata =
                        metadata,

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
                    lazy,
                    output.benchmark_sink
                );

                materialize_cpu_bitset(
                    metadata,
                    kVectorCount,
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

                const GpuPrepareSample prepare =
                    run_gpu_prepare(
                        command_queue,
                        pipeline,
                        metadata_buffer,
                        gpu_bitset_buffer,
                        column_stride_bytes,
                        kVectorCount,
                        filter,
                        output
                            .threads_per_threadgroup
                    );

                if (
                    !prepare.completed
                ) {

                    output.metal_error =
                        prepare.error;

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
            "[G-DATA3A] Midiendo 256 queries x 3 rondas..."
        );

        const std::size_t sample_count =
            kBenchmarkQueryCount *
            kMeasuredRounds;

        std::vector<double>
            lazy_total;

        std::vector<double>
            lazy_search;

        std::vector<double>
            cpu_total;

        std::vector<double>
            cpu_prepare;

        std::vector<double>
            cpu_search;

        std::vector<double>
            gpu_total;

        std::vector<double>
            gpu_prepare_wall;

        std::vector<double>
            gpu_prepare_device;

        std::vector<double>
            gpu_search;

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
                << "[G-DATA3A] Ronda "
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
                query_index < kBenchmarkQueryCount;
                ++query_index
            ) {

                const FilterSpec filter =
                    filter_for_query(
                        query_index
                    );

                const LazyPredicate lazy{
                    .metadata =
                        metadata,

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
                                lazy,
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

                auto measure_cpu =
                    [&]() {

                        const auto total_start =
                            Clock::now();

                        const auto prepare_start =
                            Clock::now();

                        materialize_cpu_bitset(
                            metadata,
                            kVectorCount,
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

                auto measure_gpu =
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
                                kVectorCount,
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
                    order ==
                    0U
                ) {

                    measure_lazy();
                    measure_cpu();

                    if (
                        !measure_gpu()
                    ) {

                        return output;
                    }

                } else if (
                    order ==
                    1U
                ) {

                    measure_cpu();

                    if (
                        !measure_gpu()
                    ) {

                        return output;
                    }

                    measure_lazy();

                } else {

                    if (
                        !measure_gpu()
                    ) {

                        return output;
                    }

                    measure_lazy();
                    measure_cpu();
                }
            }
        }

        const std::vector<double>
            empty_samples;

        fill_latency_stats(
            output.cpu_lazy,
            lazy_total,
            empty_samples,
            lazy_search,
            empty_samples
        );

        fill_latency_stats(
            output.cpu_bitset,
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
            output.gpu_bitset,
            gpu_total,
            gpu_prepare_wall,
            gpu_search,
            gpu_prepare_device
        );

        if (
            output.cpu_bitset.total_p50_ms >
            0.0
        ) {

            output.cpu_bitset_speedup_vs_lazy =
                output.cpu_lazy.total_p50_ms /
                output.cpu_bitset.total_p50_ms;
        }

        if (
            output.gpu_bitset.total_p50_ms >
            0.0
        ) {

            output.gpu_bitset_speedup_vs_lazy =
                output.cpu_lazy.total_p50_ms /
                output.gpu_bitset.total_p50_ms;
        }

        if (
            output.gpu_bitset.total_p50_ms >
            0.0
        ) {

            output.gpu_vs_cpu_bitset =
                output.cpu_bitset.total_p50_ms /
                output.gpu_bitset.total_p50_ms;
        }

        const double lazy_p50 =
            output.cpu_lazy.total_p50_ms;

        const double cpu_p50 =
            output.cpu_bitset.total_p50_ms;

        const double gpu_p50 =
            output.gpu_bitset.total_p50_ms;

        if (
            lazy_p50 <=
                cpu_p50 &&
            lazy_p50 <=
                gpu_p50
        ) {

            output.p50_winner =
                "CPU-Lazy";

        } else if (
            cpu_p50 <=
            gpu_p50
        ) {

            output.p50_winner =
                "CPU-Bitset";

        } else {

            output.p50_winner =
                "GPU-Bitset";
        }

        output.valid =
            output.base_file_valid &&
            output.query_file_valid &&
            output.metadata_file_valid &&
            output.metal_available &&
            output.unified_memory &&
            output.library_available &&
            output.kernel_available &&
            output.pipeline_available &&
            output.command_queue_available &&
            output.exact_ground_truth_complete &&
            output.filter_results_valid &&
            output.navigation_semantics_identical &&
            lazy_total.size() ==
                sample_count &&
            cpu_total.size() ==
                sample_count &&
            gpu_total.size() ==
                sample_count &&
            !output.racorn_modified;
    }

    print_progress(
        "[G-DATA3A] Benchmark completado."
    );

    return output;
}

} // namespace racorn_uma::platform::apple

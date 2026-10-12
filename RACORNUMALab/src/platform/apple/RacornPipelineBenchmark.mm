//
//  RacornPipelineBenchmark.mm
//  RACORNUMALab
//
//  Propósito:
//  Implementar G-UMA5C.
//
//  Se comparan:
//
//      1. CPU-Lazy.
//      2. CPU-Bitset Serial.
//      3. CPU-Bitset Pipeline.
//      4. GPU-Bitset Pipeline.
//
//  El pipeline procesa lotes de 16 consultas.
//
//  CPU Pipeline:
//  Un worker persistente prepara q(i+1) sobre un segundo bitset mientras
//  el thread principal ejecuta RACORN(q(i)).
//
//  GPU Pipeline:
//  Metal prepara q(i+1) de forma asíncrona mientras CPU ejecuta
//  RACORN(q(i)).
//
//  En ambos casos:
//
//      Buffer A -> RACORN lee.
//      Buffer B -> productor escribe.
//
//  Después intercambian roles.
//
//  La primera preparación del lote está incluida en el tiempo.
//  También están incluidas las esperas residuales necesarias antes de
//  consumir el siguiente bitset.
//
//  Esto mide throughput real del pipeline, no únicamente kernel time.
//
//  Configuración:
//
//      N ∈ {20k,100k,250k}
//      D = 32
//      M = 16
//      efConstruction = 100
//      efSearch = 200
//      BR = 0.50
//      k = 10
//      Q = 16
//
//  Benchmark:
//
//      warmup batches = 1
//      measured batches = 10
//
//  Restricción:
//
//      RACORN permanece completamente intacto.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Sin persistencia.
//

#include "platform/apple/RacornPipelineBenchmark.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace racorn_uma::platform::apple {

namespace {

using namespace unum::usearch;

using Clock =
    std::chrono::steady_clock;

constexpr char const* kKernelName =
    "racorn_pipeline_predicate_to_bitset";

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

constexpr std::size_t kWarmupBatches =
    1U;

constexpr std::size_t kMeasuredBatches =
    10U;

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

struct BatchMeasurement final {

    bool valid{};

    double batch_ms{};

    double exposed_wait_ms{};

    std::vector<SearchSnapshot> snapshots;

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
            unit * 2.0 -
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

FilterSpec filter_for_query(
    std::size_t query_index
) noexcept {

    return FilterSpec{
        .tenant_required =
            static_cast<std::uint32_t>(
                query_index %
                16U
            )
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
            2U &&

        category_allowed &&

        metadata.security_level[key] <=
            2U;
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

class CpuBitsetWorker final {

public:

    CpuBitsetWorker(
        MetadataView metadata,
        std::size_t element_count
    )
        : metadata_(
              metadata
          ),
          element_count_(
              element_count
          ),
          thread_(
              &CpuBitsetWorker::thread_main,
              this
          ) {
    }

    ~CpuBitsetWorker() {

        {
            std::lock_guard<std::mutex>
                lock(
                    mutex_
                );

            stop_ =
                true;
        }

        cv_job_.notify_one();

        if (
            thread_.joinable()
        ) {

            thread_.join();
        }
    }

    CpuBitsetWorker(
        CpuBitsetWorker const&
    ) = delete;

    CpuBitsetWorker& operator=(
        CpuBitsetWorker const&
    ) = delete;

    void submit(
        FilterSpec filter,
        std::uint32_t* output
    ) {

        std::unique_lock<std::mutex>
            lock(
                mutex_
            );

        cv_idle_.wait(
            lock,
            [&]() {

                return
                    !job_pending_;
            }
        );

        filter_ =
            filter;

        output_ =
            output;

        job_done_ =
            false;

        job_pending_ =
            true;

        lock.unlock();

        cv_job_.notify_one();
    }

    void wait() {

        std::unique_lock<std::mutex>
            lock(
                mutex_
            );

        cv_done_.wait(
            lock,
            [&]() {

                return
                    job_done_;
            }
        );

        job_done_ =
            false;
    }

private:

    void thread_main() {

        for (;;) {

            FilterSpec filter;
            std::uint32_t* output =
                nullptr;

            {
                std::unique_lock<std::mutex>
                    lock(
                        mutex_
                    );

                cv_job_.wait(
                    lock,
                    [&]() {

                        return
                            stop_ ||
                            job_pending_;
                    }
                );

                if (
                    stop_
                ) {

                    return;
                }

                filter =
                    filter_;

                output =
                    output_;

                job_pending_ =
                    false;

                cv_idle_.notify_one();
            }

            materialize_cpu_bitset(
                metadata_,
                element_count_,
                filter,
                output
            );

            {
                std::lock_guard<std::mutex>
                    lock(
                        mutex_
                    );

                job_done_ =
                    true;
            }

            cv_done_.notify_one();
        }
    }

    MetadataView metadata_{};

    std::size_t element_count_{};

    std::thread thread_;

    std::mutex mutex_;

    std::condition_variable cv_job_;
    std::condition_variable cv_done_;
    std::condition_variable cv_idle_;

    FilterSpec filter_{};

    std::uint32_t* output_{};

    bool stop_{};
    bool job_pending_{};
    bool job_done_{};
};

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
        1024U
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

bool same_counters(
    SearchSnapshot const& first,
    SearchSnapshot const& second
) noexcept {

    return
        first.computed_distances ==
            second.computed_distances &&

        first.visited_members ==
            second.visited_members &&

        first.predicate_evaluations ==
            second.predicate_evaluations &&

        first.fallback_count ==
            second.fallback_count;
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
        lower == upper
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

void fill_plan_stats(
    PipelinePlanStats& output,
    std::vector<double> const& batch_times,
    std::vector<double> const& waits
) {

    output.batch_p50_ms =
        percentile(
            batch_times,
            0.50
        );

    output.batch_p95_ms =
        percentile(
            batch_times,
            0.95
        );

    if (
        output.batch_p50_ms >
        0.0
    ) {

        output.ms_per_query_p50 =
            output.batch_p50_ms /
            static_cast<double>(
                kQueryCount
            );

        output.qps_p50 =
            (
                static_cast<double>(
                    kQueryCount
                ) *
                1000.0
            ) /
            output.batch_p50_ms;
    }

    if (
        !waits.empty()
    ) {

        output.exposed_wait_p50_ms =
            percentile(
                waits,
                0.50
            );

        output.exposed_wait_p95_ms =
            percentile(
                waits,
                0.95
            );
    }
}

id<MTLCommandBuffer>
submit_gpu_prepare(
    id<MTLCommandQueue> command_queue,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> metadata_buffer,
    id<MTLBuffer> bitset_buffer,
    std::size_t column_stride_bytes,
    std::size_t element_count,
    FilterSpec filter,
    std::size_t threads_per_group,
    std::string& error
) {

    id<MTLCommandBuffer> command_buffer =
        [command_queue commandBuffer];

    if (
        command_buffer == nil
    ) {

        error =
            "No se pudo crear MTLCommandBuffer.";

        return nil;
    }

    id<MTLComputeCommandEncoder> encoder =
        [
            command_buffer
            computeCommandEncoder
        ];

    if (
        encoder == nil
    ) {

        error =
            "No se pudo crear MTLComputeCommandEncoder.";

        return nil;
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

    return command_buffer;
}

bool wait_gpu(
    id<MTLCommandBuffer> command_buffer,
    std::string& error
) {

    if (
        command_buffer == nil
    ) {

        return false;
    }

    [
        command_buffer
        waitUntilCompleted
    ];

    if (
        command_buffer.status !=
        MTLCommandBufferStatusCompleted
    ) {

        if (
            command_buffer.error != nil
        ) {

            error =
                string_from_nsstring(
                    command_buffer
                        .error
                        .localizedDescription
                );

        } else {

            error =
                "Command Buffer Metal incompleto.";
        }

        return false;
    }

    return true;
}

template <typename Predicate>
SearchSnapshot run_search_snapshot(
    Index& index,
    float const* query,
    L2Metric const& metric,
    Predicate const& predicate
) {

    auto configuration =
        make_racorn_search_config();

    auto result =
        index.search(
            query,
            kNeighbors,
            metric,
            configuration,
            predicate
        );

    return
        snapshot_search_result(
            result
        );
}

template <typename Predicate>
void run_search_sink(
    Index& index,
    float const* query,
    L2Metric const& metric,
    Predicate const& predicate,
    std::size_t& sink
) {

    auto configuration =
        make_racorn_search_config();

    auto result =
        index.search(
            query,
            kNeighbors,
            metric,
            configuration,
            predicate
        );

    sink ^=
        result.size() +
        result.computed_distances +
        result.visited_members +
        result.fallback_count;
}

BatchMeasurement run_cpu_pipeline_batch(
    Index& index,
    QueryStore const& queries,
    L2Metric const& metric,
    CpuBitsetWorker& worker,
    std::uint32_t* bitset0,
    std::uint32_t* bitset1,
    bool capture,
    std::size_t& sink
) {

    BatchMeasurement measurement;

    if (
        capture
    ) {

        measurement.snapshots.reserve(
            kQueryCount
        );
    }

    std::uint32_t* buffers[2]{
        bitset0,
        bitset1
    };

    const auto batch_start =
        Clock::now();

    worker.submit(
        filter_for_query(
            0
        ),
        buffers[0]
    );

    auto wait_start =
        Clock::now();

    worker.wait();

    auto wait_end =
        Clock::now();

    measurement.exposed_wait_ms +=
        milliseconds_between(
            wait_start,
            wait_end
        );

    for (
        std::size_t query_index = 0;
        query_index < kQueryCount;
        ++query_index
    ) {

        const std::size_t current_buffer =
            query_index &
            1U;

        if (
            query_index +
                1U <
            kQueryCount
        ) {

            const std::size_t next_buffer =
                (
                    query_index +
                    1U
                ) &
                1U;

            worker.submit(
                filter_for_query(
                    query_index +
                    1U
                ),
                buffers[
                    next_buffer
                ]
            );
        }

        const BitsetPredicate predicate{
            .bitset =
                buffers[
                    current_buffer
                ]
        };

        if (
            capture
        ) {

            measurement.snapshots.push_back(
                run_search_snapshot(
                    index,
                    queries.row(
                        query_index
                    ),
                    metric,
                    predicate
                )
            );

        } else {

            run_search_sink(
                index,
                queries.row(
                    query_index
                ),
                metric,
                predicate,
                sink
            );
        }

        if (
            query_index +
                1U <
            kQueryCount
        ) {

            wait_start =
                Clock::now();

            worker.wait();

            wait_end =
                Clock::now();

            measurement.exposed_wait_ms +=
                milliseconds_between(
                    wait_start,
                    wait_end
                );
        }
    }

    const auto batch_end =
        Clock::now();

    measurement.batch_ms =
        milliseconds_between(
            batch_start,
            batch_end
        );

    measurement.valid =
        true;

    return measurement;
}

BatchMeasurement run_gpu_pipeline_batch(
    Index& index,
    QueryStore const& queries,
    L2Metric const& metric,
    id<MTLCommandQueue> command_queue,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> metadata_buffer,
    id<MTLBuffer> bitset_buffer0,
    id<MTLBuffer> bitset_buffer1,
    std::size_t column_stride_bytes,
    std::size_t element_count,
    std::size_t threads_per_group,
    bool capture,
    std::size_t& sink
) {

    BatchMeasurement measurement;

    if (
        capture
    ) {

        measurement.snapshots.reserve(
            kQueryCount
        );
    }

    id<MTLBuffer> buffers[2]{
        bitset_buffer0,
        bitset_buffer1
    };

    std::string error;

    const auto batch_start =
        Clock::now();

    id<MTLCommandBuffer> pending =
        submit_gpu_prepare(
            command_queue,
            pipeline,
            metadata_buffer,
            buffers[0],
            column_stride_bytes,
            element_count,
            filter_for_query(
                0
            ),
            threads_per_group,
            error
        );

    if (
        pending == nil
    ) {

        measurement.error =
            error;

        return measurement;
    }

    auto wait_start =
        Clock::now();

    if (
        !wait_gpu(
            pending,
            error
        )
    ) {

        measurement.error =
            error;

        return measurement;
    }

    auto wait_end =
        Clock::now();

    measurement.exposed_wait_ms +=
        milliseconds_between(
            wait_start,
            wait_end
        );

    for (
        std::size_t query_index = 0;
        query_index < kQueryCount;
        ++query_index
    ) {

        const std::size_t current_buffer =
            query_index &
            1U;

        id<MTLCommandBuffer> next_command =
            nil;

        if (
            query_index +
                1U <
            kQueryCount
        ) {

            const std::size_t next_buffer =
                (
                    query_index +
                    1U
                ) &
                1U;

            next_command =
                submit_gpu_prepare(
                    command_queue,
                    pipeline,
                    metadata_buffer,
                    buffers[
                        next_buffer
                    ],
                    column_stride_bytes,
                    element_count,
                    filter_for_query(
                        query_index +
                        1U
                    ),
                    threads_per_group,
                    error
                );

            if (
                next_command == nil
            ) {

                measurement.error =
                    error;

                return measurement;
            }
        }

        const auto* bitset =
            static_cast<std::uint32_t const*>(
                buffers[
                    current_buffer
                ].contents
            );

        const BitsetPredicate predicate{
            .bitset =
                bitset
        };

        if (
            capture
        ) {

            measurement.snapshots.push_back(
                run_search_snapshot(
                    index,
                    queries.row(
                        query_index
                    ),
                    metric,
                    predicate
                )
            );

        } else {

            run_search_sink(
                index,
                queries.row(
                    query_index
                ),
                metric,
                predicate,
                sink
            );
        }

        if (
            next_command != nil
        ) {

            wait_start =
                Clock::now();

            if (
                !wait_gpu(
                    next_command,
                    error
                )
            ) {

                measurement.error =
                    error;

                return measurement;
            }

            wait_end =
                Clock::now();

            measurement.exposed_wait_ms +=
                milliseconds_between(
                    wait_start,
                    wait_end
                );
        }
    }

    const auto batch_end =
        Clock::now();

    measurement.batch_ms =
        milliseconds_between(
            batch_start,
            batch_end
        );

    measurement.valid =
        true;

    return measurement;
}

double run_lazy_batch(
    Index& index,
    QueryStore const& queries,
    L2Metric const& metric,
    MetadataView metadata,
    std::size_t& sink
) {

    const auto start =
        Clock::now();

    for (
        std::size_t query_index = 0;
        query_index < kQueryCount;
        ++query_index
    ) {

        const LazyPredicate predicate{
            .metadata =
                metadata,

            .filter =
                filter_for_query(
                    query_index
                )
        };

        run_search_sink(
            index,
            queries.row(
                query_index
            ),
            metric,
            predicate,
            sink
        );
    }

    const auto end =
        Clock::now();

    return
        milliseconds_between(
            start,
            end
        );
}

double run_cpu_serial_batch(
    Index& index,
    QueryStore const& queries,
    L2Metric const& metric,
    MetadataView metadata,
    std::size_t vector_count,
    std::uint32_t* bitset,
    std::size_t& sink
) {

    const auto start =
        Clock::now();

    for (
        std::size_t query_index = 0;
        query_index < kQueryCount;
        ++query_index
    ) {

        materialize_cpu_bitset(
            metadata,
            vector_count,
            filter_for_query(
                query_index
            ),
            bitset
        );

        const BitsetPredicate predicate{
            .bitset =
                bitset
        };

        run_search_sink(
            index,
            queries.row(
                query_index
            ),
            metric,
            predicate,
            sink
        );
    }

    const auto end =
        Clock::now();

    return
        milliseconds_between(
            start,
            end
        );
}

void print_progress(
    std::string const& text
) {

    std::cout
        << text
        << '\n'
        << std::flush;
}

} // namespace

RacornPipelineBenchmarkResult
RacornPipelineBenchmark::run() {

    RacornPipelineBenchmarkResult output;

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

    output.warmup_batches =
        kWarmupBatches;

    output.measured_batches =
        kMeasuredBatches;

    output.racorn_modified =
        false;

    print_progress(
        "[G-UMA5C] Inicializando Metal..."
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

        NSString* function_name =
            [NSString
                stringWithUTF8String:
                    kKernelName];

        id<MTLFunction> function =
            [
                library
                newFunctionWithName:
                    function_name
            ];

        if (
            function == nil
        ) {

            output.metal_error =
                "No se encontro racorn_pipeline_predicate_to_bitset.";

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

            output.metal_error =
                pipeline_error != nil
                    ? string_from_nsstring(
                          pipeline_error
                              .localizedDescription
                      )
                    : "No se pudo crear el pipeline Metal.";

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
                "Threadgroup invalido.";

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
            "[G-UMA5C] Inicializando metadata Shared..."
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

        id<MTLBuffer> cpu_serial_buffer =
            [
                device
                newBufferWithLength:
                    max_bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        id<MTLBuffer> cpu_pipeline_buffer0 =
            [
                device
                newBufferWithLength:
                    max_bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        id<MTLBuffer> cpu_pipeline_buffer1 =
            [
                device
                newBufferWithLength:
                    max_bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        id<MTLBuffer> gpu_pipeline_buffer0 =
            [
                device
                newBufferWithLength:
                    max_bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        id<MTLBuffer> gpu_pipeline_buffer1 =
            [
                device
                newBufferWithLength:
                    max_bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        if (
            cpu_serial_buffer == nil ||
            cpu_pipeline_buffer0 == nil ||
            cpu_pipeline_buffer1 == nil ||
            gpu_pipeline_buffer0 == nil ||
            gpu_pipeline_buffer1 == nil
        ) {

            output.metal_error =
                "No se pudieron crear los bitsets Shared.";

            return output;
        }

        auto* cpu_serial_bitset =
            static_cast<std::uint32_t*>(
                cpu_serial_buffer.contents
            );

        auto* cpu_pipeline_bitset0 =
            static_cast<std::uint32_t*>(
                cpu_pipeline_buffer0.contents
            );

        auto* cpu_pipeline_bitset1 =
            static_cast<std::uint32_t*>(
                cpu_pipeline_buffer1.contents
            );

        bool all_cases_identical =
            true;

        for (
            std::size_t vector_count :
            kVectorCounts
        ) {

            std::cout
                << "\n[G-UMA5C] N="
                << vector_count
                << " - construyendo indice...\n"
                << std::flush;

            RacornPipelineCase
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
                        << "[G-UMA5C] N="
                        << vector_count
                        << " HNSW "
                        << percent
                        << "%\n"
                        << std::flush;
                }
            }

            print_progress(
                "[G-UMA5C] Construyendo referencia semantica..."
            );

            std::vector<SearchSnapshot>
                reference_snapshots;

            reference_snapshots.reserve(
                kQueryCount
            );

            std::size_t total_selected =
                0;

            double recall_sum =
                0.0;

            double distances_sum =
                0.0;

            double visited_sum =
                0.0;

            double predicate_sum =
                0.0;

            double fallback_sum =
                0.0;

            bool exact_complete =
                true;

            bool filters_valid =
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

                const LazyPredicate predicate{
                    .metadata =
                        metadata_view,

                    .filter =
                        filter
                };

                std::size_t selected =
                    0;

                for (
                    std::size_t key = 0;
                    key < vector_count;
                    ++key
                ) {

                    if (
                        predicate.accepts_key(
                            key
                        )
                    ) {

                        ++selected;
                    }
                }

                total_selected +=
                    selected;

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

                const auto exact =
                    exact_filtered_top_k(
                        base,
                        queries.row(
                            query_index
                        ),
                        metadata_view,
                        filter,
                        vector_count,
                        kNeighbors
                    );

                if (
                    exact.size() !=
                    kNeighbors
                ) {

                    exact_complete =
                        false;
                }

                const SearchSnapshot snapshot =
                    run_search_snapshot(
                        index,
                        queries.row(
                            query_index
                        ),
                        metric,
                        predicate
                    );

                reference_snapshots.push_back(
                    snapshot
                );

                recall_sum +=
                    recall_at_k(
                        snapshot.keys,
                        exact
                    );

                distances_sum +=
                    static_cast<double>(
                        snapshot.computed_distances
                    );

                visited_sum +=
                    static_cast<double>(
                        snapshot.visited_members
                    );

                predicate_sum +=
                    static_cast<double>(
                        snapshot.predicate_evaluations
                    );

                fallback_sum +=
                    static_cast<double>(
                        snapshot.fallback_count
                    );

                for (
                    std::size_t key :
                    snapshot.keys
                ) {

                    filters_valid =
                        filters_valid &&
                        predicate.accepts_key(
                            key
                        );
                }
            }

            case_result.mean_selectivity =
                static_cast<double>(
                    total_selected
                ) /
                (
                    static_cast<double>(
                        kQueryCount
                    ) *
                    static_cast<double>(
                        vector_count
                    )
                );

            case_result.mean_recall_at_k =
                recall_sum /
                static_cast<double>(
                    kQueryCount
                );

            case_result.mean_computed_distances =
                distances_sum /
                static_cast<double>(
                    kQueryCount
                );

            case_result.mean_visited_members =
                visited_sum /
                static_cast<double>(
                    kQueryCount
                );

            case_result.mean_predicate_evaluations =
                predicate_sum /
                static_cast<double>(
                    kQueryCount
                );

            case_result.mean_fallback_count =
                fallback_sum /
                static_cast<double>(
                    kQueryCount
                );

            case_result.exact_ground_truth_complete =
                exact_complete;

            case_result.filter_results_valid =
                filters_valid;

            //
            // Verificación CPU/GPU bitset representación.
            //

            const std::size_t word_count =
                bitset_word_count_for(
                    vector_count
                );

            for (
                std::size_t query_index = 0;
                query_index < kQueryCount;
                ++query_index
            ) {

                const FilterSpec filter =
                    filter_for_query(
                        query_index
                    );

                materialize_cpu_bitset(
                    metadata_view,
                    vector_count,
                    filter,
                    cpu_serial_bitset
                );

                std::string error;

                id<MTLCommandBuffer> command =
                    submit_gpu_prepare(
                        command_queue,
                        pipeline,
                        metadata_buffer,
                        gpu_pipeline_buffer0,
                        column_stride_bytes,
                        vector_count,
                        filter,
                        output
                            .threads_per_threadgroup,
                        error
                    );

                if (
                    command == nil ||
                    !wait_gpu(
                        command,
                        error
                    )
                ) {

                    output.metal_error =
                        error;

                    return output;
                }

                auto const* gpu_bits =
                    static_cast<
                        std::uint32_t const*
                    >(
                        gpu_pipeline_buffer0.contents
                    );

                case_result.bitset_word_mismatches +=
                    mismatch_word_count(
                        cpu_serial_bitset,
                        gpu_bits,
                        word_count
                    );

                const LazyPredicate direct{
                    .metadata =
                        metadata_view,

                    .filter =
                        filter
                };

                const BitsetPredicate cpu_bits{
                    .bitset =
                        cpu_serial_bitset
                };

                const BitsetPredicate gpu_predicate{
                    .bitset =
                        gpu_bits
                };

                for (
                    std::size_t key = 0;
                    key < vector_count;
                    ++key
                ) {

                    const bool expected =
                        direct.accepts_key(
                            key
                        );

                    if (
                        expected !=
                            cpu_bits.accepts_key(
                                key
                            ) ||
                        expected !=
                            gpu_predicate.accepts_key(
                                key
                            )
                    ) {

                        ++case_result
                            .predicate_representation_mismatches;
                    }
                }
            }

            //
            // Validación específica del pipeline CPU.
            //

            CpuBitsetWorker worker(
                metadata_view,
                vector_count
            );

            const BatchMeasurement
                cpu_validation =
                run_cpu_pipeline_batch(
                    index,
                    queries,
                    metric,
                    worker,
                    cpu_pipeline_bitset0,
                    cpu_pipeline_bitset1,
                    true,
                    output.benchmark_sink
                );

            if (
                !cpu_validation.valid
            ) {

                output.metal_error =
                    cpu_validation.error;

                return output;
            }

            for (
                std::size_t query_index = 0;
                query_index < kQueryCount;
                ++query_index
            ) {

                const SearchSnapshot& reference =
                    reference_snapshots[
                        query_index
                    ];

                const SearchSnapshot& observed =
                    cpu_validation.snapshots[
                        query_index
                    ];

                if (
                    reference.keys !=
                    observed.keys
                ) {

                    ++case_result
                        .cpu_pipeline_topk_mismatch_queries;
                }

                if (
                    !same_counters(
                        reference,
                        observed
                    )
                ) {

                    ++case_result
                        .cpu_pipeline_counter_mismatch_queries;
                }
            }

            case_result.cpu_pipeline_semantics_identical =
                case_result
                    .cpu_pipeline_topk_mismatch_queries ==
                    0 &&
                case_result
                    .cpu_pipeline_counter_mismatch_queries ==
                    0;

            //
            // Validación específica del pipeline GPU.
            //

            const BatchMeasurement
                gpu_validation =
                run_gpu_pipeline_batch(
                    index,
                    queries,
                    metric,
                    command_queue,
                    pipeline,
                    metadata_buffer,
                    gpu_pipeline_buffer0,
                    gpu_pipeline_buffer1,
                    column_stride_bytes,
                    vector_count,
                    output
                        .threads_per_threadgroup,
                    true,
                    output.benchmark_sink
                );

            if (
                !gpu_validation.valid
            ) {

                output.metal_error =
                    gpu_validation.error;

                return output;
            }

            for (
                std::size_t query_index = 0;
                query_index < kQueryCount;
                ++query_index
            ) {

                const SearchSnapshot& reference =
                    reference_snapshots[
                        query_index
                    ];

                const SearchSnapshot& observed =
                    gpu_validation.snapshots[
                        query_index
                    ];

                if (
                    reference.keys !=
                    observed.keys
                ) {

                    ++case_result
                        .gpu_pipeline_topk_mismatch_queries;
                }

                if (
                    !same_counters(
                        reference,
                        observed
                    )
                ) {

                    ++case_result
                        .gpu_pipeline_counter_mismatch_queries;
                }
            }

            case_result.gpu_pipeline_semantics_identical =
                case_result
                    .gpu_pipeline_topk_mismatch_queries ==
                    0 &&
                case_result
                    .gpu_pipeline_counter_mismatch_queries ==
                    0;

            if (
                case_result.bitset_word_mismatches !=
                    0 ||
                case_result.predicate_representation_mismatches !=
                    0 ||
                !case_result.cpu_pipeline_semantics_identical ||
                !case_result.gpu_pipeline_semantics_identical ||
                !case_result.exact_ground_truth_complete ||
                !case_result.filter_results_valid
            ) {

                case_result.valid =
                    false;

                all_cases_identical =
                    false;

                output.cases.push_back(
                    case_result
                );

                continue;
            }

            print_progress(
                "[G-UMA5C] Semantica PASS. Warm-up..."
            );

            for (
                std::size_t warmup = 0;
                warmup < kWarmupBatches;
                ++warmup
            ) {

                run_lazy_batch(
                    index,
                    queries,
                    metric,
                    metadata_view,
                    output.benchmark_sink
                );

                run_cpu_serial_batch(
                    index,
                    queries,
                    metric,
                    metadata_view,
                    vector_count,
                    cpu_serial_bitset,
                    output.benchmark_sink
                );

                run_cpu_pipeline_batch(
                    index,
                    queries,
                    metric,
                    worker,
                    cpu_pipeline_bitset0,
                    cpu_pipeline_bitset1,
                    false,
                    output.benchmark_sink
                );

                const BatchMeasurement gpu_warmup =
                    run_gpu_pipeline_batch(
                        index,
                        queries,
                        metric,
                        command_queue,
                        pipeline,
                        metadata_buffer,
                        gpu_pipeline_buffer0,
                        gpu_pipeline_buffer1,
                        column_stride_bytes,
                        vector_count,
                        output
                            .threads_per_threadgroup,
                        false,
                        output.benchmark_sink
                    );

                if (
                    !gpu_warmup.valid
                ) {

                    output.metal_error =
                        gpu_warmup.error;

                    return output;
                }
            }

            print_progress(
                "[G-UMA5C] Midiendo throughput por lotes..."
            );

            std::vector<double>
                lazy_times;

            std::vector<double>
                serial_times;

            std::vector<double>
                cpu_pipeline_times;

            std::vector<double>
                cpu_pipeline_waits;

            std::vector<double>
                gpu_pipeline_times;

            std::vector<double>
                gpu_pipeline_waits;

            lazy_times.reserve(
                kMeasuredBatches
            );

            serial_times.reserve(
                kMeasuredBatches
            );

            cpu_pipeline_times.reserve(
                kMeasuredBatches
            );

            cpu_pipeline_waits.reserve(
                kMeasuredBatches
            );

            gpu_pipeline_times.reserve(
                kMeasuredBatches
            );

            gpu_pipeline_waits.reserve(
                kMeasuredBatches
            );

            for (
                std::size_t batch = 0;
                batch < kMeasuredBatches;
                ++batch
            ) {

                std::cout
                    << "[G-UMA5C] N="
                    << vector_count
                    << " batch "
                    << (
                        batch +
                        1U
                    )
                    << "/"
                    << kMeasuredBatches
                    << '\n'
                    << std::flush;

                auto measure_lazy =
                    [&]() {

                        lazy_times.push_back(
                            run_lazy_batch(
                                index,
                                queries,
                                metric,
                                metadata_view,
                                output.benchmark_sink
                            )
                        );
                    };

                auto measure_serial =
                    [&]() {

                        serial_times.push_back(
                            run_cpu_serial_batch(
                                index,
                                queries,
                                metric,
                                metadata_view,
                                vector_count,
                                cpu_serial_bitset,
                                output.benchmark_sink
                            )
                        );
                    };

                auto measure_cpu_pipeline =
                    [&]() {

                        const BatchMeasurement measurement =
                            run_cpu_pipeline_batch(
                                index,
                                queries,
                                metric,
                                worker,
                                cpu_pipeline_bitset0,
                                cpu_pipeline_bitset1,
                                false,
                                output.benchmark_sink
                            );

                        cpu_pipeline_times.push_back(
                            measurement.batch_ms
                        );

                        cpu_pipeline_waits.push_back(
                            measurement.exposed_wait_ms
                        );
                    };

                auto measure_gpu_pipeline =
                    [&]() -> bool {

                        const BatchMeasurement measurement =
                            run_gpu_pipeline_batch(
                                index,
                                queries,
                                metric,
                                command_queue,
                                pipeline,
                                metadata_buffer,
                                gpu_pipeline_buffer0,
                                gpu_pipeline_buffer1,
                                column_stride_bytes,
                                vector_count,
                                output
                                    .threads_per_threadgroup,
                                false,
                                output.benchmark_sink
                            );

                        if (
                            !measurement.valid
                        ) {

                            output.metal_error =
                                measurement.error;

                            return false;
                        }

                        gpu_pipeline_times.push_back(
                            measurement.batch_ms
                        );

                        gpu_pipeline_waits.push_back(
                            measurement.exposed_wait_ms
                        );

                        return true;
                    };

                const std::size_t order =
                    batch %
                    4U;

                if (
                    order == 0U
                ) {

                    measure_lazy();
                    measure_serial();
                    measure_cpu_pipeline();

                    if (
                        !measure_gpu_pipeline()
                    ) {

                        return output;
                    }

                } else if (
                    order == 1U
                ) {

                    measure_serial();
                    measure_cpu_pipeline();

                    if (
                        !measure_gpu_pipeline()
                    ) {

                        return output;
                    }

                    measure_lazy();

                } else if (
                    order == 2U
                ) {

                    measure_cpu_pipeline();

                    if (
                        !measure_gpu_pipeline()
                    ) {

                        return output;
                    }

                    measure_lazy();
                    measure_serial();

                } else {

                    if (
                        !measure_gpu_pipeline()
                    ) {

                        return output;
                    }

                    measure_lazy();
                    measure_serial();
                    measure_cpu_pipeline();
                }
            }

            fill_plan_stats(
                case_result.cpu_lazy,
                lazy_times,
                {}
            );

            fill_plan_stats(
                case_result.cpu_bitset_serial,
                serial_times,
                {}
            );

            fill_plan_stats(
                case_result.cpu_bitset_pipeline,
                cpu_pipeline_times,
                cpu_pipeline_waits
            );

            fill_plan_stats(
                case_result.gpu_bitset_pipeline,
                gpu_pipeline_times,
                gpu_pipeline_waits
            );

            if (
                case_result
                    .cpu_bitset_pipeline
                    .batch_p50_ms >
                0.0
            ) {

                case_result
                    .cpu_pipeline_speedup_vs_serial =
                    case_result
                        .cpu_bitset_serial
                        .batch_p50_ms /
                    case_result
                        .cpu_bitset_pipeline
                        .batch_p50_ms;
            }

            if (
                case_result
                    .gpu_bitset_pipeline
                    .batch_p50_ms >
                0.0
            ) {

                case_result
                    .gpu_pipeline_speedup_vs_serial =
                    case_result
                        .cpu_bitset_serial
                        .batch_p50_ms /
                    case_result
                        .gpu_bitset_pipeline
                        .batch_p50_ms;

                case_result
                    .gpu_pipeline_speedup_vs_lazy =
                    case_result
                        .cpu_lazy
                        .batch_p50_ms /
                    case_result
                        .gpu_bitset_pipeline
                        .batch_p50_ms;
            }

            const std::array<
                std::pair<
                    double,
                    char const*
                >,
                4
            > plans{
                std::pair{
                    case_result
                        .cpu_lazy
                        .batch_p50_ms,
                    "CPU-Lazy"
                },

                std::pair{
                    case_result
                        .cpu_bitset_serial
                        .batch_p50_ms,
                    "CPU-Bitset-Serial"
                },

                std::pair{
                    case_result
                        .cpu_bitset_pipeline
                        .batch_p50_ms,
                    "CPU-Bitset-Pipeline"
                },

                std::pair{
                    case_result
                        .gpu_bitset_pipeline
                        .batch_p50_ms,
                    "GPU-Bitset-Pipeline"
                }
            };

            auto winner =
                std::min_element(
                    plans.begin(),
                    plans.end(),
                    [](
                        auto const& first,
                        auto const& second
                    ) {

                        return
                            first.first <
                            second.first;
                    }
                );

            case_result.throughput_winner =
                winner->second;

            case_result.valid =
                case_result
                    .cpu_pipeline_semantics_identical &&
                case_result
                    .gpu_pipeline_semantics_identical &&
                case_result
                    .bitset_word_mismatches ==
                    0 &&
                case_result
                    .predicate_representation_mismatches ==
                    0 &&
                case_result
                    .exact_ground_truth_complete &&
                case_result
                    .filter_results_valid;

            all_cases_identical =
                all_cases_identical &&
                case_result.valid;

            output.cases.push_back(
                case_result
            );

            std::cout
                << "[G-UMA5C] N="
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
        "[G-UMA5C] Benchmark finalizado."
    );

    return output;
}

} // namespace racorn_uma::platform::apple

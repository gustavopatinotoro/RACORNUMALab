//
//  SiftUmaPipelineBenchmark.mm
//  RACORNUMALab
//
//  Gate:
//      G-DATA3B — Pipeline CPU/GPU sobre SIFT100K.
//
//  Objetivo:
//  Determinar si el coste de lanzamiento/sincronización Metal observado
//  en G-DATA3A puede ocultarse mediante solapamiento:
//
//      preparar q[i+1] || navegar q[i]
//
//  Dataset:
//      SIFT1M train[0:100000]
//
//  Configuración:
//      N   = 100000
//      D   = 128
//      Q   = 256
//      M   = 16
//      efC = 100
//      efS = 200
//      k   = 10
//      BR  = 1.0
//      TT  = 0.30
//
//  RACORN-1+:
//      desactivado.
//
//  CPU-Pipeline:
//      - worker persistente;
//      - un hilo adicional;
//      - double buffering Shared.
//
//  GPU-Pipeline:
//      - command buffers asíncronos;
//      - double buffering Shared;
//      - navegación exclusivamente CPU.
//
//  Benchmark:
//      1 warm-up batch.
//      5 measured batches.
//      256 queries por batch.
//      orden de planes rotado.
//
//  Criterio de continuación Metal, fijado antes de observar resultados:
//
//      GPU/CPU p50 >= 1.02
//
//      Y
//
//      GPU p95 <= CPU p95 * 1.02
//
//  Si no se cumplen:
//      STOP_METAL_PREDICATE.
//
//  Importante:
//  G-DATA3A ya validó equivalencia sobre 1000 queries.
//  G-DATA3B vuelve a validar las 256 queries utilizadas en el pipeline.
//
//  Invariantes:
//      - RACORN upstream intacto.
//      - GPU nunca navega.
//      - Sin heaps GPU.
//      - Sin visited GPU.
//      - Sin bridges GPU.
//      - Sin exact fallback.
//      - CPU/GPU comparten memoria física UMA.
//

#include "platform/apple/SiftUmaPipelineBenchmark.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
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

constexpr std::size_t kQueryCount =
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

constexpr std::size_t kBitsPerWord =
    32U;

constexpr std::size_t kWarmupBatches =
    1U;

constexpr std::size_t kMeasuredBatches =
    5U;

constexpr double kTwoHopThreshold =
    0.30;

constexpr double kBridgeBudgetRatio =
    1.00;

constexpr double kRequiredGpuSpeedup =
    1.02;

constexpr double kAllowedGpuP95Ratio =
    1.02;

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

struct BatchSample final {

    double total_ms{};

    //
    // Tiempo durante el cual el hilo de navegación tuvo que esperar
    // explícitamente a que el productor terminara.
    //

    double exposed_wait_ms{};

    //
    // Trabajo total del productor:
    //
    // CPU: tiempo ejecutando materialize_cpu_bitset().
    // GPU: suma de GPUStartTime -> GPUEndTime.
    //

    double prepare_work_ms{};
};

struct CpuCompletion final {

    double exposed_wait_ms{};
    double work_ms{};
};

struct GpuCompletion final {

    bool completed{};
    bool device_timing_available{};

    double exposed_wait_ms{};
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

        return
            (
                (
                    bitset[
                        key >>
                        5U
                    ] >>
                    (
                        key &
                        31U
                    )
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

void ensure_little_endian() {

    if constexpr (
        std::endian::native !=
        std::endian::little
    ) {

        throw std::runtime_error(
            "G-DATA3B requiere little-endian."
        );
    }
}

std::string string_from_nsstring(
    NSString* value
) {

    if (
        value ==
        nil
    ) {

        return {};
    }

    char const* utf8 =
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

    return
        std::string(
            utf8
        );
}

double milliseconds_between(
    Clock::time_point first,
    Clock::time_point second
) noexcept {

    return
        std::chrono::duration<
            double,
            std::milli
        >(
            second -
            first
        ).count();
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
            "Tamaño inesperado: " +
            path.filename().string()
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
            "Lectura binaria incompleta."
        );
    }

    return output;
}

void read_binary_into(
    std::filesystem::path const& path,
    void* destination,
    std::size_t bytes
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
            "Lectura binaria incompleta."
        );
    }
}

std::size_t bitset_word_count_for(
    std::size_t count
) noexcept {

    return
        (
            count +
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
    std::size_t count,
    FilterSpec const& filter,
    std::uint32_t* output
) noexcept {

    const std::size_t word_count =
        bitset_word_count_for(
            count
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
            std::size_t bit = 0;
            bit < kBitsPerWord;
            ++bit
        ) {

            const std::size_t key =
                base +
                bit;

            if (
                key >=
                count
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
                        bit
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
        std::size_t count
    )
        : metadata_(
              metadata
          ),
          count_(
              count
          ),
          worker_(
              [this]() {
                  worker_loop();
              }
          ) {
    }

    ~CpuBitsetWorker() {

        {
            std::lock_guard<std::mutex>
                lock(
                    mutex_
                );

            stopping_ =
                true;
        }

        job_cv_.notify_one();

        if (
            worker_.joinable()
        ) {

            worker_.join();
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

        std::lock_guard<std::mutex>
            lock(
                mutex_
            );

        if (
            busy_
        ) {

            throw std::runtime_error(
                "CpuBitsetWorker recibió dos trabajos simultáneos."
            );
        }

        filter_ =
            filter;

        output_ =
            output;

        busy_ =
            true;

        has_job_ =
            true;

        completed_ =
            false;

        job_cv_.notify_one();
    }

    [[nodiscard]]
    CpuCompletion wait() {

        const auto wait_start =
            Clock::now();

        std::unique_lock<std::mutex>
            lock(
                mutex_
            );

        done_cv_.wait(
            lock,
            [this]() {
                return completed_;
            }
        );

        const auto wait_end =
            Clock::now();

        CpuCompletion result{
            .exposed_wait_ms =
                milliseconds_between(
                    wait_start,
                    wait_end
                ),

            .work_ms =
                last_work_ms_
        };

        completed_ =
            false;

        busy_ =
            false;

        return result;
    }

private:

    void worker_loop() {

        while (
            true
        ) {

            FilterSpec filter;
            std::uint32_t* output =
                nullptr;

            {
                std::unique_lock<std::mutex>
                    lock(
                        mutex_
                    );

                job_cv_.wait(
                    lock,
                    [this]() {
                        return
                            stopping_ ||
                            has_job_;
                    }
                );

                if (
                    stopping_ &&
                    !has_job_
                ) {

                    return;
                }

                filter =
                    filter_;

                output =
                    output_;

                has_job_ =
                    false;
            }

            const auto work_start =
                Clock::now();

            materialize_cpu_bitset(
                metadata_,
                count_,
                filter,
                output
            );

            const auto work_end =
                Clock::now();

            {
                std::lock_guard<std::mutex>
                    lock(
                        mutex_
                    );

                last_work_ms_ =
                    milliseconds_between(
                        work_start,
                        work_end
                    );

                completed_ =
                    true;
            }

            done_cv_.notify_one();
        }
    }

    MetadataView metadata_{};

    std::size_t count_{};

    std::thread worker_;

    std::mutex mutex_;

    std::condition_variable
        job_cv_;

    std::condition_variable
        done_cv_;

    bool stopping_{};
    bool busy_{};
    bool has_job_{};
    bool completed_{};

    FilterSpec filter_{};

    std::uint32_t* output_{};

    double last_work_ms_{};
};

float l2_squared(
    float const* first,
    float const* second
) noexcept {

    float total =
        0.0F;

    for (
        std::size_t dimension = 0;
        dimension < kDimensions;
        ++dimension
    ) {

        const float difference =
            first[dimension] -
            second[dimension];

        total +=
            difference *
            difference;
    }

    return total;
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
                        )
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

SearchSnapshot snapshot(
    Index::search_result_t const& result
) {

    SearchSnapshot output;

    output.keys.reserve(
        result.size()
    );

    for (
        std::size_t index = 0;
        index < result.size();
        ++index
    ) {

        output.keys.push_back(
            static_cast<std::size_t>(
                result[index]
                    .member
                    .key
            )
        );
    }

    output.computed_distances =
        result.computed_distances;

    output.visited_members =
        result.visited_members;

    output.predicate_evaluations =
        result.predicate_evaluations;

    output.fallback_count =
        result.fallback_count;

    return output;
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

std::size_t choose_threads(
    std::size_t width,
    std::size_t maximum
) noexcept {

    if (
        maximum ==
        0U
    ) {

        return 0U;
    }

    std::size_t selected =
        std::min<std::size_t>(
            256U,
            maximum
        );

    if (
        width >
        0U
    ) {

        selected =
            (
                selected /
                width
            ) *
            width;

        if (
            selected ==
            0U
        ) {

            selected =
                std::min(
                    width,
                    maximum
                );
        }
    }

    return selected;
}

id<MTLCommandBuffer>
submit_gpu_prepare(
    id<MTLCommandQueue> command_queue,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> metadata_buffer,
    id<MTLBuffer> bitset_buffer,
    std::size_t column_stride_bytes,
    FilterSpec const& filter,
    std::size_t threads_per_group,
    std::string& error
) {

    id<MTLCommandBuffer> command_buffer =
        [
            command_queue
            commandBuffer
        ];

    if (
        command_buffer ==
        nil
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
        encoder ==
        nil
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
        NSUInteger column = 0U;
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
            kVectorCount
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
                kVectorCount
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

    return command_buffer;
}

GpuCompletion wait_gpu(
    id<MTLCommandBuffer> command_buffer
) {

    GpuCompletion output;

    if (
        command_buffer ==
        nil
    ) {

        output.error =
            "CommandBuffer nulo.";

        return output;
    }

    const auto wait_start =
        Clock::now();

    [
        command_buffer
        waitUntilCompleted
    ];

    const auto wait_end =
        Clock::now();

    output.exposed_wait_ms =
        milliseconds_between(
            wait_start,
            wait_end
        );

    if (
        command_buffer.status !=
        MTLCommandBufferStatusCompleted
    ) {

        if (
            command_buffer.error !=
            nil
        ) {

            output.error =
                string_from_nsstring(
                    command_buffer
                        .error
                        .localizedDescription
                );

        } else {

            output.error =
                "CommandBuffer GPU incompleto.";
        }

        return output;
    }

    output.completed =
        true;

    const double start =
        command_buffer.GPUStartTime;

    const double end =
        command_buffer.GPUEndTime;

    if (
        std::isfinite(
            start
        ) &&
        std::isfinite(
            end
        ) &&
        start >
            0.0 &&
        end >=
            start
    ) {

        output.device_timing_available =
            true;

        output.device_ms =
            (
                end -
                start
            ) *
            1000.0;
    }

    return output;
}

template <typename Predicate>
SearchSnapshot search_snapshot(
    Index& index,
    float const* query,
    L2Metric const& metric,
    Predicate const& predicate
) {

    auto configuration =
        make_search_configuration();

    auto result =
        index.search(
            query,
            kNeighbors,
            metric,
            configuration,
            predicate
        );

    return
        snapshot(
            result
        );
}

template <typename Predicate>
void search_sink(
    Index& index,
    float const* query,
    L2Metric const& metric,
    Predicate const& predicate,
    std::size_t& sink
) {

    auto configuration =
        make_search_configuration();

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
        result.predicate_evaluations +
        result.fallback_count;
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

void fill_stats(
    PipelineLatencyStats& output,
    std::vector<BatchSample> const& samples
) {

    std::vector<double> total;
    std::vector<double> waits;
    std::vector<double> work;

    total.reserve(
        samples.size()
    );

    waits.reserve(
        samples.size()
    );

    work.reserve(
        samples.size()
    );

    for (
        BatchSample const& sample :
        samples
    ) {

        total.push_back(
            sample.total_ms
        );

        waits.push_back(
            sample.exposed_wait_ms
        );

        work.push_back(
            sample.prepare_work_ms
        );
    }

    output.batch_p50_ms =
        percentile(
            total,
            0.50
        );

    output.batch_p95_ms =
        percentile(
            total,
            0.95
        );

    output.batch_p99_ms =
        percentile(
            total,
            0.99
        );

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

    output.prepare_work_p50_ms =
        percentile(
            work,
            0.50
        );

    output.prepare_work_p95_ms =
        percentile(
            work,
            0.95
        );

    if (
        output.batch_p50_ms >
        0.0
    ) {

        output.qps_p50 =
            static_cast<double>(
                kQueryCount
            ) *
            1000.0 /
            output.batch_p50_ms;
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

SiftUmaPipelineBenchmarkResult
SiftUmaPipelineBenchmark::run(
    std::filesystem::path const& data_directory
) {

    ensure_little_endian();

    SiftUmaPipelineBenchmarkResult output;

    output.data_directory =
        data_directory;

    output.vector_count =
        kVectorCount;

    output.dimensions =
        kDimensions;

    output.query_count =
        kQueryCount;

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

    output.warmup_batches =
        kWarmupBatches;

    output.measured_batches =
        kMeasuredBatches;

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
        "[G-DATA3B] Validando archivos..."
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
        "[G-DATA3B] Cargando SIFT100K..."
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
        "[G-DATA3B] Cargando 256 queries..."
    );

    DenseVectors queries(
        read_binary_exact<float>(
            query_path,
            kQueryCount *
                kDimensions
        ),
        kDimensions
    );

    @autoreleasepool {

        print_progress(
            "[G-DATA3B] Inicializando Metal..."
        );

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

        NSString* function_name =
            [
                NSString
                stringWithUTF8String:
                    kKernelName
            ];

        id<MTLFunction> function =
            [
                library
                newFunctionWithName:
                    function_name
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

            output.metal_error =
                pipeline_error != nil
                    ? string_from_nsstring(
                          pipeline_error
                              .localizedDescription
                      )
                    : "No se pudo crear pipeline Metal.";

            return output;
        }

        output.pipeline_available =
            true;

        output.thread_execution_width =
            static_cast<std::size_t>(
                pipeline.threadExecutionWidth
            );

        output.threads_per_threadgroup =
            choose_threads(
                output.thread_execution_width,
                static_cast<std::size_t>(
                    pipeline
                        .maxTotalThreadsPerThreadgroup
                )
            );

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
                "No se pudo crear command queue.";

            return output;
        }

        output.command_queue_available =
            true;

        //
        // Metadata Shared.
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

        //
        // Dos buffers CPU.
        //

        id<MTLBuffer> cpu_buffer_0 =
            [
                device
                newBufferWithLength:
                    bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        id<MTLBuffer> cpu_buffer_1 =
            [
                device
                newBufferWithLength:
                    bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        //
        // Dos buffers GPU.
        //

        id<MTLBuffer> gpu_buffer_0 =
            [
                device
                newBufferWithLength:
                    bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        id<MTLBuffer> gpu_buffer_1 =
            [
                device
                newBufferWithLength:
                    bitset_bytes
                options:
                    MTLResourceStorageModeShared
            ];

        if (
            cpu_buffer_0 ==
                nil ||
            cpu_buffer_1 ==
                nil ||
            gpu_buffer_0 ==
                nil ||
            gpu_buffer_1 ==
                nil
        ) {

            output.metal_error =
                "No se pudieron crear los cuatro bitsets Shared.";

            return output;
        }

        auto* cpu_bits_0 =
            static_cast<std::uint32_t*>(
                cpu_buffer_0.contents
            );

        auto* cpu_bits_1 =
            static_cast<std::uint32_t*>(
                cpu_buffer_1.contents
            );

        auto* gpu_bits_0 =
            static_cast<std::uint32_t*>(
                gpu_buffer_0.contents
            );

        auto* gpu_bits_1 =
            static_cast<std::uint32_t*>(
                gpu_buffer_1.contents
            );

        auto cpu_bits =
            [&](
                std::size_t slot
            ) -> std::uint32_t* {

                return
                    slot ==
                            0U
                        ? cpu_bits_0
                        : cpu_bits_1;
            };

        auto gpu_bits =
            [&](
                std::size_t slot
            ) -> std::uint32_t* {

                return
                    slot ==
                            0U
                        ? gpu_bits_0
                        : gpu_bits_1;
            };

        auto gpu_buffer =
            [&](
                std::size_t slot
            ) -> id<MTLBuffer> {

                return
                    slot ==
                            0U
                        ? gpu_buffer_0
                        : gpu_buffer_1;
            };

        //
        // Precomputar candidatos filtrados.
        //

        print_progress(
            "[G-DATA3B] Precomputando candidatos filtrados..."
        );

        std::array<
            std::vector<std::size_t>,
            kTenantCount
        >
            eligible_by_tenant;

        for (
            std::size_t tenant = 0;
            tenant < kTenantCount;
            ++tenant
        ) {

            const LazyPredicate predicate{
                .metadata =
                    metadata,

                .filter =
                    FilterSpec{
                        .tenant_required =
                            static_cast<std::uint32_t>(
                                tenant
                            ),

                        .language_min =
                            2U
                    }
            };

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

                    eligible_by_tenant[
                        tenant
                    ].push_back(
                        key
                    );
                }
            }
        }

        //
        // Construir índice.
        //

        L2Metric metric{
            .vectors =
                &base,

            .dimensions =
                kDimensions
        };

        print_progress(
            "[G-DATA3B] Construyendo HNSW..."
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
                    << "[G-DATA3B] HNSW "
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
        // Ground truth exacto para las 256 queries del batch.
        //

        print_progress(
            "[G-DATA3B] Construyendo ground truth..."
        );

        std::vector<
            std::vector<std::size_t>
        >
            ground_truth;

        ground_truth.reserve(
            kQueryCount
        );

        output.ground_truth_complete =
            true;

        for (
            std::size_t query_index = 0;
            query_index < kQueryCount;
            ++query_index
        ) {

            auto exact =
                exact_filtered_top_k(
                    base,
                    queries.row(
                        query_index
                    ),
                    eligible_by_tenant[
                        query_index %
                        kTenantCount
                    ]
                );

            if (
                exact.size() !=
                kNeighbors
            ) {

                output.ground_truth_complete =
                    false;
            }

            ground_truth.push_back(
                std::move(
                    exact
                )
            );
        }

        //
        // Bitset equivalence para los 16 filtros únicos.
        //

        print_progress(
            "[G-DATA3B] Validando bitsets CPU/GPU..."
        );

        for (
            std::size_t tenant = 0;
            tenant < kTenantCount;
            ++tenant
        ) {

            const FilterSpec filter{
                .tenant_required =
                    static_cast<std::uint32_t>(
                        tenant
                    ),

                .language_min =
                    2U
            };

            materialize_cpu_bitset(
                metadata,
                kVectorCount,
                filter,
                cpu_bits_0
            );

            std::string submit_error;

            id<MTLCommandBuffer> command_buffer =
                submit_gpu_prepare(
                    command_queue,
                    pipeline,
                    metadata_buffer,
                    gpu_buffer_0,
                    column_stride_bytes,
                    filter,
                    output
                        .threads_per_threadgroup,
                    submit_error
                );

            if (
                command_buffer ==
                nil
            ) {

                output.metal_error =
                    submit_error;

                return output;
            }

            const GpuCompletion completion =
                wait_gpu(
                    command_buffer
                );

            if (
                !completion.completed
            ) {

                output.metal_error =
                    completion.error;

                return output;
            }

            output.bitset_word_mismatches +=
                mismatch_word_count(
                    cpu_bits_0,
                    gpu_bits_0,
                    word_count
                );
        }

        //
        // Validación semántica emparejada.
        //

        print_progress(
            "[G-DATA3B] Validando semantica pipeline..."
        );

        double recall_sum =
            0.0;

        double min_recall =
            1.0;

        bool filter_valid =
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

            const LazyPredicate lazy{
                .metadata =
                    metadata,

                .filter =
                    filter
            };

            materialize_cpu_bitset(
                metadata,
                kVectorCount,
                filter,
                cpu_bits_0
            );

            std::string submit_error;

            id<MTLCommandBuffer> command_buffer =
                submit_gpu_prepare(
                    command_queue,
                    pipeline,
                    metadata_buffer,
                    gpu_buffer_0,
                    column_stride_bytes,
                    filter,
                    output
                        .threads_per_threadgroup,
                    submit_error
                );

            if (
                command_buffer ==
                nil
            ) {

                output.metal_error =
                    submit_error;

                return output;
            }

            const GpuCompletion completion =
                wait_gpu(
                    command_buffer
                );

            if (
                !completion.completed
            ) {

                output.metal_error =
                    completion.error;

                return output;
            }

            const BitsetPredicate
                cpu_predicate{
                    .bitset =
                        cpu_bits_0
                };

            const BitsetPredicate
                gpu_predicate{
                    .bitset =
                        gpu_bits_0
                };

            const SearchSnapshot lazy_result =
                search_snapshot(
                    index,
                    queries.row(
                        query_index
                    ),
                    metric,
                    lazy
                );

            const SearchSnapshot cpu_result =
                search_snapshot(
                    index,
                    queries.row(
                        query_index
                    ),
                    metric,
                    cpu_predicate
                );

            const SearchSnapshot gpu_result =
                search_snapshot(
                    index,
                    queries.row(
                        query_index
                    ),
                    metric,
                    gpu_predicate
                );

            if (
                lazy_result.keys !=
                cpu_result.keys
            ) {

                ++output
                    .cpu_topk_mismatches;
            }

            if (
                !same_counters(
                    lazy_result,
                    cpu_result
                )
            ) {

                ++output
                    .cpu_counter_mismatches;
            }

            if (
                lazy_result.keys !=
                gpu_result.keys
            ) {

                ++output
                    .gpu_topk_mismatches;
            }

            if (
                !same_counters(
                    lazy_result,
                    gpu_result
                )
            ) {

                ++output
                    .gpu_counter_mismatches;
            }

            for (
                std::size_t key :
                lazy_result.keys
            ) {

                if (
                    !lazy.accepts_key(
                        key
                    )
                ) {

                    filter_valid =
                        false;
                }
            }

            const double recall =
                recall_at_k(
                    lazy_result.keys,
                    ground_truth[
                        query_index
                    ]
                );

            recall_sum +=
                recall;

            min_recall =
                std::min(
                    min_recall,
                    recall
                );

            if (
                lazy_result.keys.size() ==
                    kNeighbors &&
                std::abs(
                    recall -
                    1.0
                ) <=
                    1e-12
            ) {

                ++output
                    .perfect_queries;

            } else {

                ++output
                    .imperfect_queries;
            }

            if (
                lazy_result.keys.size() <
                kNeighbors
            ) {

                ++output
                    .underfilled_queries;
            }
        }

        output.mean_recall_at_k =
            recall_sum /
            static_cast<double>(
                kQueryCount
            );

        output.min_recall_at_k =
            min_recall;

        output.filter_results_valid =
            filter_valid;

        output.semantic_identity =
            output.bitset_word_mismatches ==
                0U &&

            output.cpu_topk_mismatches ==
                0U &&

            output.cpu_counter_mismatches ==
                0U &&

            output.gpu_topk_mismatches ==
                0U &&

            output.gpu_counter_mismatches ==
                0U;

        if (
            !output.semantic_identity ||
            !output.ground_truth_complete ||
            !output.filter_results_valid
        ) {

            output.decision =
                "INVALID_EXPERIMENT";

            return output;
        }

        //
        // Worker persistente.
        //

        CpuBitsetWorker
            cpu_worker(
                metadata,
                kVectorCount
            );

        //
        // Plan A — Lazy.
        //

        auto run_lazy_batch =
            [&]() -> BatchSample {

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

                    search_sink(
                        index,
                        queries.row(
                            query_index
                        ),
                        metric,
                        predicate,
                        output.benchmark_sink
                    );
                }

                const auto end =
                    Clock::now();

                return BatchSample{
                    .total_ms =
                        milliseconds_between(
                            start,
                            end
                        )
                };
            };

        //
        // Plan B — CPU Pipeline.
        //

        auto run_cpu_pipeline =
            [&]() -> BatchSample {

                BatchSample sample;

                const auto batch_start =
                    Clock::now();

                //
                // Priming q0.
                //

                cpu_worker.submit(
                    filter_for_query(
                        0U
                    ),
                    cpu_bits(
                        0U
                    )
                );

                CpuCompletion completion =
                    cpu_worker.wait();

                sample.exposed_wait_ms +=
                    completion
                        .exposed_wait_ms;

                sample.prepare_work_ms +=
                    completion
                        .work_ms;

                for (
                    std::size_t query_index = 0;
                    query_index < kQueryCount;
                    ++query_index
                ) {

                    const std::size_t current_slot =
                        query_index &
                        1U;

                    const bool has_next =
                        query_index +
                            1U <
                        kQueryCount;

                    if (
                        has_next
                    ) {

                        const std::size_t next_slot =
                            (
                                query_index +
                                1U
                            ) &
                            1U;

                        cpu_worker.submit(
                            filter_for_query(
                                query_index +
                                    1U
                            ),
                            cpu_bits(
                                next_slot
                            )
                        );
                    }

                    const BitsetPredicate predicate{
                        .bitset =
                            cpu_bits(
                                current_slot
                            )
                    };

                    search_sink(
                        index,
                        queries.row(
                            query_index
                        ),
                        metric,
                        predicate,
                        output.benchmark_sink
                    );

                    if (
                        has_next
                    ) {

                        completion =
                            cpu_worker.wait();

                        sample.exposed_wait_ms +=
                            completion
                                .exposed_wait_ms;

                        sample.prepare_work_ms +=
                            completion
                                .work_ms;
                    }
                }

                const auto batch_end =
                    Clock::now();

                sample.total_ms =
                    milliseconds_between(
                        batch_start,
                        batch_end
                    );

                return sample;
            };

        //
        // Plan C — GPU Pipeline.
        //

        auto run_gpu_pipeline =
            [&]() -> BatchSample {

                BatchSample sample;

                const auto batch_start =
                    Clock::now();

                //
                // Priming q0.
                //

                {
                    @autoreleasepool {

                        std::string submit_error;

                        id<MTLCommandBuffer> first =
                            submit_gpu_prepare(
                                command_queue,
                                pipeline,
                                metadata_buffer,
                                gpu_buffer(
                                    0U
                                ),
                                column_stride_bytes,
                                filter_for_query(
                                    0U
                                ),
                                output
                                    .threads_per_threadgroup,
                                submit_error
                            );

                        if (
                            first ==
                            nil
                        ) {

                            throw std::runtime_error(
                                submit_error
                            );
                        }

                        const GpuCompletion completion =
                            wait_gpu(
                                first
                            );

                        if (
                            !completion.completed
                        ) {

                            throw std::runtime_error(
                                completion.error
                            );
                        }

                        sample.exposed_wait_ms +=
                            completion
                                .exposed_wait_ms;

                        if (
                            completion
                                .device_timing_available
                        ) {

                            sample.prepare_work_ms +=
                                completion
                                    .device_ms;
                        }
                    }
                }

                for (
                    std::size_t query_index = 0;
                    query_index < kQueryCount;
                    ++query_index
                ) {

                    @autoreleasepool {

                        const std::size_t current_slot =
                            query_index &
                            1U;

                        const bool has_next =
                            query_index +
                                1U <
                            kQueryCount;

                        id<MTLCommandBuffer> pending =
                            nil;

                        if (
                            has_next
                        ) {

                            const std::size_t next_slot =
                                (
                                    query_index +
                                    1U
                                ) &
                                1U;

                            std::string submit_error;

                            pending =
                                submit_gpu_prepare(
                                    command_queue,
                                    pipeline,
                                    metadata_buffer,
                                    gpu_buffer(
                                        next_slot
                                    ),
                                    column_stride_bytes,
                                    filter_for_query(
                                        query_index +
                                            1U
                                    ),
                                    output
                                        .threads_per_threadgroup,
                                    submit_error
                                );

                            if (
                                pending ==
                                nil
                            ) {

                                throw std::runtime_error(
                                    submit_error
                                );
                            }
                        }

                        const BitsetPredicate predicate{
                            .bitset =
                                gpu_bits(
                                    current_slot
                                )
                        };

                        //
                        // Mientras CPU ejecuta RACORN(q_i),
                        // Metal produce bitset(q_i+1).
                        //

                        search_sink(
                            index,
                            queries.row(
                                query_index
                            ),
                            metric,
                            predicate,
                            output.benchmark_sink
                        );

                        if (
                            has_next
                        ) {

                            const GpuCompletion completion =
                                wait_gpu(
                                    pending
                                );

                            if (
                                !completion.completed
                            ) {

                                throw std::runtime_error(
                                    completion.error
                                );
                            }

                            sample.exposed_wait_ms +=
                                completion
                                    .exposed_wait_ms;

                            if (
                                completion
                                    .device_timing_available
                            ) {

                                sample.prepare_work_ms +=
                                    completion
                                        .device_ms;
                            }
                        }
                    }
                }

                const auto batch_end =
                    Clock::now();

                sample.total_ms =
                    milliseconds_between(
                        batch_start,
                        batch_end
                    );

                return sample;
            };

        //
        // Warm-up.
        //

        print_progress(
            "[G-DATA3B] Semantica PASS. Warm-up..."
        );

        for (
            std::size_t batch = 0;
            batch < kWarmupBatches;
            ++batch
        ) {

            static_cast<void>(
                run_lazy_batch()
            );

            static_cast<void>(
                run_cpu_pipeline()
            );

            static_cast<void>(
                run_gpu_pipeline()
            );
        }

        //
        // Benchmark.
        //

        print_progress(
            "[G-DATA3B] Midiendo pipelines..."
        );

        std::vector<BatchSample>
            lazy_samples;

        std::vector<BatchSample>
            cpu_samples;

        std::vector<BatchSample>
            gpu_samples;

        lazy_samples.reserve(
            kMeasuredBatches
        );

        cpu_samples.reserve(
            kMeasuredBatches
        );

        gpu_samples.reserve(
            kMeasuredBatches
        );

        for (
            std::size_t batch = 0;
            batch < kMeasuredBatches;
            ++batch
        ) {

            std::cout
                << "[G-DATA3B] Batch "
                << (
                    batch +
                    1U
                )
                << "/"
                << kMeasuredBatches
                << '\n'
                << std::flush;

            const std::size_t order =
                batch %
                3U;

            if (
                order ==
                0U
            ) {

                lazy_samples.push_back(
                    run_lazy_batch()
                );

                cpu_samples.push_back(
                    run_cpu_pipeline()
                );

                gpu_samples.push_back(
                    run_gpu_pipeline()
                );

            } else if (
                order ==
                1U
            ) {

                cpu_samples.push_back(
                    run_cpu_pipeline()
                );

                gpu_samples.push_back(
                    run_gpu_pipeline()
                );

                lazy_samples.push_back(
                    run_lazy_batch()
                );

            } else {

                gpu_samples.push_back(
                    run_gpu_pipeline()
                );

                lazy_samples.push_back(
                    run_lazy_batch()
                );

                cpu_samples.push_back(
                    run_cpu_pipeline()
                );
            }
        }

        fill_stats(
            output.cpu_lazy,
            lazy_samples
        );

        fill_stats(
            output.cpu_pipeline,
            cpu_samples
        );

        fill_stats(
            output.gpu_pipeline,
            gpu_samples
        );

        if (
            output.cpu_pipeline.batch_p50_ms >
            0.0
        ) {

            output.cpu_speedup_vs_lazy =
                output.cpu_lazy.batch_p50_ms /
                output.cpu_pipeline.batch_p50_ms;
        }

        if (
            output.gpu_pipeline.batch_p50_ms >
            0.0
        ) {

            output.gpu_speedup_vs_lazy =
                output.cpu_lazy.batch_p50_ms /
                output.gpu_pipeline.batch_p50_ms;
        }

        if (
            output.gpu_pipeline.batch_p50_ms >
            0.0
        ) {

            output.gpu_speedup_vs_cpu =
                output.cpu_pipeline.batch_p50_ms /
                output.gpu_pipeline.batch_p50_ms;
        }

        if (
            output.cpu_pipeline.batch_p95_ms >
            0.0
        ) {

            output.gpu_p95_ratio_vs_cpu =
                output.gpu_pipeline.batch_p95_ms /
                output.cpu_pipeline.batch_p95_ms;
        }

        output.gpu_p50_margin_pass =
            output.gpu_speedup_vs_cpu >=
            kRequiredGpuSpeedup;

        output.gpu_p95_guard_pass =
            output.gpu_p95_ratio_vs_cpu <=
            kAllowedGpuP95Ratio;

        output.gpu_pipeline_worthwhile =
            output.gpu_p50_margin_pass &&
            output.gpu_p95_guard_pass;

        output.decision =
            output.gpu_pipeline_worthwhile
                ? "CONTINUE_METAL"
                : "STOP_METAL_PREDICATE";

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
            output.ground_truth_complete &&
            output.filter_results_valid &&
            output.semantic_identity &&
            lazy_samples.size() ==
                kMeasuredBatches &&
            cpu_samples.size() ==
                kMeasuredBatches &&
            gpu_samples.size() ==
                kMeasuredBatches &&
            !output.racorn_modified;
    }

    print_progress(
        "[G-DATA3B] Experimento completado."
    );

    return output;
}

} // namespace racorn_uma::platform::apple

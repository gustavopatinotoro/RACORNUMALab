//
//  RepeatedPredicateBaseline.cpp
//  RACORNUMALab
//
//  Gate:
//      G-REP0 — Repeated Predicate Sanity Baseline.
//
//  Objetivo:
//  Comparar reparación transitoria RACORN frente a búsqueda exacta sobre
//  un conjunto materializado, cuando el mismo predicado se reutiliza.
//
//  Dataset:
//      SIFT1M train[0:100000]
//      N = 100000
//      D = 128
//      Q = 1000
//
//  Predicado fijo:
//      tenant == 7
//      year >= 2022
//      language >= 2
//      category IN {3,7,9}
//      security_level <= 2
//
//  RACORN:
//      BR  = 1.0
//      TT  = 0.30
//      efS = 200
//      exact fallback = OFF
//
//  Justicia experimental:
//  - El predicado NO se recalcula por query.
//  - RACORN recibe un bitset cacheado.
//  - Exact recibe una lista de IDs cacheada.
//  - HNSW se construye fuera del timing.
//  - El coste de materialización se contabiliza una sola vez.
//
//  Validación exacta:
//  La implementación rápida Exact-Materialized se comprueba contra un
//  oráculo independiente basado en partial_sort.
//
//  Hipótesis:
//  En selectividad extremadamente baja, Exact-Materialized puede dominar
//  a RACORN incluso antes de introducir reparación topológica persistente.
//

#include "core/RepeatedPredicateBaseline.hpp"

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
#include <utility>
#include <vector>

namespace racorn_uma::core {

namespace {

using namespace unum::usearch;

using Clock =
    std::chrono::steady_clock;

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
    1'000U;

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

constexpr std::size_t kMaterializationSamples =
    25U;

constexpr std::size_t kWarmupRounds =
    1U;

constexpr std::size_t kMeasuredRounds =
    5U;

constexpr double kTwoHopThreshold =
    0.30;

constexpr double kBridgeBudgetRatio =
    1.00;

constexpr std::uint32_t kTenantRequired =
    7U;

constexpr std::uint32_t kLanguageMin =
    2U;

constexpr std::array<std::size_t, 8>
    kRepetitionPoints{
        1U,
        10U,
        25U,
        50U,
        100U,
        250U,
        500U,
        1'000U
    };

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
    std::uint32_t key{};
};

struct RacornSnapshot final {

    std::vector<std::uint32_t> keys;

    std::size_t computed_distances{};
    std::size_t visited_members{};
    std::size_t predicate_evaluations{};
    std::size_t fallback_count{};
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
            "G-REP0 requiere arquitectura little-endian."
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
            "Tamaño inesperado para " +
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

std::vector<float> read_query_prefix(
    std::filesystem::path const& path
) {

    std::ifstream input(
        path,
        std::ios::binary
    );

    if (
        !input
    ) {

        throw std::runtime_error(
            "No se pudo abrir archivo de queries."
        );
    }

    const std::size_t element_count =
        kQueryCount *
        kDimensions;

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
            "Lectura incompleta del prefijo de queries."
        );
    }

    return output;
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

void materialize_bitset(
    MetadataView const& metadata,
    FilterSpec const& filter,
    std::uint32_t* output
) noexcept {

    const std::size_t word_count =
        bitset_word_count_for(
            kVectorCount
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
                kVectorCount
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

std::vector<std::uint32_t>
materialize_eligible_ids(
    MetadataView const& metadata,
    FilterSpec const& filter
) {

    std::vector<std::uint32_t>
        eligible;

    for (
        std::size_t key = 0;
        key < kVectorCount;
        ++key
    ) {

        if (
            evaluate_filter(
                metadata,
                key,
                filter
            )
        ) {

            eligible.push_back(
                static_cast<std::uint32_t>(
                    key
                )
            );
        }
    }

    return eligible;
}

float l2_squared(
    float const* first,
    float const* second
) noexcept {

    float sum =
        0.0F;

    for (
        std::size_t dimension = 0;
        dimension < kDimensions;
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

bool candidate_less(
    ExactCandidate const& first,
    ExactCandidate const& second
) noexcept {

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
}

std::vector<std::uint32_t>
exact_reference_top_k(
    DenseVectors const& base,
    float const* query,
    std::vector<std::uint32_t> const& eligible
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
        std::uint32_t key :
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

    std::partial_sort(
        candidates.begin(),
        candidates.begin() +
            static_cast<std::ptrdiff_t>(
                kNeighbors
            ),
        candidates.end(),
        candidate_less
    );

    std::vector<std::uint32_t>
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

std::array<ExactCandidate, kNeighbors>
exact_fast_candidates(
    DenseVectors const& base,
    float const* query,
    std::vector<std::uint32_t> const& eligible
) {

    std::array<
        ExactCandidate,
        kNeighbors
    >
        best{};

    std::size_t count =
        0U;

    for (
        std::uint32_t key :
        eligible
    ) {

        const ExactCandidate candidate{
            .distance =
                l2_squared(
                    query,
                    base.row(
                        key
                    )
                ),

            .key =
                key
        };

        std::size_t position =
            count;

        for (
            std::size_t index = 0;
            index < count;
            ++index
        ) {

            if (
                candidate_less(
                    candidate,
                    best[index]
                )
            ) {

                position =
                    index;

                break;
            }
        }

        if (
            count <
            kNeighbors
        ) {

            for (
                std::size_t index = count;
                index > position;
                --index
            ) {

                best[index] =
                    best[
                        index -
                        1U
                    ];
            }

            best[position] =
                candidate;

            ++count;

        } else if (
            position <
            kNeighbors
        ) {

            for (
                std::size_t index =
                    kNeighbors -
                    1U;
                index > position;
                --index
            ) {

                best[index] =
                    best[
                        index -
                        1U
                    ];
            }

            best[position] =
                candidate;
        }
    }

    return best;
}

std::vector<std::uint32_t>
exact_fast_top_k(
    DenseVectors const& base,
    float const* query,
    std::vector<std::uint32_t> const& eligible
) {

    if (
        eligible.size() <
        kNeighbors
    ) {

        return {};
    }

    const auto best =
        exact_fast_candidates(
            base,
            query,
            eligible
        );

    std::vector<std::uint32_t>
        result;

    result.reserve(
        kNeighbors
    );

    for (
        ExactCandidate const& candidate :
        best
    ) {

        result.push_back(
            candidate.key
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

RacornSnapshot run_racorn(
    Index& index,
    float const* query,
    L2Metric const& metric,
    BitsetPredicate const& predicate
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

    RacornSnapshot output;

    output.keys.reserve(
        result.size()
    );

    for (
        std::size_t index = 0;
        index < result.size();
        ++index
    ) {

        output.keys.push_back(
            static_cast<std::uint32_t>(
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

double recall_at_k(
    std::vector<std::uint32_t> const& approximate,
    std::vector<std::uint32_t> const& exact
) {

    if (
        exact.empty()
    ) {

        return 0.0;
    }

    std::size_t hits =
        0U;

    for (
        std::uint32_t key :
        approximate
    ) {

        if (
            std::find(
                exact.begin(),
                exact.end(),
                key
            ) !=
            exact.end()
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

double mean(
    std::vector<double> const& values
) {

    if (
        values.empty()
    ) {

        return 0.0;
    }

    double sum =
        0.0;

    for (
        double value :
        values
    ) {

        sum +=
            value;
    }

    return
        sum /
        static_cast<double>(
            values.size()
        );
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
    RepeatedPredicateStrategyStats& output,
    std::vector<double> const& samples
) {

    output.mean_ms =
        mean(
            samples
        );

    output.p50_ms =
        percentile(
            samples,
            0.50
        );

    output.p95_ms =
        percentile(
            samples,
            0.95
        );

    output.p99_ms =
        percentile(
            samples,
            0.99
        );

    if (
        output.mean_ms >
        0.0
    ) {

        output.qps_from_mean =
            1000.0 /
            output.mean_ms;
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

RepeatedPredicateBaselineResult
RepeatedPredicateBaseline::run(
    std::filesystem::path const& data_directory
) {

    ensure_little_endian();

    RepeatedPredicateBaselineResult output;

    output.data_directory =
        data_directory;

    output.vector_count =
        kVectorCount;

    output.dimensions =
        kDimensions;

    output.query_count =
        kQueryCount;

    output.k =
        kNeighbors;

    output.connectivity =
        kConnectivity;

    output.ef_construction =
        kEfConstruction;

    output.ef_search =
        kEfSearch;

    output.bridge_budget_ratio =
        kBridgeBudgetRatio;

    output.two_hop_threshold =
        kTwoHopThreshold;

    output.tenant_required =
        kTenantRequired;

    output.language_min =
        kLanguageMin;

    output.materialization_samples =
        kMaterializationSamples;

    output.warmup_rounds =
        kWarmupRounds;

    output.measured_rounds =
        kMeasuredRounds;

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
        "[G-REP0] Validando archivos..."
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
        "[G-REP0] Cargando SIFT100K..."
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
        "[G-REP0] Cargando 1000 queries..."
    );

    DenseVectors queries(
        read_query_prefix(
            query_path
        ),
        kDimensions
    );

    print_progress(
        "[G-REP0] Cargando metadata..."
    );

    auto metadata_storage =
        read_binary_exact<std::uint32_t>(
            metadata_path,
            kMetadataColumnCount *
                kVectorCount
        );

    const MetadataView metadata{
        .tenant =
            metadata_storage.data() +
            0U *
                kVectorCount,

        .category =
            metadata_storage.data() +
            1U *
                kVectorCount,

        .year =
            metadata_storage.data() +
            2U *
                kVectorCount,

        .language =
            metadata_storage.data() +
            3U *
                kVectorCount,

        .security_level =
            metadata_storage.data() +
            4U *
                kVectorCount
    };

    const FilterSpec filter{
        .tenant_required =
            kTenantRequired,

        .language_min =
            kLanguageMin
    };

    const std::size_t word_count =
        bitset_word_count_for(
            kVectorCount
        );

    //
    // Medición robusta del coste one-shot de materialización.
    //

    print_progress(
        "[G-REP0] Midiendo materializacion one-shot..."
    );

    std::vector<double>
        bitset_materialization_samples;

    std::vector<double>
        eligible_materialization_samples;

    bitset_materialization_samples.reserve(
        kMaterializationSamples
    );

    eligible_materialization_samples.reserve(
        kMaterializationSamples
    );

    for (
        std::size_t sample = 0;
        sample < kMaterializationSamples;
        ++sample
    ) {

        {
            const auto start =
                Clock::now();

            std::vector<std::uint32_t>
                temporary_bitset(
                    word_count
                );

            materialize_bitset(
                metadata,
                filter,
                temporary_bitset.data()
            );

            const auto end =
                Clock::now();

            bitset_materialization_samples.push_back(
                milliseconds_between(
                    start,
                    end
                )
            );

            output.benchmark_sink ^=
                temporary_bitset.front();
        }

        {
            const auto start =
                Clock::now();

            auto temporary_ids =
                materialize_eligible_ids(
                    metadata,
                    filter
                );

            const auto end =
                Clock::now();

            eligible_materialization_samples.push_back(
                milliseconds_between(
                    start,
                    end
                )
            );

            output.benchmark_sink ^=
                temporary_ids.size();
        }
    }

    output.racorn_materialization_mean_ms =
        mean(
            bitset_materialization_samples
        );

    output.racorn_materialization_p50_ms =
        percentile(
            bitset_materialization_samples,
            0.50
        );

    output.exact_materialization_mean_ms =
        mean(
            eligible_materialization_samples
        );

    output.exact_materialization_p50_ms =
        percentile(
            eligible_materialization_samples,
            0.50
        );

    //
    // Artefactos cacheados definitivos.
    //

    std::vector<std::uint32_t>
        cached_bitset(
            word_count
        );

    materialize_bitset(
        metadata,
        filter,
        cached_bitset.data()
    );

    auto cached_eligible =
        materialize_eligible_ids(
            metadata,
            filter
        );

    output.eligible_count =
        cached_eligible.size();

    output.selectivity =
        static_cast<double>(
            output.eligible_count
        ) /
        static_cast<double>(
            kVectorCount
        );

    output.bitset_logical_bytes =
        cached_bitset.size() *
        sizeof(
            std::uint32_t
        );

    output.eligible_logical_bytes =
        cached_eligible.size() *
        sizeof(
            std::uint32_t
        );

    output.eligible_allocated_bytes =
        cached_eligible.capacity() *
        sizeof(
            std::uint32_t
        );

    if (
        cached_eligible.size() <
        kNeighbors
    ) {

        throw std::runtime_error(
            "El predicado fijo produce menos de k candidatos."
        );
    }

    //
    // Validar que bitset y lista representan exactamente P.
    //

    print_progress(
        "[G-REP0] Validando materializaciones..."
    );

    const BitsetPredicate
        bitset_predicate{
            .bitset =
                cached_bitset.data()
        };

    std::size_t eligible_cursor =
        0U;

    for (
        std::size_t key = 0;
        key < kVectorCount;
        ++key
    ) {

        const bool expected =
            evaluate_filter(
                metadata,
                key,
                filter
            );

        const bool bitset_value =
            bitset_predicate
                .accepts_key(
                    key
                );

        bool list_value =
            false;

        if (
            eligible_cursor <
                cached_eligible.size() &&
            cached_eligible[
                eligible_cursor
            ] ==
                key
        ) {

            list_value =
                true;

            ++eligible_cursor;
        }

        if (
            expected !=
                bitset_value ||
            expected !=
                list_value
        ) {

            ++output
                .representation_mismatches;
        }
    }

    output.predicate_materialization_valid =
        output.representation_mismatches ==
        0U;

    //
    // Ground truth + validación de implementación exacta rápida.
    //

    print_progress(
        "[G-REP0] Construyendo ground truth exacto..."
    );

    std::vector<
        std::vector<std::uint32_t>
    >
        ground_truth;

    ground_truth.reserve(
        kQueryCount
    );

    bool ground_truth_complete =
        true;

    for (
        std::size_t query_index = 0;
        query_index < kQueryCount;
        ++query_index
    ) {

        auto reference =
            exact_reference_top_k(
                base,
                queries.row(
                    query_index
                ),
                cached_eligible
            );

        auto fast =
            exact_fast_top_k(
                base,
                queries.row(
                    query_index
                ),
                cached_eligible
            );

        if (
            reference.size() !=
            kNeighbors
        ) {

            ground_truth_complete =
                false;
        }

        if (
            reference !=
            fast
        ) {

            ++output
                .exact_reference_mismatch_queries;
        }

        ground_truth.push_back(
            std::move(
                reference
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
                << "[G-REP0] GT "
                << (
                    query_index +
                        1U
                )
                << "/"
                << kQueryCount
                << '\n'
                << std::flush;
        }
    }

    output.ground_truth_complete =
        ground_truth_complete;

    output.exact_implementation_valid =
        output.exact_reference_mismatch_queries ==
        0U;

    //
    // HNSW base.
    //

    print_progress(
        "[G-REP0] Construyendo HNSW base..."
    );

    L2Metric metric{
        .vectors =
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
                << "[G-REP0] HNSW "
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
    // Validación RACORN contra GT.
    //

    print_progress(
        "[G-REP0] Validando RACORN con predicado cacheado..."
    );

    double recall_sum =
        0.0;

    double min_recall =
        1.0;

    double distances_sum =
        0.0;

    double visited_sum =
        0.0;

    double predicate_sum =
        0.0;

    double fallback_sum =
        0.0;

    bool racorn_filter_valid =
        true;

    for (
        std::size_t query_index = 0;
        query_index < kQueryCount;
        ++query_index
    ) {

        const RacornSnapshot result =
            run_racorn(
                index,
                queries.row(
                    query_index
                ),
                metric,
                bitset_predicate
            );

        const double recall =
            recall_at_k(
                result.keys,
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

        distances_sum +=
            static_cast<double>(
                result
                    .computed_distances
            );

        visited_sum +=
            static_cast<double>(
                result
                    .visited_members
            );

        predicate_sum +=
            static_cast<double>(
                result
                    .predicate_evaluations
            );

        fallback_sum +=
            static_cast<double>(
                result
                    .fallback_count
            );

        if (
            result.keys.size() ==
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
            result.keys.size() <
            kNeighbors
        ) {

            ++output
                .underfilled_queries;
        }

        for (
            std::uint32_t key :
            result.keys
        ) {

            if (
                !bitset_predicate
                    .accepts_key(
                        key
                    )
            ) {

                racorn_filter_valid =
                    false;

                ++output
                    .invalid_racorn_result_queries;

                break;
            }
        }
    }

    output.mean_recall_at_k =
        recall_sum /
        static_cast<double>(
            kQueryCount
        );

    output.min_recall_at_k =
        min_recall;

    output.mean_racorn_computed_distances =
        distances_sum /
        static_cast<double>(
            kQueryCount
        );

    output.mean_racorn_visited_members =
        visited_sum /
        static_cast<double>(
            kQueryCount
        );

    output.mean_racorn_predicate_evaluations =
        predicate_sum /
        static_cast<double>(
            kQueryCount
        );

    output.mean_racorn_fallback_count =
        fallback_sum /
        static_cast<double>(
            kQueryCount
        );

    output.exact_distance_evaluations_per_query =
        static_cast<double>(
            cached_eligible.size()
        );

    output.racorn_results_filter_valid =
        racorn_filter_valid;

    //
    // Warm-up.
    //

    print_progress(
        "[G-REP0] Warm-up..."
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

            const auto racorn_result =
                run_racorn(
                    index,
                    queries.row(
                        query_index
                    ),
                    metric,
                    bitset_predicate
                );

            const auto exact_result =
                exact_fast_top_k(
                    base,
                    queries.row(
                        query_index
                    ),
                    cached_eligible
                );

            output.benchmark_sink ^=
                racorn_result.keys.size();

            output.benchmark_sink ^=
                exact_result.front();
        }
    }

    //
    // Benchmark.
    //

    print_progress(
        "[G-REP0] Midiendo RACORN-Cached vs Exact-Materialized..."
    );

    std::vector<double>
        racorn_samples;

    std::vector<double>
        exact_samples;

    racorn_samples.reserve(
        kQueryCount *
        kMeasuredRounds
    );

    exact_samples.reserve(
        kQueryCount *
        kMeasuredRounds
    );

    auto measure_racorn_round =
        [&]() {

            for (
                std::size_t query_index = 0;
                query_index < kQueryCount;
                ++query_index
            ) {

                const auto start =
                    Clock::now();

                const auto result =
                    run_racorn(
                        index,
                        queries.row(
                            query_index
                        ),
                        metric,
                        bitset_predicate
                    );

                const auto end =
                    Clock::now();

                racorn_samples.push_back(
                    milliseconds_between(
                        start,
                        end
                    )
                );

                output.benchmark_sink ^=
                    result.keys.size();

                if (
                    !result.keys.empty()
                ) {

                    output.benchmark_sink ^=
                        result.keys.front();
                }
            }
        };

    auto measure_exact_round =
        [&]() {

            for (
                std::size_t query_index = 0;
                query_index < kQueryCount;
                ++query_index
            ) {

                const auto start =
                    Clock::now();

                const auto result =
                    exact_fast_top_k(
                        base,
                        queries.row(
                            query_index
                        ),
                        cached_eligible
                    );

                const auto end =
                    Clock::now();

                exact_samples.push_back(
                    milliseconds_between(
                        start,
                        end
                    )
                );

                output.benchmark_sink ^=
                    result.front();
            }
        };

    for (
        std::size_t round = 0;
        round < kMeasuredRounds;
        ++round
    ) {

        std::cout
            << "[G-REP0] Ronda "
            << (
                round +
                    1U
            )
            << "/"
            << kMeasuredRounds
            << '\n'
            << std::flush;

        if (
            round %
                2U ==
                0U
        ) {

            measure_racorn_round();
            measure_exact_round();

        } else {

            measure_exact_round();
            measure_racorn_round();
        }
    }

    fill_stats(
        output.racorn,
        racorn_samples
    );

    fill_stats(
        output.exact,
        exact_samples
    );

    output.exact_query_faster =
        output.exact.mean_ms <
        output.racorn.mean_ms;

    output.exact_build_cheaper =
        output.exact_materialization_mean_ms <=
        output.racorn_materialization_mean_ms;

    if (
        output.exact.mean_ms >
        0.0
    ) {

        output.exact_speedup_vs_racorn =
            output.racorn.mean_ms /
            output.exact.mean_ms;
    }

    //
    // Break-even de Exact frente a RACORN.
    //
    // B_E + R L_E <= B_R + R L_R
    //

    if (
        output.exact_query_faster
    ) {

        output.exact_break_even_exists =
            true;

        const double build_penalty =
            output.exact_materialization_mean_ms -
            output.racorn_materialization_mean_ms;

        if (
            build_penalty <=
            0.0
        ) {

            output.exact_break_even_repetitions =
                0.0;

        } else {

            const double saving_per_query =
                output.racorn.mean_ms -
                output.exact.mean_ms;

            output.exact_break_even_repetitions =
                build_penalty /
                saving_per_query;
        }
    }

    //
    // Coste acumulado para frecuencias de reutilización concretas.
    //

    for (
        std::size_t repetitions :
        kRepetitionPoints
    ) {

        const double racorn_cost =
            output.racorn_materialization_mean_ms +
            static_cast<double>(
                repetitions
            ) *
            output.racorn.mean_ms;

        const double exact_cost =
            output.exact_materialization_mean_ms +
            static_cast<double>(
                repetitions
            ) *
            output.exact.mean_ms;

        output.cumulative_costs.push_back(
            RepeatedPredicateCostPoint{
                .repetitions =
                    repetitions,

                .racorn_cumulative_ms =
                    racorn_cost,

                .exact_cumulative_ms =
                    exact_cost,

                .winner =
                    exact_cost <
                            racorn_cost
                        ? "Exact-Materialized"
                        : "RACORN-Cached"
            }
        );
    }

    output.valid =
        output.base_file_valid &&
        output.query_file_valid &&
        output.metadata_file_valid &&
        output.ground_truth_complete &&
        output.exact_implementation_valid &&
        output.predicate_materialization_valid &&
        output.racorn_results_filter_valid &&
        !output.racorn_modified;

    print_progress(
        "[G-REP0] Experimento completado."
    );

    return output;
}

} // namespace racorn_uma::core

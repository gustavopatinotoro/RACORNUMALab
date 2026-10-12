//
//  FilteredSearchBaseline.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar sobre un único índice HNSW las rutas:
//
//      1. HNSW filtrado
//      2. ACORN-1
//      3. RACORN-1
//
//  Gate:
//  G4A — Activación funcional de búsqueda filtrada.
//
//  Dataset:
//  - Sintético.
//  - Determinista.
//  - Distribución pseudoaleatoria.
//  - 20.000 vectores de 32 dimensiones.
//
//  Filtro:
//      key % 200 == 0
//
//  Selectividad nominal:
//      1 / 200 = 0.005 = 0.5 %
//
//  Ground truth:
//  Exact search filtrado del propio index_gt.
//
//  Importante:
//  Este Gate no pretende establecer todavía rendimiento definitivo.
//  Su finalidad es comprobar que las tres rutas algorítmicas funcionan,
//  respetan el predicado y producen telemetría diferenciable.
//

#include "filtered/FilteredSearchBaseline.hpp"

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <unordered_set>
#include <utility>
#include <vector>

namespace racorn_uma::filtered {

namespace {

using namespace unum::usearch;

constexpr std::size_t kVectorCount = 20'000;
constexpr std::size_t kDimensions = 32;

constexpr std::size_t kConnectivity = 16;
constexpr std::size_t kEfConstruction = 100;
constexpr std::size_t kEfSearch = 200;

constexpr std::size_t kNeighbors = 10;
constexpr std::size_t kQueryCount = 32;

constexpr std::size_t kFilterModulus = 200;
constexpr std::size_t kFilterThreshold = 1;

constexpr double kTwoHopThreshold = 0.30;
constexpr double kBridgeBudgetRatio = 1.0;

using VectorKey = std::size_t;
using Slot = std::size_t;

using Index =
    index_gt<float, VectorKey, Slot>;

//
// PRNG determinista SplitMix64.
//
// No se utiliza std::random_device para garantizar reproducibilidad.
//

std::uint64_t splitmix64(
    std::uint64_t& state
) noexcept {

    state +=
        0x9E3779B97F4A7C15ULL;

    std::uint64_t value =
        state;

    value =
        (value ^ (value >> 30U)) *
        0xBF58476D1CE4E5B9ULL;

    value =
        (value ^ (value >> 27U)) *
        0x94D049BB133111EBULL;

    return
        value ^
        (value >> 31U);
}

float random_signed_float(
    std::uint64_t& state
) noexcept {

    constexpr double denominator =
        16777215.0;

    const std::uint32_t bits =
        static_cast<std::uint32_t>(
            splitmix64(state) >> 40U
        );

    const double unit =
        static_cast<double>(bits) /
        denominator;

    return
        static_cast<float>(
            unit * 2.0 - 1.0
        );
}

class VectorStore final {

public:

    VectorStore(
        std::size_t count,
        std::size_t dimensions,
        std::uint64_t seed
    )
        : dimensions_(dimensions),
          data_(
              count * dimensions
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
            index * dimensions_;
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
        : dimensions_(dimensions),
          data_(
              count * dimensions
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
            index * dimensions_;
    }

private:

    std::size_t dimensions_{};
    std::vector<float> data_;
};

struct L2Metric final {

    using IndexType = Index;

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
            store->row(key);
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
                row(get_key(second)),
                row(get_key(first)),
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
                row(get_key(member)),
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
                row(get_key(*second)),
                row(get_key(*first)),
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
                row(get_key(*member)),
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
            Allocator{}.allocate(1);

        new (pointer_)
            IndexType(configuration);
    }

    ~AlignedIndex() {

        if (pointer_ == nullptr) {
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

    IndexType* pointer_{nullptr};
};

struct MetadataPredicate final {

    template <typename MemberLike>
    bool operator()(
        MemberLike&& member
    ) const noexcept {

        return
            (
                get_key(member) %
                kFilterModulus
            ) <
            kFilterThreshold;
    }

    [[nodiscard]]
    static bool accepts_key(
        std::size_t key
    ) noexcept {

        return
            (
                key %
                kFilterModulus
            ) <
            kFilterThreshold;
    }
};

template <typename SearchResult>
std::vector<std::size_t>
extract_keys(
    SearchResult const& result
) {

    std::vector<std::size_t> keys;

    keys.reserve(
        result.size()
    );

    for (
        std::size_t i = 0;
        i < result.size();
        ++i
    ) {

        keys.push_back(
            static_cast<std::size_t>(
                result[i].member.key
            )
        );
    }

    return keys;
}

double recall_at_k(
    std::vector<std::size_t> const& approximate,
    std::vector<std::size_t> const& exact
) {

    if (exact.empty()) {
        return 0.0;
    }

    std::unordered_set<std::size_t>
        truth(
            exact.begin(),
            exact.end()
        );

    std::size_t hits = 0;

    for (
        std::size_t key :
        approximate
    ) {

        if (truth.contains(key)) {
            ++hits;
        }
    }

    return
        static_cast<double>(hits) /
        static_cast<double>(
            exact.size()
        );
}

bool filter_is_valid(
    std::vector<std::size_t> const& keys
) noexcept {

    for (
        std::size_t key :
        keys
    ) {

        if (
            !MetadataPredicate::
                accepts_key(key)
        ) {

            return false;
        }
    }

    return true;
}

enum class SearchMode {
    hnsw,
    acorn,
    racorn
};

index_search_config_t
make_search_config(
    SearchMode mode
) {

    index_search_config_t config;

    config.expansion =
        kEfSearch;

    config.thread =
        0;

    config.three_hop_threshold =
        -1.0;

    config.fallback_exact_threshold =
        0.0;

    config.bridge_budget_ratio =
        kBridgeBudgetRatio;

    config.racorn_stride_sampling =
        true;

    switch (mode) {

    case SearchMode::hnsw:

        config.two_hop_threshold =
            -1.0;

        config.racorn =
            false;

        break;

    case SearchMode::acorn:

        config.two_hop_threshold =
            kTwoHopThreshold;

        config.racorn =
            false;

        break;

    case SearchMode::racorn:

        config.two_hop_threshold =
            kTwoHopThreshold;

        config.racorn =
            true;

        break;
    }

    return config;
}

struct Accumulator final {

    double recall_sum{};
    double result_count_sum{};
    double computed_distances_sum{};
    double visited_members_sum{};
    double predicate_evaluations_sum{};
    double fallback_count_sum{};

    double fallback_selectivity_sum{};

    std::size_t fallback_selectivity_samples{};
    std::size_t queries_with_fallback{};

    bool filter_valid = true;
};

void accumulate(
    Accumulator& accumulator,
    std::vector<std::size_t> const& keys,
    std::vector<std::size_t> const& ground_truth,
    Index::search_result_t const& result
) {

    accumulator.recall_sum +=
        recall_at_k(
            keys,
            ground_truth
        );

    accumulator.result_count_sum +=
        static_cast<double>(
            keys.size()
        );

    accumulator.computed_distances_sum +=
        static_cast<double>(
            result.computed_distances
        );

    accumulator.visited_members_sum +=
        static_cast<double>(
            result.visited_members
        );

    accumulator.predicate_evaluations_sum +=
        static_cast<double>(
            result.predicate_evaluations
        );

    accumulator.fallback_count_sum +=
        static_cast<double>(
            result.fallback_count
        );

    if (
        result.fallback_count > 0
    ) {

        ++accumulator.queries_with_fallback;

        if (
            std::isfinite(
                result.fallback_selectivity
            )
        ) {

            accumulator.fallback_selectivity_sum +=
                result.fallback_selectivity;

            ++accumulator
                .fallback_selectivity_samples;
        }
    }

    accumulator.filter_valid =
        accumulator.filter_valid &&
        filter_is_valid(keys);
}

FilteredModeStats finish_stats(
    char const* name,
    Accumulator const& accumulator
) {

    const double query_count =
        static_cast<double>(
            kQueryCount
        );

    FilteredModeStats stats;

    stats.name =
        name;

    stats.mean_recall_at_k =
        accumulator.recall_sum /
        query_count;

    stats.mean_result_count =
        accumulator.result_count_sum /
        query_count;

    stats.mean_computed_distances =
        accumulator.computed_distances_sum /
        query_count;

    stats.mean_visited_members =
        accumulator.visited_members_sum /
        query_count;

    stats.mean_predicate_evaluations =
        accumulator.predicate_evaluations_sum /
        query_count;

    stats.mean_fallback_count =
        accumulator.fallback_count_sum /
        query_count;

    stats.queries_with_fallback =
        accumulator.queries_with_fallback;

    if (
        accumulator
            .fallback_selectivity_samples >
        0
    ) {

        stats.mean_fallback_selectivity =
            accumulator
                .fallback_selectivity_sum /
            static_cast<double>(
                accumulator
                    .fallback_selectivity_samples
            );
    } else {

        stats.mean_fallback_selectivity =
            std::numeric_limits<
                double
            >::quiet_NaN();
    }

    stats.all_results_satisfy_filter =
        accumulator.filter_valid;

    return stats;
}

} // namespace

FilteredSearchResult
FilteredSearchBaseline::run() {

    VectorStore base(
        kVectorCount,
        kDimensions,
        0x13579BDF2468ACE0ULL
    );

    QueryStore queries(
        kQueryCount,
        kDimensions,
        0xCAFEBABE12345678ULL
    );

    L2Metric metric{
        .store = &base,
        .dimensions = kDimensions
    };

    index_config_t index_configuration(
        kConnectivity
    );

    AlignedIndex<Index> aligned_index(
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

    index_update_config_t update_configuration;

    update_configuration.expansion =
        kEfConstruction;

    update_configuration.thread =
        0;

    for (
        std::size_t i = 0;
        i < kVectorCount;
        ++i
    ) {

        index.add(
            static_cast<VectorKey>(i),
            base.row(i),
            metric,
            update_configuration
        );
    }

    MetadataPredicate predicate;

    Accumulator hnsw_accumulator;
    Accumulator acorn_accumulator;
    Accumulator racorn_accumulator;

    bool exact_complete =
        true;

    for (
        std::size_t query_index = 0;
        query_index < kQueryCount;
        ++query_index
    ) {

        float const* query =
            queries.row(
                query_index
            );

        //
        // Exact filtered ground truth.
        //

        index_search_config_t exact_config;

        exact_config.exact =
            true;

        exact_config.thread =
            0;

        auto exact_result =
            index.search(
                query,
                kNeighbors,
                metric,
                exact_config,
                predicate
            );

        const auto exact_keys =
            extract_keys(
                exact_result
            );

        if (
            exact_keys.size() !=
                kNeighbors ||
            !filter_is_valid(
                exact_keys
            )
        ) {

            exact_complete =
                false;
        }

        //
        // HNSW in-filtering.
        //

        auto hnsw_config =
            make_search_config(
                SearchMode::hnsw
            );

        auto hnsw_result =
            index.search(
                query,
                kNeighbors,
                metric,
                hnsw_config,
                predicate
            );

        const auto hnsw_keys =
            extract_keys(
                hnsw_result
            );

        accumulate(
            hnsw_accumulator,
            hnsw_keys,
            exact_keys,
            hnsw_result
        );

        //
        // ACORN-1.
        //

        auto acorn_config =
            make_search_config(
                SearchMode::acorn
            );

        auto acorn_result =
            index.search(
                query,
                kNeighbors,
                metric,
                acorn_config,
                predicate
            );

        const auto acorn_keys =
            extract_keys(
                acorn_result
            );

        accumulate(
            acorn_accumulator,
            acorn_keys,
            exact_keys,
            acorn_result
        );

        //
        // RACORN-1.
        //

        auto racorn_config =
            make_search_config(
                SearchMode::racorn
            );

        auto racorn_result =
            index.search(
                query,
                kNeighbors,
                metric,
                racorn_config,
                predicate
            );

        const auto racorn_keys =
            extract_keys(
                racorn_result
            );

        accumulate(
            racorn_accumulator,
            racorn_keys,
            exact_keys,
            racorn_result
        );
    }

    std::size_t eligible_count =
        0;

    for (
        std::size_t key = 0;
        key < kVectorCount;
        ++key
    ) {

        if (
            MetadataPredicate::
                accepts_key(key)
        ) {

            ++eligible_count;
        }
    }

    FilteredSearchResult output;

    output.vector_count =
        index.size();

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

    output.eligible_count =
        eligible_count;

    output.nominal_selectivity =
        static_cast<double>(
            kFilterThreshold
        ) /
        static_cast<double>(
            kFilterModulus
        );

    output.observed_selectivity =
        static_cast<double>(
            eligible_count
        ) /
        static_cast<double>(
            kVectorCount
        );

    output.hnsw =
        finish_stats(
            "HNSW",
            hnsw_accumulator
        );

    output.acorn =
        finish_stats(
            "ACORN-1",
            acorn_accumulator
        );

    output.racorn =
        finish_stats(
            "RACORN-1",
            racorn_accumulator
        );

    output.exact_ground_truth_complete =
        exact_complete;

    output.racorn_fallback_observed =
        output.racorn
            .queries_with_fallback >
        0;

    const bool filters_valid =
        output.hnsw
            .all_results_satisfy_filter &&
        output.acorn
            .all_results_satisfy_filter &&
        output.racorn
            .all_results_satisfy_filter;

    output.passed =
        output.vector_count ==
            kVectorCount &&
        output.eligible_count >=
            kNeighbors &&
        output.exact_ground_truth_complete &&
        filters_valid &&
        output.racorn_fallback_observed;

    return output;
}

} // namespace racorn_uma::filtered

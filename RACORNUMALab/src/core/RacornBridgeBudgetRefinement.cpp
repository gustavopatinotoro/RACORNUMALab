//
//  RacornBridgeBudgetRefinement.cpp
//  RACORNUMALab
//
//  Propósito:
//  Implementar G-UMA5E.
//
//  Este Gate elimina deliberadamente GPU, Metal y cualquier estrategia
//  de materialización de filtros para aislar exclusivamente:
//
//      bridge_budget_ratio
//          ↓
//      Recall
//      trabajo RACORN
//
//  Configuración:
//
//      N  ∈ {250000, 500000}
//
//      BR ∈ {
//          0.50,
//          0.55,
//          0.60,
//          0.65,
//          0.70,
//          0.75
//      }
//
//      D   = 32
//      M   = 16
//      efC = 100
//      efS = 200
//      k   = 10
//      Q   = 16
//
//  Filtro:
//
//      tenant_id == query_index
//      AND year >= 2022
//      AND language >= 2
//      AND category IN {3,7,9}
//      AND security_level <= 2
//
//  Ground truth:
//  Scan exacto independiente sobre los elementos que satisfacen el
//  predicado.
//
//  Criterio robusto:
//
//      mean Recall@10 >= 0.99
//
//      AND
//
//      min Recall@10 >= 0.99
//
//  Debido a k=10, Recall@10 por query toma incrementos de 0.1.
//  Por ello min Recall@10 >= 0.99 equivale en este experimento a exigir
//  Recall@10 = 1.0 para cada una de las 16 queries.
//
//  Invariantes:
//  - RACORN upstream permanece intacto.
//  - Navegación exclusivamente CPU.
//  - Sin GPU.
//  - Sin Metal.
//  - Sin bitset.
//  - Sin exact=true de USEARCH.
//  - El índice se construye una sola vez por N.
//

#include "core/RacornBridgeBudgetRefinement.hpp"

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new>
#include <unordered_set>
#include <vector>

namespace racorn_uma::core {

namespace {

using namespace unum::usearch;

constexpr std::array<std::size_t, 2>
    kVectorCounts{
        250'000U,
        500'000U
    };

constexpr std::array<double, 6>
    kBridgeRatios{
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75
    };

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

constexpr double kTwoHopThreshold =
    0.30;

constexpr double kMeanRecallTarget =
    0.99;

constexpr double kMinRecallTarget =
    0.99;

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

struct MetadataColumns final {

    explicit MetadataColumns(
        std::size_t count
    )
        : tenant(
              count
          ),
          category(
              count
          ),
          year(
              count
          ),
          language(
              count
          ),
          security_level(
              count
          ) {
    }

    std::vector<std::uint32_t> tenant;
    std::vector<std::uint32_t> category;
    std::vector<std::uint32_t> year;
    std::vector<std::uint32_t> language;
    std::vector<std::uint32_t> security_level;
};

struct ExactCandidate final {

    float distance{};

    std::size_t key{};
};

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
          values_(
              count *
              dimensions
          ) {

        std::uint64_t state =
            seed;

        for (
            float& value :
            values_
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
            values_.data() +
            index *
                dimensions_;
    }

private:

    static std::uint64_t
    splitmix64_step(
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

    static float
    random_signed_float(
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

    std::size_t dimensions_{};

    std::vector<float> values_;
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
          values_(
              count *
              dimensions
          ) {

        std::uint64_t state =
            seed;

        for (
            float& value :
            values_
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
            values_.data() +
            index *
                dimensions_;
    }

private:

    static std::uint64_t
    splitmix64_step(
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

    static float
    random_signed_float(
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

    std::size_t dimensions_{};

    std::vector<float> values_;
};

std::uint64_t splitmix64_value(
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

void generate_metadata(
    MetadataColumns& metadata
) {

    const std::size_t count =
        metadata.tenant.size();

    for (
        std::size_t index = 0;
        index < count;
        ++index
    ) {

        const std::uint64_t hash =
            splitmix64_value(
                kMetadataSeed ^
                static_cast<std::uint64_t>(
                    index
                )
            );

        metadata.tenant[index] =
            static_cast<std::uint32_t>(
                hash &
                0x0FU
            );

        metadata.category[index] =
            static_cast<std::uint32_t>(
                (
                    hash >>
                    8U
                ) &
                0x0FU
            );

        metadata.year[index] =
            2019U +
            static_cast<std::uint32_t>(
                (
                    hash >>
                    16U
                ) %
                8U
            );

        metadata.language[index] =
            static_cast<std::uint32_t>(
                (
                    hash >>
                    24U
                ) &
                0x03U
            );

        metadata.security_level[index] =
            static_cast<std::uint32_t>(
                (
                    hash >>
                    32U
                ) %
                6U
            );
    }
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
    MetadataColumns const& metadata,
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

struct LazyPredicate final {

    MetadataColumns const* metadata{};

    FilterSpec filter{};

    [[nodiscard]]
    bool accepts_key(
        std::size_t key
    ) const noexcept {

        return
            evaluate_filter(
                *metadata,
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

std::vector<std::size_t>
exact_filtered_top_k(
    VectorStore const& base,
    float const* query,
    MetadataColumns const& metadata,
    FilterSpec const& filter,
    std::size_t vector_count
) {

    std::vector<ExactCandidate>
        candidates;

    candidates.reserve(
        2048U
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

        const float distance =
            metric_l2sq_gt<
                float,
                float
            >{}(
                query,
                base.row(
                    key
                ),
                kDimensions
            );

        candidates.push_back(
            ExactCandidate{
                .distance =
                    distance,

                .key =
                    key
            }
        );
    }

    if (
        candidates.size() <
        kNeighbors
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
            candidates[index].key
        );
    }

    return result;
}

double recall_at_k(
    Index::search_result_t const& approximate,
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
        std::size_t index = 0;
        index < approximate.size();
        ++index
    ) {

        const std::size_t key =
            static_cast<std::size_t>(
                approximate[index]
                    .member
                    .key
            );

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

bool results_respect_filter(
    Index::search_result_t const& result,
    LazyPredicate const& predicate
) noexcept {

    for (
        std::size_t index = 0;
        index < result.size();
        ++index
    ) {

        const std::size_t key =
            static_cast<std::size_t>(
                result[index]
                    .member
                    .key
            );

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

index_search_config_t
make_racorn_configuration(
    double bridge_budget_ratio
) {

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
        bridge_budget_ratio;

    config.three_hop_threshold =
        -1.0;

    config.fallback_exact_threshold =
        0.0;

    config.racorn_stride_sampling =
        true;

    return config;
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

RacornBridgeBudgetRefinementResult
RacornBridgeBudgetRefinement::run() {

    RacornBridgeBudgetRefinementResult
        output;

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

    output.mean_recall_target =
        kMeanRecallTarget;

    output.min_recall_target =
        kMinRecallTarget;

    output.racorn_modified =
        false;

    bool all_cases_valid =
        true;

    bool robust_point_for_all =
        true;

    for (
        std::size_t vector_count :
        kVectorCounts
    ) {

        std::cout
            << "\n[G-UMA5E] N="
            << vector_count
            << " - generando datos...\n"
            << std::flush;

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

        MetadataColumns metadata(
            vector_count
        );

        generate_metadata(
            metadata
        );

        L2Metric metric{
            .store =
                &base,

            .dimensions =
                kDimensions
        };

        print_progress(
            "[G-UMA5E] Reservando indice..."
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
                    << "[G-UMA5E] N="
                    << vector_count
                    << " HNSW "
                    << percent
                    << "%\n"
                    << std::flush;
            }
        }

        //
        // Ground truth: se calcula una sola vez por N.
        //

        print_progress(
            "[G-UMA5E] Construyendo ground truth filtrado..."
        );

        std::vector<
            std::vector<std::size_t>
        >
            exact_by_query;

        exact_by_query.reserve(
            kQueryCount
        );

        std::vector<std::size_t>
            selected_by_query;

        selected_by_query.reserve(
            kQueryCount
        );

        bool ground_truth_complete =
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

            std::size_t selected =
                0;

            for (
                std::size_t key = 0;
                key < vector_count;
                ++key
            ) {

                if (
                    evaluate_filter(
                        metadata,
                        key,
                        filter
                    )
                ) {

                    ++selected;
                }
            }

            selected_by_query.push_back(
                selected
            );

            auto exact =
                exact_filtered_top_k(
                    base,
                    queries.row(
                        query_index
                    ),
                    metadata,
                    filter,
                    vector_count
                );

            if (
                exact.size() !=
                kNeighbors
            ) {

                ground_truth_complete =
                    false;
            }

            exact_by_query.push_back(
                std::move(
                    exact
                )
            );
        }

        const std::size_t selected_min =
            *std::min_element(
                selected_by_query.begin(),
                selected_by_query.end()
            );

        const std::size_t selected_max =
            *std::max_element(
                selected_by_query.begin(),
                selected_by_query.end()
            );

        std::size_t selected_sum =
            0;

        for (
            std::size_t selected :
            selected_by_query
        ) {

            selected_sum +=
                selected;
        }

        const double mean_selectivity =
            static_cast<double>(
                selected_sum
            ) /
            (
                static_cast<double>(
                    kQueryCount
                ) *
                static_cast<double>(
                    vector_count
                )
            );

        BridgeBudgetOptimalPoint
            optimal;

        optimal.vector_count =
            vector_count;

        //
        // Sweep BR.
        //

        for (
            double bridge_ratio :
            kBridgeRatios
        ) {

            std::cout
                << "[G-UMA5E] N="
                << vector_count
                << " BR="
                << bridge_ratio
                << "...\n"
                << std::flush;

            BridgeBudgetRefinementCase
                case_result;

            case_result.vector_count =
                vector_count;

            case_result.bridge_budget_ratio =
                bridge_ratio;

            case_result.mean_selectivity =
                mean_selectivity;

            case_result.selected_min =
                selected_min;

            case_result.selected_max =
                selected_max;

            case_result.exact_ground_truth_complete =
                ground_truth_complete;

            case_result.result_count_min =
                std::numeric_limits<
                    std::size_t
                >::max();

            double recall_sum =
                0.0;

            double minimum_recall =
                1.0;

            std::size_t worst_query =
                0;

            double distances_sum =
                0.0;

            double visited_sum =
                0.0;

            double predicates_sum =
                0.0;

            double fallback_sum =
                0.0;

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
                        &metadata,

                    .filter =
                        filter
                };

                auto configuration =
                    make_racorn_configuration(
                        bridge_ratio
                    );

                auto result =
                    index.search(
                        queries.row(
                            query_index
                        ),
                        kNeighbors,
                        metric,
                        configuration,
                        predicate
                    );

                const double recall =
                    recall_at_k(
                        result,
                        exact_by_query[
                            query_index
                        ]
                    );

                recall_sum +=
                    recall;

                if (
                    recall <
                    minimum_recall
                ) {

                    minimum_recall =
                        recall;

                    worst_query =
                        query_index;
                }

                distances_sum +=
                    static_cast<double>(
                        result.computed_distances
                    );

                visited_sum +=
                    static_cast<double>(
                        result.visited_members
                    );

                predicates_sum +=
                    static_cast<double>(
                        result.predicate_evaluations
                    );

                fallback_sum +=
                    static_cast<double>(
                        result.fallback_count
                    );

                case_result.result_count_min =
                    std::min(
                        case_result.result_count_min,
                        result.size()
                    );

                case_result.result_count_max =
                    std::max(
                        case_result.result_count_max,
                        result.size()
                    );

                filters_valid =
                    filters_valid &&
                    results_respect_filter(
                        result,
                        predicate
                    );
            }

            const double query_count =
                static_cast<double>(
                    kQueryCount
                );

            case_result.mean_recall_at_k =
                recall_sum /
                query_count;

            case_result.min_recall_at_k =
                minimum_recall;

            case_result.worst_query_index =
                worst_query;

            case_result.mean_computed_distances =
                distances_sum /
                query_count;

            case_result.mean_visited_members =
                visited_sum /
                query_count;

            case_result.mean_predicate_evaluations =
                predicates_sum /
                query_count;

            case_result.mean_fallback_count =
                fallback_sum /
                query_count;

            case_result.returned_results_filter_valid =
                filters_valid;

            case_result.mean_recall_pass =
                case_result.mean_recall_at_k +
                    1e-12 >=
                kMeanRecallTarget;

            case_result.min_recall_pass =
                case_result.min_recall_at_k +
                    1e-12 >=
                kMinRecallTarget;

            case_result.robust_quality_pass =
                case_result.mean_recall_pass &&
                case_result.min_recall_pass;

            case_result.valid =
                case_result.exact_ground_truth_complete &&
                case_result.returned_results_filter_valid &&
                case_result.result_count_min >
                    0;

            all_cases_valid =
                all_cases_valid &&
                case_result.valid;

            //
            // Primer BR robustamente admisible = mínimo del sweep.
            //

            if (
                !optimal.found &&
                case_result.valid &&
                case_result.robust_quality_pass
            ) {

                optimal.found =
                    true;

                optimal.minimum_robust_bridge_budget_ratio =
                    bridge_ratio;

                optimal.mean_recall_at_k =
                    case_result.mean_recall_at_k;

                optimal.min_recall_at_k =
                    case_result.min_recall_at_k;

                optimal.mean_computed_distances =
                    case_result.mean_computed_distances;

                optimal.mean_predicate_evaluations =
                    case_result.mean_predicate_evaluations;
            }

            output.cases.push_back(
                case_result
            );
        }

        robust_point_for_all =
            robust_point_for_all &&
            optimal.found;

        output.optimal_points.push_back(
            optimal
        );
    }

    output.all_cases_valid =
        all_cases_valid;

    output.robust_point_found_for_all_sizes =
        robust_point_for_all;

    output.valid =
        output.all_cases_valid &&
        output.robust_point_found_for_all_sizes &&
        !output.racorn_modified;

    print_progress(
        "[G-UMA5E] Refinamiento completado."
    );

    return output;
}

} // namespace racorn_uma::core

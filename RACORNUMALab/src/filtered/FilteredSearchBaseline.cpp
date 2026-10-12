//
//  FilteredSearchBaseline.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar múltiples realizaciones de un filtro con selectividad
//  aproximadamente constante sobre exactamente el mismo índice HNSW,
//  comparando:
//
//      1. HNSW con in-filtering.
//      2. ACORN-1.
//      3. RACORN-1.
//
//  Gate:
//  G4C — Replicación topológica del filtro.
//
//  Motivación:
//  G4B mostró que el Recall de ACORN no es perfectamente monótono
//  respecto de la selectividad. Esa observación impide interpretar
//  una única realización del filtro como un umbral universal.
//
//  Este Gate separa:
//
//      efecto de selectividad
//
//  de:
//
//      efecto de la realización concreta del filtro sobre el grafo.
//
//  Dataset:
//  - 20.000 vectores.
//  - 32 dimensiones.
//  - Pseudoaleatorio determinista.
//
//  Consultas:
//  - 32 consultas.
//  - Pseudoaleatorias deterministas.
//  - Independientes de los vectores base.
//
//  Predicado:
//  Para residue = r:
//
//      shifted_residue(key, r) < threshold
//
//  Con threshold = 1 equivale a:
//
//      key % modulus == r
//
//  Los residuos utilizados se distribuyen de forma aproximadamente
//  uniforme sobre [0, modulus).
//
//  Ground truth:
//  Búsqueda exacta filtrada del propio index_gt.
//
//  RACORN:
//  - two_hop_threshold = 0.30.
//  - bridge_budget_ratio = 1.0.
//  - stride sampling = true.
//  - 3-hop = desactivado.
//  - exact fallback = desactivado.
//
//  Estadística:
//  La desviación estándar reportada entre realizaciones utiliza la
//  definición poblacional sobre el conjunto determinista de residuos
//  evaluados en este Gate.
//
//  Invariantes:
//  - El índice se construye una sola vez.
//  - Las consultas se generan una sola vez.
//  - No se modifica la topología entre realizaciones.
//  - No se modifica RACORN upstream.
//  - No se mide latencia en este Gate.
//  - No se mide memoria en este Gate.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Sin estado persistente.
//

#include "filtered/FilteredSearchBaseline.hpp"

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <algorithm>
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

constexpr double kTwoHopThreshold = 0.30;
constexpr double kBridgeBudgetRatio = 1.0;

constexpr std::uint64_t kBaseSeed =
    0x13579BDF2468ACE0ULL;

constexpr std::uint64_t kQuerySeed =
    0xCAFEBABE12345678ULL;

using VectorKey = std::size_t;
using Slot = std::size_t;

using Index =
    index_gt<float, VectorKey, Slot>;

//
// Generador determinista SplitMix64.
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

//
// Almacén de vectores.
//

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

//
// Almacén de consultas.
//

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

//
// Métrica L2 al cuadrado.
//

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

//
// Wrapper RAII alineado para index_gt.
//

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

//
// Predicado desplazable.
//
// El campo residue define el inicio de la ventana residual.
//
// Para threshold = 1:
//
//      accepts_key(key)
//
// equivale a:
//
//      key % modulus == residue
//

struct MetadataPredicate final {

    std::size_t modulus{};

    std::size_t threshold{};

    std::size_t residue{};

    [[nodiscard]]
    bool accepts_key(
        std::size_t key
    ) const noexcept {

        const std::size_t key_residue =
            key % modulus;

        const std::size_t start =
            residue % modulus;

        const std::size_t shifted =
            key_residue >= start
                ? key_residue - start
                : modulus -
                    (start - key_residue);

        return
            shifted <
            threshold;
    }

    template <typename MemberLike>
    bool operator()(
        MemberLike&& member
    ) const noexcept {

        return
            accepts_key(
                static_cast<std::size_t>(
                    get_key(member)
                )
            );
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
        static_cast<double>(hits) /
        static_cast<double>(
            exact.size()
        );
}

bool filter_is_valid(
    std::vector<std::size_t> const& keys,
    MetadataPredicate const& predicate
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

    std::size_t
        fallback_selectivity_samples{};

    std::size_t
        queries_with_fallback{};

    std::size_t
        samples{};

    bool filter_valid =
        true;
};

void accumulate(
    Accumulator& accumulator,
    std::vector<std::size_t> const& keys,
    std::vector<std::size_t> const& ground_truth,
    Index::search_result_t const& result,
    MetadataPredicate const& predicate
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
        result.fallback_count >
        0
    ) {

        ++accumulator
            .queries_with_fallback;

        if (
            std::isfinite(
                result.fallback_selectivity
            )
        ) {

            accumulator
                .fallback_selectivity_sum +=
                result
                    .fallback_selectivity;

            ++accumulator
                .fallback_selectivity_samples;
        }
    }

    accumulator.filter_valid =
        accumulator.filter_valid &&
        filter_is_valid(
            keys,
            predicate
        );

    ++accumulator.samples;
}

double accumulator_mean_recall(
    Accumulator const& accumulator
) noexcept {

    if (
        accumulator.samples ==
        0
    ) {

        return 0.0;
    }

    return
        accumulator.recall_sum /
        static_cast<double>(
            accumulator.samples
        );
}

double population_stddev(
    std::vector<double> const& values
) {

    if (
        values.empty()
    ) {

        return 0.0;
    }

    double mean =
        0.0;

    for (
        double value :
        values
    ) {

        mean +=
            value;
    }

    mean /=
        static_cast<double>(
            values.size()
        );

    double squared_error_sum =
        0.0;

    for (
        double value :
        values
    ) {

        const double error =
            value -
            mean;

        squared_error_sum +=
            error *
            error;
    }

    return
        std::sqrt(
            squared_error_sum /
            static_cast<double>(
                values.size()
            )
        );
}

FilteredModeStats finish_stats(
    char const* name,
    Accumulator const& accumulator,
    std::vector<double> const&
        realization_recalls
) {

    FilteredModeStats stats;

    stats.name =
        name;

    if (
        accumulator.samples >
        0
    ) {

        const double samples =
            static_cast<double>(
                accumulator.samples
            );

        stats.mean_recall_at_k =
            accumulator.recall_sum /
            samples;

        stats.mean_result_count =
            accumulator
                .result_count_sum /
            samples;

        stats.mean_computed_distances =
            accumulator
                .computed_distances_sum /
            samples;

        stats.mean_visited_members =
            accumulator
                .visited_members_sum /
            samples;

        stats.mean_predicate_evaluations =
            accumulator
                .predicate_evaluations_sum /
            samples;

        stats.mean_fallback_count =
            accumulator
                .fallback_count_sum /
            samples;
    }

    stats.total_query_evaluations =
        accumulator.samples;

    stats.queries_with_fallback =
        accumulator
            .queries_with_fallback;

    if (
        accumulator.samples >
        0
    ) {

        stats.fallback_query_fraction =
            static_cast<double>(
                accumulator
                    .queries_with_fallback
            ) /
            static_cast<double>(
                accumulator.samples
            );
    }

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

    stats.realization_recall_stddev =
        population_stddev(
            realization_recalls
        );

    if (
        !realization_recalls.empty()
    ) {

        const auto minmax =
            std::minmax_element(
                realization_recalls.begin(),
                realization_recalls.end()
            );

        stats.min_realization_recall =
            *minmax.first;

        stats.max_realization_recall =
            *minmax.second;
    }

    stats.all_results_satisfy_filter =
        accumulator.filter_valid;

    return stats;
}

std::size_t count_eligible(
    MetadataPredicate const& predicate
) noexcept {

    std::size_t count =
        0;

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

            ++count;
        }
    }

    return count;
}

std::size_t distributed_residue(
    std::size_t realization_index,
    std::size_t realization_count,
    std::size_t modulus
) noexcept {

    if (
        realization_count <= 1
    ) {

        return 0;
    }

    return
        (
            realization_index *
            modulus
        ) /
        realization_count;
}

} // namespace

FilteredSearchResult
FilteredSearchBaseline::run(
    std::size_t filter_modulus,
    std::size_t filter_threshold,
    std::size_t requested_realization_count
) {

    FilteredSearchResult output;

    if (
        filter_modulus == 0 ||
        filter_threshold == 0 ||
        filter_threshold >
            filter_modulus ||
        requested_realization_count ==
            0
    ) {

        return output;
    }

    const std::size_t realization_count =
        std::min(
            requested_realization_count,
            filter_modulus
        );

    VectorStore base(
        kVectorCount,
        kDimensions,
        kBaseSeed
    );

    QueryStore queries(
        kQueryCount,
        kDimensions,
        kQuerySeed
    );

    L2Metric metric{
        .store = &base,
        .dimensions = kDimensions
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

    Accumulator
        global_hnsw;

    Accumulator
        global_acorn;

    Accumulator
        global_racorn;

    std::vector<double>
        hnsw_realization_recalls;

    std::vector<double>
        acorn_realization_recalls;

    std::vector<double>
        racorn_realization_recalls;

    hnsw_realization_recalls.reserve(
        realization_count
    );

    acorn_realization_recalls.reserve(
        realization_count
    );

    racorn_realization_recalls.reserve(
        realization_count
    );

    output.realizations.reserve(
        realization_count
    );

    bool exact_complete =
        true;

    std::size_t min_eligible =
        std::numeric_limits<
            std::size_t
        >::max();

    std::size_t max_eligible =
        0;

    for (
        std::size_t realization_index = 0;
        realization_index <
            realization_count;
        ++realization_index
    ) {

        const std::size_t residue =
            distributed_residue(
                realization_index,
                realization_count,
                filter_modulus
            );

        const MetadataPredicate predicate{
            .modulus =
                filter_modulus,

            .threshold =
                filter_threshold,

            .residue =
                residue
        };

        const std::size_t eligible_count =
            count_eligible(
                predicate
            );

        min_eligible =
            std::min(
                min_eligible,
                eligible_count
            );

        max_eligible =
            std::max(
                max_eligible,
                eligible_count
            );

        if (
            eligible_count <
            kNeighbors
        ) {

            exact_complete =
                false;
        }

        Accumulator
            local_hnsw;

        Accumulator
            local_acorn;

        Accumulator
            local_racorn;

        for (
            std::size_t query_index = 0;
            query_index <
                kQueryCount;
            ++query_index
        ) {

            float const* query =
                queries.row(
                    query_index
                );

            //
            // Ground truth exacto.
            //

            index_search_config_t
                exact_config;

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
                    exact_keys,
                    predicate
                )
            ) {

                exact_complete =
                    false;
            }

            //
            // HNSW filtrado.
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
                local_hnsw,
                hnsw_keys,
                exact_keys,
                hnsw_result,
                predicate
            );

            accumulate(
                global_hnsw,
                hnsw_keys,
                exact_keys,
                hnsw_result,
                predicate
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
                local_acorn,
                acorn_keys,
                exact_keys,
                acorn_result,
                predicate
            );

            accumulate(
                global_acorn,
                acorn_keys,
                exact_keys,
                acorn_result,
                predicate
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
                local_racorn,
                racorn_keys,
                exact_keys,
                racorn_result,
                predicate
            );

            accumulate(
                global_racorn,
                racorn_keys,
                exact_keys,
                racorn_result,
                predicate
            );
        }

        const double hnsw_recall =
            accumulator_mean_recall(
                local_hnsw
            );

        const double acorn_recall =
            accumulator_mean_recall(
                local_acorn
            );

        const double racorn_recall =
            accumulator_mean_recall(
                local_racorn
            );

        hnsw_realization_recalls.push_back(
            hnsw_recall
        );

        acorn_realization_recalls.push_back(
            acorn_recall
        );

        racorn_realization_recalls.push_back(
            racorn_recall
        );

        FilterRealizationStats
            realization_stats;

        realization_stats.realization_index =
            realization_index;

        realization_stats.residue =
            residue;

        realization_stats.eligible_count =
            eligible_count;

        realization_stats.observed_selectivity =
            static_cast<double>(
                eligible_count
            ) /
            static_cast<double>(
                kVectorCount
            );

        realization_stats.hnsw_recall_at_k =
            hnsw_recall;

        realization_stats.acorn_recall_at_k =
            acorn_recall;

        realization_stats.racorn_recall_at_k =
            racorn_recall;

        if (
            local_racorn.samples >
            0
        ) {

            realization_stats.racorn_mean_distances =
                local_racorn
                    .computed_distances_sum /
                static_cast<double>(
                    local_racorn.samples
                );

            realization_stats
                .racorn_mean_fallback_count =
                local_racorn
                    .fallback_count_sum /
                static_cast<double>(
                    local_racorn.samples
                );

            realization_stats
                .racorn_fallback_query_fraction =
                static_cast<double>(
                    local_racorn
                        .queries_with_fallback
                ) /
                static_cast<double>(
                    local_racorn.samples
                );
        }

        output.realizations.push_back(
            realization_stats
        );
    }

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

    output.filter_modulus =
        filter_modulus;

    output.filter_threshold =
        filter_threshold;

    output.realization_count =
        realization_count;

    output.min_eligible_count =
        min_eligible;

    output.max_eligible_count =
        max_eligible;

    output.nominal_selectivity =
        static_cast<double>(
            filter_threshold
        ) /
        static_cast<double>(
            filter_modulus
        );

    output.hnsw =
        finish_stats(
            "HNSW",
            global_hnsw,
            hnsw_realization_recalls
        );

    output.acorn =
        finish_stats(
            "ACORN-1",
            global_acorn,
            acorn_realization_recalls
        );

    output.racorn =
        finish_stats(
            "RACORN-1",
            global_racorn,
            racorn_realization_recalls
        );

    output.exact_ground_truth_complete =
        exact_complete;

    output.filters_valid =
        output.hnsw
            .all_results_satisfy_filter &&
        output.acorn
            .all_results_satisfy_filter &&
        output.racorn
            .all_results_satisfy_filter;

    output.racorn_fallback_observed =
        output.racorn
            .queries_with_fallback >
        0;

    output.passed =
        output.vector_count ==
            kVectorCount &&
        output.realization_count >
            0 &&
        output.min_eligible_count >=
            kNeighbors &&
        output.exact_ground_truth_complete &&
        output.filters_valid;

    return output;
}

} // namespace racorn_uma::filtered

//
//  RacornSiftExactFallback.cpp
//  RACORNUMALab
//
//  Propósito:
//  Implementar G-DATA2.
//
//  Comparación:
//
//      RACORN-1
//          fallback_exact_threshold = 0
//
//      RACORN-1+
//          fallback_exact_threshold > 0
//
//  Dataset:
//
//      SIFT1M-base[0:100000]
//      N = 100000
//      D = 128
//      Q = 1000
//      L2
//
//  Configuración fija:
//
//      M   = 16
//      efC = 100
//      efS = 200
//      k   = 10
//      TT  = 0.30
//
//  Matriz:
//
//      BR = {
//          0.75,
//          1.00,
//          1.25
//      }
//
//      exact_threshold = {
//          0.000,
//          0.001,
//          0.002,
//          0.003,
//          0.005
//      }
//
//  Criterio robusto:
//
//      mean Recall@10 >= 0.99
//
//      AND
//
//      min Recall@10 >= 0.99
//
//  Nota:
//  Con k=10, min Recall@10 >= 0.99 implica que todas las queries deben
//  obtener Recall@10 = 1.0.
//
//  PASS experimental:
//  Significa que todos los casos fueron ejecutados con ground truth
//  completo, cardinalidad válida y resultados que cumplen el filtro.
//
//  PASS experimental NO implica que exista una configuración que cumpla
//  la garantía robusta.
//

#include "core/RacornSiftExactFallback.hpp"

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <algorithm>
#include <array>
#include <bit>
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

namespace racorn_uma::core {

namespace {

using namespace unum::usearch;

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

constexpr std::size_t kExperimentalQueryCount =
    1'000U;

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

constexpr double kTwoHopThreshold =
    0.30;

constexpr double kMeanRecallTarget =
    0.99;

constexpr double kMinRecallTarget =
    0.99;

constexpr std::array<double, 3>
    kBridgeRatios{
        0.75,
        1.00,
        1.25
    };

constexpr std::array<double, 5>
    kExactThresholds{
        0.000,
        0.001,
        0.002,
        0.003,
        0.005
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

struct RecallMeasurement final {

    double recall{};
    std::size_t hits{};
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

struct LazyPredicate final {

    MetadataView metadata{};
    FilterSpec filter{};

    [[nodiscard]]
    bool accepts_key(
        std::size_t key
    ) const noexcept {

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
                2U &&

            category_allowed &&

            metadata.security_level[key] <=
                2U;
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
            "G-DATA2 requiere arquitectura little-endian."
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
            "Archivo ausente o invalido: " +
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
            "Tamaño incorrecto para " +
            path.filename().string() +
            ": " +
            std::to_string(
                actual
            ) +
            " != " +
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
            "Lectura incompleta de: " +
            path.string()
        );
    }

    return output;
}

std::vector<float> read_query_prefix(
    std::filesystem::path const& path
) {

    const std::size_t elements =
        kExperimentalQueryCount *
        kDimensions;

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

    std::vector<float> output(
        elements
    );

    const std::size_t bytes =
        elements *
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

FilterSpec filter_for_query(
    std::size_t query_index
) noexcept {

    return FilterSpec{
        .tenant_required =
            static_cast<std::uint32_t>(
                query_index %
                kTenantCount
            )
    };
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
    std::vector<std::size_t> const& eligible_keys
) {

    if (
        eligible_keys.size() <
        kNeighbors
    ) {

        return {};
    }

    std::vector<ExactCandidate>
        candidates;

    candidates.reserve(
        eligible_keys.size()
    );

    for (
        std::size_t key :
        eligible_keys
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
            candidates[index].key
        );
    }

    return result;
}

RecallMeasurement measure_recall(
    Index::search_result_t const& approximate,
    std::vector<std::size_t> const& exact
) {

    if (
        exact.empty()
    ) {

        return {};
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

    return RecallMeasurement{
        .recall =
            static_cast<double>(
                hits
            ) /
            static_cast<double>(
                exact.size()
            ),

        .hits =
            hits
    };
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

index_search_config_t make_search_configuration(
    double bridge_budget_ratio,
    double fallback_exact_threshold
) {

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
        bridge_budget_ratio;

    configuration.three_hop_threshold =
        -1.0;

    configuration.fallback_exact_threshold =
        fallback_exact_threshold;

    configuration.racorn_stride_sampling =
        true;

    return configuration;
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

RacornSiftExactFallbackResult
RacornSiftExactFallback::run(
    std::filesystem::path const& data_directory
) {

    ensure_little_endian();

    RacornSiftExactFallbackResult
        output;

    output.data_directory =
        data_directory;

    output.vector_count =
        kVectorCount;

    output.dimensions =
        kDimensions;

    output.query_count =
        kExperimentalQueryCount;

    output.connectivity =
        kConnectivity;

    output.ef_construction =
        kEfConstruction;

    output.ef_search =
        kEfSearch;

    output.k =
        kNeighbors;

    output.two_hop_threshold =
        kTwoHopThreshold;

    output.mean_recall_target =
        kMeanRecallTarget;

    output.min_recall_target =
        kMinRecallTarget;

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
        "[G-DATA2] Validando archivos..."
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
        "[G-DATA2] Cargando SIFT100K..."
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
        "[G-DATA2] Cargando 1000 queries..."
    );

    DenseVectors queries(
        read_query_prefix(
            query_path
        ),
        kDimensions
    );

    print_progress(
        "[G-DATA2] Cargando metadata..."
    );

    auto metadata_storage =
        read_binary_exact<std::uint32_t>(
            metadata_path,
            kMetadataColumnCount *
                kVectorCount
        );

    MetadataView metadata{
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

    //
    // Precomputar candidatos elegibles.
    //

    print_progress(
        "[G-DATA2] Precomputando conjuntos filtrados..."
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
                        )
                }
        };

        auto& keys =
            eligible_by_tenant[
                tenant
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

                keys.push_back(
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
        query_index < kExperimentalQueryCount;
        ++query_index
    ) {

        const auto& keys =
            eligible_by_tenant[
                query_index %
                kTenantCount
            ];

        output.selected_min =
            std::min(
                output.selected_min,
                keys.size()
            );

        output.selected_max =
            std::max(
                output.selected_max,
                keys.size()
            );

        selected_sum +=
            keys.size();
    }

    output.mean_selectivity =
        static_cast<double>(
            selected_sum
        ) /
        (
            static_cast<double>(
                kExperimentalQueryCount
            ) *
            static_cast<double>(
                kVectorCount
            )
        );

    //
    // HNSW.
    //

    L2Metric metric{
        .vectors =
            &base,

        .dimensions =
            kDimensions
    };

    print_progress(
        "[G-DATA2] Construyendo HNSW..."
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

    constexpr std::size_t
        progress_step =
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
                << "[G-DATA2] HNSW "
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
    // Ground truth.
    //

    print_progress(
        "[G-DATA2] Construyendo ground truth exacto..."
    );

    std::vector<
        std::vector<std::size_t>
    >
        exact_by_query;

    exact_by_query.reserve(
        kExperimentalQueryCount
    );

    bool ground_truth_complete =
        true;

    for (
        std::size_t query_index = 0;
        query_index < kExperimentalQueryCount;
        ++query_index
    ) {

        const std::size_t tenant =
            query_index %
            kTenantCount;

        auto exact =
            exact_filtered_top_k(
                base,
                queries.row(
                    query_index
                ),
                eligible_by_tenant[
                    tenant
                ]
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

        if (
            (
                query_index +
                    1U
            ) %
                100U ==
                0U
        ) {

            std::cout
                << "[G-DATA2] GT "
                << (
                    query_index +
                        1U
                )
                << "/"
                << kExperimentalQueryCount
                << '\n'
                << std::flush;
        }
    }

    output.ground_truth_complete =
        ground_truth_complete;

    //
    // Matriz BR × threshold.
    //

    bool all_cases_valid =
        true;

    bool robust_configuration_found =
        false;

    for (
        double bridge_ratio :
        kBridgeRatios
    ) {

        for (
            double exact_threshold :
            kExactThresholds
        ) {

            std::cout
                << "\n[G-DATA2] BR="
                << bridge_ratio
                << " threshold="
                << exact_threshold
                << "...\n"
                << std::flush;

            SiftExactFallbackCase
                case_result;

            case_result.bridge_budget_ratio =
                bridge_ratio;

            case_result.fallback_exact_threshold =
                exact_threshold;

            case_result.racorn_one_plus =
                exact_threshold >
                0.0;

            case_result.result_count_min =
                std::numeric_limits<
                    std::size_t
                >::max();

            std::vector<double>
                recalls;

            recalls.reserve(
                kExperimentalQueryCount
            );

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

            double fallback_count_sum =
                0.0;

            double fallback_selectivity_sum =
                0.0;

            std::size_t fallback_selectivity_samples =
                0U;

            bool filter_valid =
                true;

            bool cardinality_valid =
                true;

            for (
                std::size_t query_index = 0;
                query_index < kExperimentalQueryCount;
                ++query_index
            ) {

                const FilterSpec filter =
                    filter_for_query(
                        query_index
                    );

                const LazyPredicate predicate{
                    .metadata =
                        metadata,

                    .filter =
                        filter
                };

                auto configuration =
                    make_search_configuration(
                        bridge_ratio,
                        exact_threshold
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

                const RecallMeasurement measured =
                    measure_recall(
                        result,
                        exact_by_query[
                            query_index
                        ]
                    );

                recalls.push_back(
                    measured.recall
                );

                recall_sum +=
                    measured.recall;

                minimum_recall =
                    std::min(
                        minimum_recall,
                        measured.recall
                    );

                const std::size_t result_count =
                    result.size();

                case_result.result_count_min =
                    std::min(
                        case_result.result_count_min,
                        result_count
                    );

                case_result.result_count_max =
                    std::max(
                        case_result.result_count_max,
                        result_count
                    );

                if (
                    result_count >
                    kNeighbors
                ) {

                    cardinality_valid =
                        false;
                }

                const bool perfect =
                    result_count ==
                        kNeighbors &&
                    measured.hits ==
                        kNeighbors;

                const bool underfilled =
                    result_count <
                    kNeighbors;

                const bool full_but_inexact =
                    result_count ==
                        kNeighbors &&
                    measured.hits <
                        kNeighbors;

                if (
                    perfect
                ) {

                    ++case_result
                        .perfect_queries;

                } else {

                    ++case_result
                        .imperfect_queries;
                }

                if (
                    underfilled
                ) {

                    ++case_result
                        .underfilled_queries;
                }

                if (
                    full_but_inexact
                ) {

                    ++case_result
                        .full_but_inexact_queries;
                }

                if (
                    result.fallback_to_exact
                ) {

                    ++case_result
                        .exact_fallback_queries;
                }

                if (
                    !std::isnan(
                        result.fallback_selectivity
                    )
                ) {

                    fallback_selectivity_sum +=
                        result.fallback_selectivity;

                    ++fallback_selectivity_samples;
                }

                filter_valid =
                    filter_valid &&
                    results_respect_filter(
                        result,
                        predicate
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

                fallback_count_sum +=
                    static_cast<double>(
                        result
                            .fallback_count
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
                        << "[G-DATA2] BR="
                        << bridge_ratio
                        << " threshold="
                        << exact_threshold
                        << " "
                        << (
                            query_index +
                                1U
                        )
                        << "/"
                        << kExperimentalQueryCount
                        << '\n'
                        << std::flush;
                }
            }

            const double query_count =
                static_cast<double>(
                    kExperimentalQueryCount
                );

            case_result.mean_recall_at_k =
                recall_sum /
                query_count;

            case_result.min_recall_at_k =
                minimum_recall;

            case_result.recall_p01 =
                percentile(
                    recalls,
                    0.01
                );

            case_result.recall_p05 =
                percentile(
                    recalls,
                    0.05
                );

            case_result.exact_fallback_fraction =
                static_cast<double>(
                    case_result
                        .exact_fallback_queries
                ) /
                query_count;

            case_result.fallback_selectivity_samples =
                fallback_selectivity_samples;

            if (
                fallback_selectivity_samples >
                0U
            ) {

                case_result.mean_fallback_selectivity =
                    fallback_selectivity_sum /
                    static_cast<double>(
                        fallback_selectivity_samples
                    );
            }

            case_result.mean_computed_distances =
                distances_sum /
                query_count;

            case_result.mean_visited_members =
                visited_sum /
                query_count;

            case_result.mean_predicate_evaluations =
                predicate_sum /
                query_count;

            case_result.mean_fallback_count =
                fallback_count_sum /
                query_count;

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

            case_result.ground_truth_complete =
                ground_truth_complete;

            case_result.returned_results_filter_valid =
                filter_valid;

            case_result.cardinality_valid =
                cardinality_valid;

            case_result.valid =
                case_result.ground_truth_complete &&
                case_result.returned_results_filter_valid &&
                case_result.cardinality_valid;

            all_cases_valid =
                all_cases_valid &&
                case_result.valid;

            robust_configuration_found =
                robust_configuration_found ||
                (
                    case_result.valid &&
                    case_result.robust_quality_pass
                );

            output.cases.push_back(
                case_result
            );
        }
    }

    output.all_cases_valid =
        all_cases_valid;

    output.robust_configuration_found =
        robust_configuration_found;

    output.valid =
        output.base_file_valid &&
        output.query_file_valid &&
        output.metadata_file_valid &&
        output.ground_truth_complete &&
        output.all_cases_valid &&
        !output.racorn_modified;

    print_progress(
        "\n[G-DATA2] Matriz completada."
    );

    return output;
}

} // namespace racorn_uma::core

//
//  RacornSiftTargetingAudit.cpp
//  RACORNUMALab
//
//  Propósito:
//  Implementar G-DATA2A.
//
//  Este Gate trata Adaptive Exact Fallback como un detector binario:
//
//      hard query  -> debería activar exact.
//      easy query  -> debería permanecer en RACORN.
//
//  Baseline:
//      BR  = 1.0
//      tau = 0
//
//  Sweep:
//      0.0010 ... 0.0020
//
//  Dataset:
//      SIFT1M-base[0:100000]
//      N = 100000
//      D = 128
//      Q = 1000
//
//  Configuración:
//      M   = 16
//      efC = 100
//      efS = 200
//      k   = 10
//      TT  = 0.30
//
//  Importante:
//  `fallback_selectivity` almacenada para el baseline corresponde a la
//  selectividad final observada durante RACORN-1. Es una señal diagnóstica.
//  El trigger RACORN-1+ puede ocurrir antes, durante la acumulación del
//  estado del fallback.
//
//  Invariantes:
//  - RACORN upstream intacto.
//  - Navegación CPU.
//  - Sin Metal.
//  - Sin GPU.
//  - Sin modelo aprendido.
//  - Ground truth exacto filtrado.
//

#include "core/RacornSiftTargetingAudit.hpp"

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

constexpr double kBridgeBudgetRatio =
    1.0;

constexpr double kTwoHopThreshold =
    0.30;

constexpr double kMeanRecallTarget =
    0.99;

constexpr double kMinRecallTarget =
    0.99;

constexpr double kPerfectTolerance =
    1e-12;

constexpr std::array<double, 11>
    kThresholds{
        0.0010,
        0.0011,
        0.0012,
        0.0013,
        0.0014,
        0.0015,
        0.0016,
        0.0017,
        0.0018,
        0.0019,
        0.0020
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

struct BaselineQuery final {

    bool hard{};

    double recall{};

    std::size_t result_count{};

    double fallback_selectivity{};

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
            "G-DATA2A requiere arquitectura little-endian."
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

    const std::size_t element_count =
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
            "No se pudo abrir queries: " +
            path.string()
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
            "Lectura incompleta del prefijo de queries."
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

    std::size_t hits =
        0U;

    for (
        std::size_t approximate_index = 0;
        approximate_index < approximate.size();
        ++approximate_index
    ) {

        const std::size_t approximate_key =
            static_cast<std::size_t>(
                approximate[
                    approximate_index
                ]
                    .member
                    .key
            );

        if (
            std::find(
                exact.begin(),
                exact.end(),
                approximate_key
            ) !=
            exact.end()
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
                result[
                    index
                ]
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

TargetingDistributionStats distribution_stats(
    std::vector<double> const& input
) {

    TargetingDistributionStats
        output;

    output.samples =
        input.size();

    if (
        input.empty()
    ) {

        return output;
    }

    std::vector<double>
        values =
            input;

    std::sort(
        values.begin(),
        values.end()
    );

    double sum =
        0.0;

    for (
        double value :
        values
    ) {

        sum +=
            value;
    }

    output.minimum =
        values.front();

    output.maximum =
        values.back();

    output.mean =
        sum /
        static_cast<double>(
            values.size()
        );

    output.p05 =
        percentile(
            values,
            0.05
        );

    output.median =
        percentile(
            values,
            0.50
        );

    output.p95 =
        percentile(
            values,
            0.95
        );

    return output;
}

index_search_config_t
make_search_configuration(
    double exact_threshold
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
        kBridgeBudgetRatio;

    configuration.three_hop_threshold =
        -1.0;

    configuration.fallback_exact_threshold =
        exact_threshold;

    configuration.racorn_stride_sampling =
        true;

    return configuration;
}

bool is_perfect(
    std::size_t result_count,
    std::size_t hits
) noexcept {

    return
        result_count ==
            kNeighbors &&
        hits ==
            kNeighbors;
}

double safe_ratio(
    std::size_t numerator,
    std::size_t denominator
) noexcept {

    if (
        denominator ==
        0U
    ) {

        return 0.0;
    }

    return
        static_cast<double>(
            numerator
        ) /
        static_cast<double>(
            denominator
        );
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

RacornSiftTargetingAuditResult
RacornSiftTargetingAudit::run(
    std::filesystem::path const& data_directory
) {

    ensure_little_endian();

    RacornSiftTargetingAuditResult
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

    output.bridge_budget_ratio =
        kBridgeBudgetRatio;

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

    //
    // Archivos.
    //

    print_progress(
        "[G-DATA2A] Validando archivos..."
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

    //
    // Datos.
    //

    print_progress(
        "[G-DATA2A] Cargando SIFT100K..."
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
        "[G-DATA2A] Cargando 1000 queries..."
    );

    DenseVectors queries(
        read_query_prefix(
            query_path
        ),
        kDimensions
    );

    print_progress(
        "[G-DATA2A] Cargando metadata..."
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
    // Conjuntos elegibles.
    //

    print_progress(
        "[G-DATA2A] Precomputando conjuntos filtrados..."
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
                kExperimentalQueryCount
            ) *
            static_cast<double>(
                kVectorCount
            )
        );

    //
    // Índice.
    //

    L2Metric metric{
        .vectors =
            &base,

        .dimensions =
            kDimensions
    };

    print_progress(
        "[G-DATA2A] Construyendo HNSW..."
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
                << "[G-DATA2A] HNSW "
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
        "[G-DATA2A] Construyendo ground truth exacto..."
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
                << "[G-DATA2A] GT "
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
    // RACORN-1 baseline.
    //

    print_progress(
        "[G-DATA2A] Ejecutando baseline RACORN-1..."
    );

    std::vector<BaselineQuery>
        baseline;

    baseline.reserve(
        kExperimentalQueryCount
    );

    std::vector<double>
        hard_fallback_selectivities;

    std::vector<double>
        easy_fallback_selectivities;

    double baseline_recall_sum =
        0.0;

    double baseline_min_recall =
        1.0;

    bool baseline_filter_valid =
        true;

    bool baseline_cardinality_valid =
        true;

    for (
        std::size_t query_index = 0;
        query_index < kExperimentalQueryCount;
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

        auto configuration =
            make_search_configuration(
                0.0
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

        const bool perfect =
            is_perfect(
                result.size(),
                measured.hits
            );

        const bool hard =
            !perfect;

        baseline_recall_sum +=
            measured.recall;

        baseline_min_recall =
            std::min(
                baseline_min_recall,
                measured.recall
            );

        if (
            perfect
        ) {

            ++output
                .baseline_perfect_queries;

        } else {

            ++output
                .baseline_hard_queries;

            output
                .baseline_hard_query_ids
                .push_back(
                    query_index
                );
        }

        if (
            !std::isnan(
                result.fallback_selectivity
            )
        ) {

            if (
                hard
            ) {

                hard_fallback_selectivities
                    .push_back(
                        result
                            .fallback_selectivity
                    );

            } else {

                easy_fallback_selectivities
                    .push_back(
                        result
                            .fallback_selectivity
                    );
            }
        }

        baseline_filter_valid =
            baseline_filter_valid &&
            results_respect_filter(
                result,
                predicate
            );

        if (
            result.size() >
            kNeighbors
        ) {

            baseline_cardinality_valid =
                false;
        }

        baseline.push_back(
            BaselineQuery{
                .hard =
                    hard,

                .recall =
                    measured.recall,

                .result_count =
                    result.size(),

                .fallback_selectivity =
                    result
                        .fallback_selectivity,

                .computed_distances =
                    result
                        .computed_distances,

                .visited_members =
                    result
                        .visited_members,

                .predicate_evaluations =
                    result
                        .predicate_evaluations,

                .fallback_count =
                    result
                        .fallback_count
            }
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
                << "[G-DATA2A] Baseline "
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

    output.baseline_mean_recall =
        baseline_recall_sum /
        static_cast<double>(
            kExperimentalQueryCount
        );

    output.baseline_min_recall =
        baseline_min_recall;

    output.baseline_hard_fallback_selectivity =
        distribution_stats(
            hard_fallback_selectivities
        );

    output.baseline_easy_fallback_selectivity =
        distribution_stats(
            easy_fallback_selectivities
        );

    output.baseline_valid =
        ground_truth_complete &&
        baseline_filter_valid &&
        baseline_cardinality_valid &&
        baseline.size() ==
            kExperimentalQueryCount;

    //
    // Threshold sweep.
    //

    bool all_cases_valid =
        true;

    for (
        double threshold :
        kThresholds
    ) {

        std::cout
            << "\n[G-DATA2A] tau="
            << threshold
            << "...\n"
            << std::flush;

        ExactFallbackTargetingCase
            case_result;

        case_result.threshold =
            threshold;

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

        bool filter_valid =
            true;

        bool cardinality_valid =
            true;

        for (
            std::size_t query_index = 0;
            query_index < kExperimentalQueryCount;
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

            auto configuration =
                make_search_configuration(
                    threshold
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

            const bool predicted_hard =
                result.fallback_to_exact;

            const bool actually_hard =
                baseline[
                    query_index
                ].hard;

            if (
                actually_hard &&
                predicted_hard
            ) {

                ++case_result
                    .true_positive;

            } else if (
                !actually_hard &&
                predicted_hard
            ) {

                ++case_result
                    .false_positive;

                case_result
                    .false_positive_query_ids
                    .push_back(
                        query_index
                    );

            } else if (
                actually_hard &&
                !predicted_hard
            ) {

                ++case_result
                    .false_negative;

                case_result
                    .missed_hard_query_ids
                    .push_back(
                        query_index
                    );

            } else {

                ++case_result
                    .true_negative;
            }

            if (
                predicted_hard
            ) {

                ++case_result
                    .exact_fallback_queries;
            }

            const bool final_perfect =
                is_perfect(
                    result.size(),
                    measured.hits
                );

            if (
                final_perfect
            ) {

                ++case_result
                    .perfect_queries;

            } else {

                ++case_result
                    .imperfect_queries;
            }

            if (
                result.size() <
                kNeighbors
            ) {

                ++case_result
                    .underfilled_queries;
            }

            if (
                actually_hard &&
                final_perfect
            ) {

                ++case_result
                    .rescued_hard_queries;
            }

            if (
                actually_hard &&
                !final_perfect
            ) {

                ++case_result
                    .remaining_hard_queries;
            }

            if (
                !actually_hard &&
                !final_perfect
            ) {

                ++case_result
                    .newly_imperfect_easy_queries;
            }

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
                !std::isnan(
                    result
                        .fallback_selectivity
                )
            ) {

                fallback_selectivity_sum +=
                    result
                        .fallback_selectivity;

                ++case_result
                    .fallback_selectivity_samples;
            }

            filter_valid =
                filter_valid &&
                results_respect_filter(
                    result,
                    predicate
                );

            if (
                result.size() >
                kNeighbors
            ) {

                cardinality_valid =
                    false;
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
                    << "[G-DATA2A] tau="
                    << threshold
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

        case_result.precision =
            safe_ratio(
                case_result.true_positive,
                case_result.true_positive +
                    case_result.false_positive
            );

        case_result.coverage =
            safe_ratio(
                case_result.true_positive,
                case_result.true_positive +
                    case_result.false_negative
            );

        case_result.specificity =
            safe_ratio(
                case_result.true_negative,
                case_result.true_negative +
                    case_result.false_positive
            );

        if (
            case_result.precision +
                case_result.coverage >
            0.0
        ) {

            case_result.f1 =
                2.0 *
                case_result.precision *
                case_result.coverage /
                (
                    case_result.precision +
                    case_result.coverage
                );
        }

        case_result.balanced_accuracy =
            0.5 *
            (
                case_result.coverage +
                case_result.specificity
            );

        case_result.exact_fallback_fraction =
            static_cast<double>(
                case_result
                    .exact_fallback_queries
            ) /
            query_count;

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

        if (
            case_result
                .fallback_selectivity_samples >
            0U
        ) {

            case_result.mean_fallback_selectivity =
                fallback_selectivity_sum /
                static_cast<double>(
                    case_result
                        .fallback_selectivity_samples
                );
        }

        case_result.robust_quality_pass =
            case_result.mean_recall_at_k +
                kPerfectTolerance >=
                kMeanRecallTarget &&
            case_result.min_recall_at_k +
                kPerfectTolerance >=
                kMinRecallTarget;

        case_result.returned_results_filter_valid =
            filter_valid;

        case_result.cardinality_valid =
            cardinality_valid;

        case_result.valid =
            ground_truth_complete &&
            filter_valid &&
            cardinality_valid;

        all_cases_valid =
            all_cases_valid &&
            case_result.valid;

        if (
            case_result.valid &&
            (
                !output.best_f1_found ||
                case_result.f1 >
                    output.best_f1
            )
        ) {

            output.best_f1_found =
                true;

            output.best_f1 =
                case_result.f1;

            output.best_f1_threshold =
                threshold;
        }

        output.cases.push_back(
            std::move(
                case_result
            )
        );
    }

    output.all_cases_valid =
        all_cases_valid;

    output.valid =
        output.base_file_valid &&
        output.query_file_valid &&
        output.metadata_file_valid &&
        output.ground_truth_complete &&
        output.baseline_valid &&
        output.all_cases_valid &&
        !output.racorn_modified;

    print_progress(
        "\n[G-DATA2A] Auditoria completada."
    );

    return output;
}

} // namespace racorn_uma::core

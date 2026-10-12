//
//  HnswBaseline.cpp
//  RACORNUMALab
//
//  Propósito:
//  Construir un HNSW parametrizable para medir cómo escala su memoria
//  en función del número de vectores.
//
//  Gate:
//  G3C — Escalado controlado de memoria.
//
//  Diseño:
//  Cada tamaño N debe ejecutarse en un proceso independiente para evitar
//  contaminación entre experimentos por cachés y retención del allocator.
//
//  Dataset:
//      x_i = [i, 0, 0, 0]
//
//  Consulta:
//      q = [N/2 + 0.1, 0, 0, 0]
//
//  Esto hace que el ground truth sea determinista.
//
//  Advertencia:
//  D=4 es deliberado. Este Gate caracteriza principalmente el coste
//  estructural del índice, no el coste final de un embedding real.
//

#include "baseline/HnswBaseline.hpp"

#include "platform/apple/MemoryInfo.hpp"

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <algorithm>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace racorn_uma::baseline {

namespace {

using namespace unum::usearch;

using racorn_uma::platform::apple::MemoryInfo;

constexpr std::size_t kDimensions = 4;
constexpr std::size_t kConnectivity = 16;
constexpr std::size_t kEfConstruction = 64;
constexpr std::size_t kEfSearch = 64;
constexpr std::size_t kNeighbors = 10;

using VectorKey = std::size_t;
using Slot = std::size_t;

using Index =
    index_gt<float, VectorKey, Slot>;

class VectorStore final {

public:

    explicit VectorStore(
        std::size_t vector_count
    ) {

        data_.resize(
            vector_count *
            kDimensions
        );

        for (
            std::size_t i = 0;
            i < vector_count;
            ++i
        ) {

            float* vector =
                data_.data() +
                i * kDimensions;

            vector[0] =
                static_cast<float>(i);

            vector[1] = 0.0F;
            vector[2] = 0.0F;
            vector[3] = 0.0F;
        }
    }

    [[nodiscard]]
    float const* row(
        std::size_t index
    ) const noexcept {

        return
            data_.data() +
            index * kDimensions;
    }

private:

    std::vector<float> data_;
};

struct L2Metric final {

    using IndexType = Index;

    using MemberRef =
        typename IndexType::member_cref_t;

    using MemberIterator =
        typename IndexType::member_citerator_t;

    VectorStore const* store{};

    [[nodiscard]]
    float const* row(
        std::size_t key
    ) const noexcept {

        return store->row(key);
    }

    float operator()(
        MemberRef const& first,
        MemberRef const& second
    ) const {

        return metric_l2sq_gt<float, float>{}(
            row(get_key(second)),
            row(get_key(first)),
            kDimensions
        );
    }

    float operator()(
        float const* query,
        MemberRef const& member
    ) const {

        return metric_l2sq_gt<float, float>{}(
            query,
            row(get_key(member)),
            kDimensions
        );
    }

    float operator()(
        MemberIterator const& first,
        MemberIterator const& second
    ) const {

        return metric_l2sq_gt<float, float>{}(
            row(get_key(*second)),
            row(get_key(*first)),
            kDimensions
        );
    }

    float operator()(
        float const* query,
        MemberIterator const& member
    ) const {

        return metric_l2sq_gt<float, float>{}(
            query,
            row(get_key(*member)),
            kDimensions
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

struct AllowAll final {

    template <typename MemberLike>
    bool operator()(
        MemberLike&&
    ) const noexcept {

        return true;
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
        exact_set(
            exact.begin(),
            exact.end()
        );

    std::size_t hits = 0;

    for (
        std::size_t key :
        approximate
    ) {

        if (exact_set.contains(key)) {
            ++hits;
        }
    }

    return
        static_cast<double>(hits) /
        static_cast<double>(
            exact.size()
        );
}

} // namespace

HnswBaselineResult
HnswBaseline::run(
    std::size_t vector_count
) {

    if (vector_count < kNeighbors) {

        throw std::invalid_argument(
            "vector_count debe ser >= k"
        );
    }

    HnswBaselineResult output;

    output.memory_start =
        MemoryInfo::capture();

    VectorStore store(
        vector_count
    );

    output.memory_after_vectors =
        MemoryInfo::capture();

    L2Metric metric{
        .store = &store
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

    output.memory_after_reserve =
        MemoryInfo::capture();

    index_update_config_t
        update_configuration;

    update_configuration.expansion =
        kEfConstruction;

    update_configuration.thread =
        0;

    for (
        std::size_t i = 0;
        i < vector_count;
        ++i
    ) {

        index.add(
            static_cast<VectorKey>(i),
            store.row(i),
            metric,
            update_configuration
        );
    }

    output.memory_after_build =
        MemoryInfo::capture();

    const float query[kDimensions]{

        static_cast<float>(
            vector_count / 2
        ) + 0.1F,

        0.0F,
        0.0F,
        0.0F
    };

    AllowAll allow_all;

    index_search_config_t
        exact_configuration;

    exact_configuration.exact =
        true;

    exact_configuration.thread =
        0;

    auto exact_result =
        index.search(
            query,
            kNeighbors,
            metric,
            exact_configuration,
            allow_all
        );

    index_search_config_t
        hnsw_configuration;

    hnsw_configuration.expansion =
        kEfSearch;

    hnsw_configuration.thread =
        0;

    hnsw_configuration.two_hop_threshold =
        -1.0;

    hnsw_configuration.racorn =
        false;

    hnsw_configuration.three_hop_threshold =
        -1.0;

    hnsw_configuration.fallback_exact_threshold =
        0.0;

    std::vector<std::size_t>
        exact_keys =
            extract_keys(
                exact_result
            );

    auto hnsw_result =
        index.search(
            query,
            kNeighbors,
            metric,
            hnsw_configuration,
            allow_all
        );

    std::vector<std::size_t>
        hnsw_keys =
            extract_keys(
                hnsw_result
            );

    output.memory_after_search =
        MemoryInfo::capture();

    const double recall =
        recall_at_k(
            hnsw_keys,
            exact_keys
        );

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

    output.exact_keys =
        std::move(
            exact_keys
        );

    output.hnsw_keys =
        std::move(
            hnsw_keys
        );

    output.recall_at_k =
        recall;

    const bool memory_ok =
        output.memory_start.valid &&
        output.memory_after_vectors.valid &&
        output.memory_after_reserve.valid &&
        output.memory_after_build.valid &&
        output.memory_after_search.valid;

    const bool index_ok =
        output.vector_count ==
        vector_count;

    const bool exact_ok =
        output.exact_keys.size() ==
        kNeighbors;

    const bool hnsw_ok =
        output.hnsw_keys.size() ==
        kNeighbors;

    const bool recall_ok =
        recall == 1.0;

    output.passed =
        memory_ok &&
        index_ok &&
        exact_ok &&
        hnsw_ok &&
        recall_ok;

    return output;
}

} // namespace racorn_uma::baseline

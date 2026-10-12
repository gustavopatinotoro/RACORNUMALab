//
//  HnswBaseline.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar un HNSW parametrizable en N y D para separar el coste
//  del almacenamiento vectorial del coste estructural del índice.
//
//  Gate:
//  G3D — Descomposición memoria VectorStore / HNSW.
//
//  Dataset:
//      x_i = [i, 0, 0, ..., 0]
//
//  Consulta:
//      q = [N/2 + 0.1, 0, 0, ..., 0]
//
//  La geometría es deliberadamente simple y determinista.
//
//  Invariantes:
//  - M = 16.
//  - efConstruction = 64.
//  - efSearch = 64.
//  - Un hilo.
//  - HNSW puro.
//  - Sin filtros.
//  - Sin ACORN.
//  - Sin RACORN.
//  - Sin Metal.
//

#include "baseline/HnswBaseline.hpp"

#include "platform/apple/MemoryInfo.hpp"

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace racorn_uma::baseline {

namespace {

using namespace unum::usearch;

using racorn_uma::platform::apple::MemoryInfo;

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

    VectorStore(
        std::size_t vector_count,
        std::size_t dimensions
    )
        : dimensions_(dimensions),
          data_(
              vector_count *
              dimensions,
              0.0F
          ) {

        for (
            std::size_t i = 0;
            i < vector_count;
            ++i
        ) {

            data_[
                i * dimensions_
            ] =
                static_cast<float>(i);
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

        return store->row(key);
    }

    float operator()(
        MemberRef const& first,
        MemberRef const& second
    ) const {

        return metric_l2sq_gt<
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

        return metric_l2sq_gt<
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

        return metric_l2sq_gt<
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

        return metric_l2sq_gt<
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
    std::size_t vector_count,
    std::size_t dimensions
) {

    if (vector_count < kNeighbors) {
        throw std::invalid_argument(
            "vector_count debe ser >= k"
        );
    }

    if (dimensions == 0) {
        throw std::invalid_argument(
            "dimensions debe ser > 0"
        );
    }

    HnswBaselineResult output;

    output.memory_start =
        MemoryInfo::capture();

    VectorStore store(
        vector_count,
        dimensions
    );

    output.memory_after_vectors =
        MemoryInfo::capture();

    L2Metric metric{
        .store = &store,
        .dimensions = dimensions
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

    std::vector<float>
        query(
            dimensions,
            0.0F
        );

    query[0] =
        static_cast<float>(
            vector_count / 2
        ) + 0.1F;

    AllowAll allow_all;

    index_search_config_t
        exact_configuration;

    exact_configuration.exact =
        true;

    exact_configuration.thread =
        0;

    auto exact_result =
        index.search(
            query.data(),
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
            query.data(),
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
        dimensions;

    output.connectivity =
        kConnectivity;

    output.ef_construction =
        kEfConstruction;

    output.ef_search =
        kEfSearch;

    output.k =
        kNeighbors;

    output.expected_vector_bytes =
        static_cast<std::uint64_t>(
            vector_count
        ) *
        static_cast<std::uint64_t>(
            dimensions
        ) *
        sizeof(float);

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

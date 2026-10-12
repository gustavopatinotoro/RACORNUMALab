//
//  HnswBaseline.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar la primera construcción y consulta HNSW real utilizando
//  exactamente la infraestructura index_gt incluida en RACORN/USEARCH.
//
//  Gate:
//  G3A — Baseline funcional HNSW.
//
//  Diseño experimental:
//  - Dataset sintético determinista.
//  - 16 vectores.
//  - 4 dimensiones.
//  - Distancia L2 al cuadrado.
//  - Construcción secuencial.
//  - Un único contexto de búsqueda.
//  - Oráculo mediante exact search del mismo índice.
//
//  Justificación:
//  La pequeña cardinalidad elimina deliberadamente presión de memoria,
//  concurrencia y efectos de rendimiento. El único objetivo de este Gate
//  es verificar la ruta funcional:
//
//      datos -> index_gt -> HNSW -> búsqueda
//
//  Invariantes:
//  - two_hop_threshold permanece en su valor desactivado (< 0).
//  - racorn permanece desactivado.
//  - three_hop_threshold permanece desactivado.
//  - fallback_exact_threshold permanece desactivado.
//  - No se modifica código vendorizado.
//
//  Seguridad:
//  - No existe entrada externa.
//  - No existe acceso a red.
//  - No existe acceso a archivos.
//  - Todo el almacenamiento utilizado por el experimento es RAII.
//

#include "baseline/HnswBaseline.hpp"

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <algorithm>
#include <cstddef>
#include <new>
#include <unordered_set>
#include <utility>
#include <vector>

namespace racorn_uma::baseline {

namespace {

using namespace unum::usearch;

constexpr std::size_t kVectorCount = 16;
constexpr std::size_t kDimensions = 4;
constexpr std::size_t kConnectivity = 16;
constexpr std::size_t kEfConstruction = 64;
constexpr std::size_t kEfSearch = 64;
constexpr std::size_t kNeighbors = 3;

using VectorKey = std::size_t;
using Slot = std::size_t;

using Index =
    index_gt<float, VectorKey, Slot>;

//
// Almacén de vectores.
//
// RACORN/USEARCH conserva la estructura HNSW mientras la métrica utilizada
// por este experimento resuelve el vector asociado a cada key desde este
// almacenamiento externo.
//
class VectorStore final {

public:

    VectorStore() {

        data_.reserve(
            kVectorCount * kDimensions
        );

        for (std::size_t i = 0; i < kVectorCount; ++i) {

            data_.push_back(
                static_cast<float>(i)
            );

            data_.push_back(0.0F);
            data_.push_back(0.0F);
            data_.push_back(0.0F);
        }
    }

    [[nodiscard]]
    float const* row(
        std::size_t index
    ) const noexcept {

        return data_.data() +
               index * kDimensions;
    }

private:

    std::vector<float> data_;
};

//
// Métrica L2².
//
// Sigue el contrato de métrica requerido por index_gt.
// La distancia al cuadrado conserva exactamente el mismo orden
// de vecinos que la distancia euclídea.
//

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

//
// index_gt necesita conservar el alineamiento utilizado por el
// artefacto oficial. Esta envoltura administra ese almacenamiento
// mediante RAII.
//

template <typename IndexType>
class AlignedIndex final {

public:

    using Allocator =
        aligned_allocator_gt<IndexType, 64>;

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
// Predicado neutro.
//
// Se utiliza para que HNSW y exact search observen exactamente
// el mismo universo de nodos.
//

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

        if (
            exact_set.contains(key)
        ) {

            ++hits;
        }
    }

    return
        static_cast<double>(hits) /
        static_cast<double>(exact.size());
}

} // namespace

HnswBaselineResult
HnswBaseline::run() {

    VectorStore store;

    L2Metric metric{
        .store = &store
    };

    index_config_t index_configuration(
        kConnectivity
    );

    AlignedIndex<Index>
        aligned_index(
            index_configuration
        );

    Index& index =
        aligned_index.get();

    //
    // Un único contexto de trabajo.
    //

    index.reserve(
        index_limits_t(
            kVectorCount,
            1
        )
    );

    //
    // Construcción HNSW.
    //

    index_update_config_t
        update_configuration;

    update_configuration.expansion =
        kEfConstruction;

    update_configuration.thread = 0;

    for (
        std::size_t i = 0;
        i < kVectorCount;
        ++i
    ) {

        index.add(
            static_cast<VectorKey>(i),
            store.row(i),
            metric,
            update_configuration
        );
    }

    //
    // Consulta deliberadamente situada cerca del vector 5.
    //
    // Distancias L2² principales:
    //
    // key 5 -> (5.1 - 5)^2 = 0.01
    // key 6 -> (5.1 - 6)^2 = 0.81
    // key 4 -> (5.1 - 4)^2 = 1.21
    //

    const float query[kDimensions]{
        5.1F,
        0.0F,
        0.0F,
        0.0F
    };

    AllowAll allow_all;

    //
    // Oráculo exacto.
    //

    index_search_config_t
        exact_configuration;

    exact_configuration.exact = true;
    exact_configuration.thread = 0;

    auto exact_result =
        index.search(
            query,
            kNeighbors,
            metric,
            exact_configuration,
            allow_all
        );

    //
    // HNSW puro.
    //
    // two_hop_threshold < 0 implica ruta HNSW estándar.
    //

    index_search_config_t
        hnsw_configuration;

    hnsw_configuration.expansion =
        kEfSearch;

    hnsw_configuration.thread = 0;

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

    const double recall =
        recall_at_k(
            hnsw_keys,
            exact_keys
        );

    const bool size_ok =
        index.size() ==
        kVectorCount;

    const bool exact_size_ok =
        exact_keys.size() ==
        kNeighbors;

    const bool hnsw_size_ok =
        hnsw_keys.size() ==
        kNeighbors;

    const bool recall_ok =
        recall == 1.0;

    HnswBaselineResult output;

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
        std::move(exact_keys);

    output.hnsw_keys =
        std::move(hnsw_keys);

    output.recall_at_k =
        recall;

    output.passed =
        size_ok &&
        exact_size_ok &&
        hnsw_size_ok &&
        recall_ok;

    return output;
}

} // namespace racorn_uma::baseline

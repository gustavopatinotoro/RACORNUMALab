//
//  RacornUmaSemanticProbe.mm
//  RACORNUMALab
//
//  Propósito:
//  Integrar por primera vez RACORN CPU con un bitset de elegibilidad
//  producido por GPU Metal directamente sobre memoria unificada.
//
//  Gate:
//  G-UMA4 — RACORN CPU consumiendo bitset GPU/UMA.
//
//  Restricción fundamental:
//  RACORN y HNSW permanecen completamente inalterados.
//
//  La GPU solamente transforma:
//
//      metadata -> eligibility bitset
//
//  La CPU sigue ejecutando:
//
//      HNSW
//      ACORN
//      RACORN ASF
//      bridge selection
//      heaps
//      visited state
//      termination
//
//  Diseño experimental:
//  - N = 20.000.
//  - D = 32.
//  - M = 16.
//  - efConstruction = 100.
//  - efSearch = 200.
//  - k = 10.
//  - Q = 32.
//  - two_hop_threshold = 0.30.
//  - bridge_budget_ratio = 1.0.
//  - stride sampling = true.
//  - 3-hop desactivado.
//  - RACORN-1+ desactivado.
//
//  Rutas comparadas:
//
//  A. Predicate CPU sobre las mismas columnas Shared.
//  B. Lookup CPU O(1) sobre bitset Shared producido por GPU.
//
//  La única diferencia permitida es el mecanismo que responde:
//
//      ¿este ID satisface el filtro?
//
//  Condición PASS:
//  - bitset GPU == predicate CPU para todos los IDs;
//  - ground truth exacto completo;
//  - mismo Top-k por consulta;
//  - mismo computed_distances;
//  - mismo visited_members;
//  - mismo predicate_evaluations;
//  - mismo fallback_count;
//  - todos los resultados válidos.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - vendor/racorn permanece immutable.
//

#include "platform/apple/RacornUmaSemanticProbe.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace racorn_uma::platform::apple {

namespace {

using namespace unum::usearch;

//
// Configuración reproducible.
//


constexpr std::size_t kVectorCount =
    20'000;

constexpr std::size_t kDimensions =
    32;

constexpr std::size_t kConnectivity =
    16;

constexpr std::size_t kEfConstruction =
    100;

constexpr std::size_t kEfSearch =
    200;

constexpr std::size_t kNeighbors =
    10;

constexpr std::size_t kQueryCount =
    32;

constexpr double kTwoHopThreshold =
    0.30;

constexpr double kBridgeBudgetRatio =
    1.0;

constexpr std::size_t kMetadataColumnCount =
    5;

constexpr std::size_t kBitsPerWord =
    32;

constexpr char const* kKernelName =
    "evaluate_metadata_predicate_to_bitset";

constexpr std::uint64_t kBaseSeed =
    0x13579BDF2468ACE0ULL;

constexpr std::uint64_t kQuerySeed =
    0xCAFEBABE12345678ULL;

constexpr std::uint64_t kMetadataSeed =
    0xD1B54A32D192ED03ULL;

constexpr std::uint32_t kTenantRequired =
    7U;

constexpr std::uint32_t kMinimumYear =
    2022U;

constexpr std::uint32_t kLanguageRequired =
    2U;

constexpr std::uint32_t kCategoryA =
    3U;

constexpr std::uint32_t kCategoryB =
    7U;

constexpr std::uint32_t kCategoryC =
    9U;

constexpr std::uint32_t kMaximumSecurityLevel =
    2U;

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

//
// Utilidades Metal.
//

std::string string_from_nsstring(
    NSString* value
) {

    if (
        value ==
        nil
    ) {

        return {};
    }

    const char* utf8 =
        [value UTF8String];

    if (
        utf8 ==
        nullptr
    ) {

        return {};
    }

    return std::string(
        utf8
    );
}

std::size_t choose_threads_per_threadgroup(
    std::size_t execution_width,
    std::size_t maximum
) noexcept {

    if (
        maximum ==
        0
    ) {

        return 0;
    }

    constexpr std::size_t preferred =
        256;

    std::size_t selected =
        std::min(
            preferred,
            maximum
        );

    if (
        execution_width ==
        0
    ) {

        return selected;
    }

    selected =
        (
            selected /
            execution_width
        ) *
        execution_width;

    if (
        selected ==
        0
    ) {

        selected =
            std::min(
                execution_width,
                maximum
            );
    }

    return selected;
}

//
// Generador SplitMix64.
//


std::uint64_t splitmix64_step(
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

float random_signed_float(
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

std::uint64_t metadata_hash(
    std::size_t index
) noexcept {

    return
        splitmix64_value(
            kMetadataSeed ^
            static_cast<std::uint64_t>(
                index
            )
        );
}

//
// Vectores base.
//

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
          data_(
              count *
              dimensions
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
            index *
                dimensions_;
    }

private:

    std::size_t dimensions_{};

    std::vector<float> data_;
};

//
// Consultas.
//

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
          data_(
              count *
              dimensions
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
            index *
                dimensions_;
    }

private:

    std::size_t dimensions_{};

    std::vector<float> data_;
};

//
// Métrica L2sq idéntica al baseline validado.
//

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

//
// Wrapper RAII para index_gt.
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

//
// Vista de las cinco columnas Shared.
//

struct MetadataView final {

    std::uint32_t const* tenant{};
    std::uint32_t const* category{};
    std::uint32_t const* year{};
    std::uint32_t const* language{};
    std::uint32_t const* security_level{};
};

bool category_allowed(
    std::uint32_t category
) noexcept {

    return
        category ==
            kCategoryA ||
        category ==
            kCategoryB ||
        category ==
            kCategoryC;
}

bool evaluate_metadata(
    MetadataView const& metadata,
    std::size_t key
) noexcept {

    return
        metadata.tenant[key] ==
            kTenantRequired &&

        metadata.year[key] >=
            kMinimumYear &&

        metadata.language[key] ==
            kLanguageRequired &&

        category_allowed(
            metadata.category[key]
        ) &&

        metadata.security_level[key] <=
            kMaximumSecurityLevel;
}

//
// Predicate de referencia.
//
// RACORN llama a este functor exactamente como antes.
// La lógica de navegación no conoce la representación de metadata.
//

struct CpuMetadataPredicate final {

    MetadataView metadata{};

    [[nodiscard]]
    bool accepts_key(
        std::size_t key
    ) const noexcept {

        return
            evaluate_metadata(
                metadata,
                key
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

//
// Predicate basado en bitset.
//
// Lookup:
//
//      word = key >> 5
//      bit  = key & 31
//
// Tiempo O(1), sin lógica Metal durante navegación.
//

struct BitsetPredicate final {

    std::uint32_t const* bitset{};

    [[nodiscard]]
    bool accepts_key(
        std::size_t key
    ) const noexcept {

        const std::size_t word_index =
            key >>
            5U;

        const std::size_t bit_index =
            key &
            31U;

        return
            (
                (
                    bitset[
                        word_index
                    ] >>
                    bit_index
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

std::size_t bitset_word_count_for(
    std::size_t element_count
) noexcept {

    return
        (
            element_count +
            kBitsPerWord -
            1U
        ) /
        kBitsPerWord;
}

//
// Configuración RACORN.
//
// Es exactamente la misma para ambas rutas.
//

index_search_config_t
make_racorn_search_config() {

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
        kBridgeBudgetRatio;

    config.three_hop_threshold =
        -1.0;

    config.fallback_exact_threshold =
        0.0;

    config.racorn_stride_sampling =
        true;

    return config;
}

//
// Snapshot independiente del scratch interno de USEARCH.
//

struct SearchSnapshot final {

    std::vector<std::size_t> keys;

    std::size_t computed_distances{};
    std::size_t visited_members{};
    std::size_t predicate_evaluations{};
    std::size_t fallback_count{};
};

SearchSnapshot snapshot_search_result(
    Index::search_result_t const& result
) {

    SearchSnapshot snapshot;

    snapshot.keys.reserve(
        result.size()
    );

    for (
        std::size_t index = 0;
        index < result.size();
        ++index
    ) {

        snapshot.keys.push_back(
            static_cast<std::size_t>(
                result[
                    index
                ].member.key
            )
        );
    }

    snapshot.computed_distances =
        result.computed_distances;

    snapshot.visited_members =
        result.visited_members;

    snapshot.predicate_evaluations =
        result.predicate_evaluations;

    snapshot.fallback_count =
        result.fallback_count;

    return snapshot;
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
        static_cast<double>(
            hits
        ) /
        static_cast<double>(
            exact.size()
        );
}

template <typename Predicate>
bool result_satisfies_filter(
    std::vector<std::size_t> const& keys,
    Predicate const& predicate
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

bool same_double(
    double first,
    double second
) noexcept {

    return
        std::abs(
            first -
            second
        ) <=
        1e-12;
}

} // namespace

RacornUmaSemanticProbeResult
RacornUmaSemanticProbe::run() {

    RacornUmaSemanticProbeResult output;

    output.vector_count =
        kVectorCount;

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

    output.bridge_budget_ratio =
        kBridgeBudgetRatio;

    output.metadata_column_count =
        kMetadataColumnCount;

    output.kernel_name =
        kKernelName;

    output.racorn_modified =
        false;

    @autoreleasepool {

        //
        // ============================================================
        // 1. Metal
        // ============================================================
        //

        id<MTLDevice> device =
            MTLCreateSystemDefaultDevice();

        if (
            device ==
            nil
        ) {

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
            [device newDefaultLibrary];

        if (
            library ==
            nil
        ) {

            output.metal_error =
                "No se pudo cargar la biblioteca Metal predeterminada.";

            return output;
        }

        output.library_available =
            true;

        NSString* kernel_name =
            [NSString
                stringWithUTF8String:
                    kKernelName];

        id<MTLFunction> function =
            [
                library
                newFunctionWithName:
                    kernel_name
            ];

        if (
            function ==
            nil
        ) {

            output.metal_error =
                "No se encontro evaluate_metadata_predicate_to_bitset.";

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

            if (
                pipeline_error !=
                nil
            ) {

                output.metal_error =
                    string_from_nsstring(
                        pipeline_error
                            .localizedDescription
                    );

            } else {

                output.metal_error =
                    "No se pudo crear el pipeline Metal.";
            }

            return output;
        }

        output.pipeline_available =
            true;

        id<MTLCommandQueue> command_queue =
            [device newCommandQueue];

        if (
            command_queue ==
            nil
        ) {

            output.metal_error =
                "No se pudo crear MTLCommandQueue.";

            return output;
        }

        output.command_queue_available =
            true;

        //
        // ============================================================
        // 2. Metadata Shared
        // ============================================================
        //

        const std::size_t column_bytes =
            kVectorCount *
            sizeof(
                std::uint32_t
            );

        const std::size_t metadata_bytes =
            column_bytes *
            kMetadataColumnCount;

        output.metadata_bytes =
            static_cast<std::uint64_t>(
                metadata_bytes
            );

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
            metadata_buffer.storageMode !=
                MTLStorageModeShared ||
            metadata_buffer.contents ==
                nullptr
        ) {

            output.metal_error =
                "No se pudo crear metadata Shared.";

            return output;
        }

        output.metadata_buffer_available =
            true;

        auto* metadata =
            static_cast<std::uint32_t*>(
                metadata_buffer.contents
            );

        std::uint32_t* tenant =
            metadata +
            (
                0 *
                kVectorCount
            );

        std::uint32_t* category =
            metadata +
            (
                1 *
                kVectorCount
            );

        std::uint32_t* year =
            metadata +
            (
                2 *
                kVectorCount
            );

        std::uint32_t* language =
            metadata +
            (
                3 *
                kVectorCount
            );

        std::uint32_t* security_level =
            metadata +
            (
                4 *
                kVectorCount
            );

        //
        // CPU inicializa directamente el recurso Shared.
        //

        for (
            std::size_t index = 0;
            index < kVectorCount;
            ++index
        ) {

            const std::uint64_t hash =
                metadata_hash(
                    index
                );

            tenant[index] =
                static_cast<std::uint32_t>(
                    hash &
                    0x0FU
                );

            category[index] =
                static_cast<std::uint32_t>(
                    (
                        hash >>
                        8U
                    ) &
                    0x0FU
                );

            year[index] =
                2019U +
                static_cast<std::uint32_t>(
                    (
                        hash >>
                        16U
                    ) %
                    8U
                );

            language[index] =
                static_cast<std::uint32_t>(
                    (
                        hash >>
                        24U
                    ) &
                    0x03U
                );

            security_level[index] =
                static_cast<std::uint32_t>(
                    (
                        hash >>
                        32U
                    ) %
                    6U
                );
        }

        const MetadataView metadata_view{
            .tenant =
                tenant,

            .category =
                category,

            .year =
                year,

            .language =
                language,

            .security_level =
                security_level
        };

        const CpuMetadataPredicate cpu_predicate{
            .metadata =
                metadata_view
        };

        //
        // ============================================================
        // 3. Bitset Shared
        // ============================================================
        //

        output.bitset_word_count =
            bitset_word_count_for(
                kVectorCount
            );

        output.bitset_bytes =
            static_cast<std::uint64_t>(
                output.bitset_word_count *
                sizeof(
                    std::uint32_t
                )
            );

        id<MTLBuffer> bitset_buffer =
            [
                device
                newBufferWithLength:
                    static_cast<NSUInteger>(
                        output.bitset_bytes
                    )
                options:
                    MTLResourceStorageModeShared
            ];

        if (
            bitset_buffer ==
                nil ||
            bitset_buffer.storageMode !=
                MTLStorageModeShared ||
            bitset_buffer.contents ==
                nullptr
        ) {

            output.metal_error =
                "No se pudo crear eligibility bitset Shared.";

            return output;
        }

        output.bitset_buffer_available =
            true;

        auto* bitset =
            static_cast<std::uint32_t*>(
                bitset_buffer.contents
            );

        //
        // ============================================================
        // 4. GPU: metadata -> bitset
        // ============================================================
        //

        id<MTLCommandBuffer> command_buffer =
            [command_queue commandBuffer];

        if (
            command_buffer ==
            nil
        ) {

            output.metal_error =
                "No se pudo crear MTLCommandBuffer.";

            return output;
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

            output.metal_error =
                "No se pudo crear MTLComputeCommandEncoder.";

            return output;
        }

        [
            encoder
            setComputePipelineState:
                pipeline
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                0 *
                column_bytes
            atIndex:
                0
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                1 *
                column_bytes
            atIndex:
                1
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                2 *
                column_bytes
            atIndex:
                2
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                3 *
                column_bytes
            atIndex:
                3
        ];

        [
            encoder
            setBuffer:
                metadata_buffer
            offset:
                4 *
                column_bytes
            atIndex:
                4
        ];

        [
            encoder
            setBuffer:
                bitset_buffer
            offset:
                0
            atIndex:
                5
        ];

        const std::uint32_t element_count =
            static_cast<std::uint32_t>(
                kVectorCount
            );

        [
            encoder
            setBytes:
                &element_count
            length:
                sizeof(
                    element_count
                )
            atIndex:
                6
        ];

        const std::size_t execution_width =
            static_cast<std::size_t>(
                pipeline.threadExecutionWidth
            );

        const std::size_t maximum_threads =
            static_cast<std::size_t>(
                pipeline
                    .maxTotalThreadsPerThreadgroup
            );

        const std::size_t threads_per_group =
            choose_threads_per_threadgroup(
                execution_width,
                maximum_threads
            );

        if (
            threads_per_group ==
            0
        ) {

            output.metal_error =
                "Configuracion invalida de threadgroup.";

            return output;
        }

        const MTLSize grid_size =
            MTLSizeMake(
                output.bitset_word_count,
                1,
                1
            );

        const MTLSize threadgroup_size =
            MTLSizeMake(
                threads_per_group,
                1,
                1
            );

        [
            encoder
            dispatchThreads:
                grid_size
            threadsPerThreadgroup:
                threadgroup_size
        ];

        [
            encoder
            endEncoding
        ];

        [
            command_buffer
            commit
        ];

        //
        // Único punto de sincronización GPU -> CPU.
        //

        [
            command_buffer
            waitUntilCompleted
        ];

        if (
            command_buffer.status !=
                MTLCommandBufferStatusCompleted
        ) {

            if (
                command_buffer.error !=
                    nil
            ) {

                output.metal_error =
                    string_from_nsstring(
                        command_buffer
                            .error
                            .localizedDescription
                    );

            } else {

                output.metal_error =
                    "El Command Buffer Metal no finalizo correctamente.";
            }

            return output;
        }

        output.command_completed =
            true;

        //
        // ============================================================
        // 5. Verificación completa CPU predicate == GPU bitset
        // ============================================================
        //

        const BitsetPredicate bitset_predicate{
            .bitset =
                bitset
        };

        std::size_t cpu_selected =
            0;

        std::size_t gpu_selected =
            0;

        std::size_t predicate_mismatches =
            0;

        for (
            std::size_t key = 0;
            key < kVectorCount;
            ++key
        ) {

            const bool cpu_accepts =
                cpu_predicate.accepts_key(
                    key
                );

            const bool gpu_accepts =
                bitset_predicate.accepts_key(
                    key
                );

            if (
                cpu_accepts
            ) {

                ++cpu_selected;
            }

            if (
                gpu_accepts
            ) {

                ++gpu_selected;
            }

            if (
                cpu_accepts !=
                gpu_accepts
            ) {

                ++predicate_mismatches;
            }
        }

        output.cpu_selected_count =
            cpu_selected;

        output.gpu_selected_count =
            gpu_selected;

        output.observed_selectivity =
            static_cast<double>(
                cpu_selected
            ) /
            static_cast<double>(
                kVectorCount
            );

        output.predicate_bit_mismatch_count =
            predicate_mismatches;

        output.predicate_exact_match =
            predicate_mismatches ==
                0 &&
            cpu_selected ==
                gpu_selected;

        if (
            !output.predicate_exact_match
        ) {

            return output;
        }

        //
        // ============================================================
        // 6. Construcción del único índice HNSW
        // ============================================================
        //

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
            .store =
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

        for (
            std::size_t vector_index = 0;
            vector_index <
                kVectorCount;
            ++vector_index
        ) {

            index.add(
                static_cast<VectorKey>(
                    vector_index
                ),
                base.row(
                    vector_index
                ),
                metric,
                update_configuration
            );
        }

        //
        // ============================================================
        // 7. Comparación RACORN CPU vs RACORN GPU-bitset
        // ============================================================
        //

        double cpu_recall_sum =
            0.0;

        double gpu_recall_sum =
            0.0;

        double cpu_distances_sum =
            0.0;

        double gpu_distances_sum =
            0.0;

        double cpu_visited_sum =
            0.0;

        double gpu_visited_sum =
            0.0;

        double cpu_predicate_evaluations_sum =
            0.0;

        double gpu_predicate_evaluations_sum =
            0.0;

        double cpu_fallback_sum =
            0.0;

        double gpu_fallback_sum =
            0.0;

        bool exact_complete =
            true;

        bool cpu_filter_valid =
            true;

        bool gpu_filter_valid =
            true;

        std::size_t topk_mismatch_queries =
            0;

        std::size_t recall_mismatch_queries =
            0;

        std::size_t distances_mismatch_queries =
            0;

        std::size_t visited_mismatch_queries =
            0;

        std::size_t predicate_evaluation_mismatch_queries =
            0;

        std::size_t fallback_mismatch_queries =
            0;

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
            // Ground truth exacto filtrado.
            //

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
                    cpu_predicate
                );

            const SearchSnapshot exact_snapshot =
                snapshot_search_result(
                    exact_result
                );

            if (
                exact_snapshot.keys.size() !=
                    kNeighbors ||
                !result_satisfies_filter(
                    exact_snapshot.keys,
                    cpu_predicate
                )
            ) {

                exact_complete =
                    false;
            }

            //
            // RACORN con predicado CPU original.
            //

            auto cpu_configuration =
                make_racorn_search_config();

            auto cpu_result =
                index.search(
                    query,
                    kNeighbors,
                    metric,
                    cpu_configuration,
                    cpu_predicate
                );

            const SearchSnapshot cpu_snapshot =
                snapshot_search_result(
                    cpu_result
                );

            //
            // RACORN con bitset generado por GPU.
            //

            auto gpu_configuration =
                make_racorn_search_config();

            auto gpu_result =
                index.search(
                    query,
                    kNeighbors,
                    metric,
                    gpu_configuration,
                    bitset_predicate
                );

            const SearchSnapshot gpu_snapshot =
                snapshot_search_result(
                    gpu_result
                );

            //
            // Calidad.
            //

            const double cpu_recall =
                recall_at_k(
                    cpu_snapshot.keys,
                    exact_snapshot.keys
                );

            const double gpu_recall =
                recall_at_k(
                    gpu_snapshot.keys,
                    exact_snapshot.keys
                );

            cpu_recall_sum +=
                cpu_recall;

            gpu_recall_sum +=
                gpu_recall;

            //
            // Contadores.
            //

            cpu_distances_sum +=
                static_cast<double>(
                    cpu_snapshot
                        .computed_distances
                );

            gpu_distances_sum +=
                static_cast<double>(
                    gpu_snapshot
                        .computed_distances
                );

            cpu_visited_sum +=
                static_cast<double>(
                    cpu_snapshot
                        .visited_members
                );

            gpu_visited_sum +=
                static_cast<double>(
                    gpu_snapshot
                        .visited_members
                );

            cpu_predicate_evaluations_sum +=
                static_cast<double>(
                    cpu_snapshot
                        .predicate_evaluations
                );

            gpu_predicate_evaluations_sum +=
                static_cast<double>(
                    gpu_snapshot
                        .predicate_evaluations
                );

            cpu_fallback_sum +=
                static_cast<double>(
                    cpu_snapshot
                        .fallback_count
                );

            gpu_fallback_sum +=
                static_cast<double>(
                    gpu_snapshot
                        .fallback_count
                );

            //
            // Validez del filtro.
            //

            cpu_filter_valid =
                cpu_filter_valid &&
                result_satisfies_filter(
                    cpu_snapshot.keys,
                    cpu_predicate
                );

            gpu_filter_valid =
                gpu_filter_valid &&
                result_satisfies_filter(
                    gpu_snapshot.keys,
                    cpu_predicate
                );

            //
            // Equivalencia semántica fuerte.
            //

            if (
                cpu_snapshot.keys !=
                gpu_snapshot.keys
            ) {

                ++topk_mismatch_queries;
            }

            if (
                !same_double(
                    cpu_recall,
                    gpu_recall
                )
            ) {

                ++recall_mismatch_queries;
            }

            if (
                cpu_snapshot.computed_distances !=
                gpu_snapshot.computed_distances
            ) {

                ++distances_mismatch_queries;
            }

            if (
                cpu_snapshot.visited_members !=
                gpu_snapshot.visited_members
            ) {

                ++visited_mismatch_queries;
            }

            if (
                cpu_snapshot.predicate_evaluations !=
                gpu_snapshot.predicate_evaluations
            ) {

                ++predicate_evaluation_mismatch_queries;
            }

            if (
                cpu_snapshot.fallback_count !=
                gpu_snapshot.fallback_count
            ) {

                ++fallback_mismatch_queries;
            }
        }

        //
        // ============================================================
        // 8. Agregación
        // ============================================================
        //

        const double query_count =
            static_cast<double>(
                kQueryCount
            );

        output.exact_ground_truth_complete =
            exact_complete;

        output.cpu_mean_recall_at_k =
            cpu_recall_sum /
            query_count;

        output.gpu_bitset_mean_recall_at_k =
            gpu_recall_sum /
            query_count;

        output.cpu_mean_computed_distances =
            cpu_distances_sum /
            query_count;

        output.gpu_mean_computed_distances =
            gpu_distances_sum /
            query_count;

        output.cpu_mean_visited_members =
            cpu_visited_sum /
            query_count;

        output.gpu_mean_visited_members =
            gpu_visited_sum /
            query_count;

        output.cpu_mean_predicate_evaluations =
            cpu_predicate_evaluations_sum /
            query_count;

        output.gpu_mean_predicate_evaluations =
            gpu_predicate_evaluations_sum /
            query_count;

        output.cpu_mean_fallback_count =
            cpu_fallback_sum /
            query_count;

        output.gpu_mean_fallback_count =
            gpu_fallback_sum /
            query_count;

        output.topk_mismatch_queries =
            topk_mismatch_queries;

        output.recall_mismatch_queries =
            recall_mismatch_queries;

        output.computed_distances_mismatch_queries =
            distances_mismatch_queries;

        output.visited_members_mismatch_queries =
            visited_mismatch_queries;

        output.predicate_evaluations_mismatch_queries =
            predicate_evaluation_mismatch_queries;

        output.fallback_count_mismatch_queries =
            fallback_mismatch_queries;

        output.cpu_results_filter_valid =
            cpu_filter_valid;

        output.gpu_results_filter_valid =
            gpu_filter_valid;

        output.navigation_semantics_identical =
            topk_mismatch_queries ==
                0 &&
            recall_mismatch_queries ==
                0 &&
            distances_mismatch_queries ==
                0 &&
            visited_mismatch_queries ==
                0 &&
            predicate_evaluation_mismatch_queries ==
                0 &&
            fallback_mismatch_queries ==
                0;

        output.valid =
            output.metal_available &&
            output.unified_memory &&
            output.library_available &&
            output.kernel_available &&
            output.pipeline_available &&
            output.command_queue_available &&
            output.metadata_buffer_available &&
            output.bitset_buffer_available &&
            output.command_completed &&
            output.predicate_exact_match &&
            output.cpu_selected_count >=
                kNeighbors &&
            output.exact_ground_truth_complete &&
            output.cpu_results_filter_valid &&
            output.gpu_results_filter_valid &&
            output.navigation_semantics_identical &&
            !output.racorn_modified;
    }

    return output;
}

} // namespace racorn_uma::platform::apple

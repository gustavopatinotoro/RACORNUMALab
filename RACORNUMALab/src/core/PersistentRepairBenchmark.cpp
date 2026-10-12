//
//  PersistentRepairBenchmark.cpp
//  RACORNUMALab
//
//  Gate:
//      G-REPAIR0 — Persistent Repair vs RACORN Transient.
//
//  Pregunta experimental:
//
//      Si el mismo predicado se repite muchas veces:
//
//      ¿resulta rentable pagar una reparación topológica una sola vez
//      y reutilizarla, frente a permitir que RACORN reconstruya
//      transitoriamente sus bridges durante cada consulta?
//
//  Reparación persistente:
//
//      1. Materializar P una sola vez.
//
//      2. Extraer una instantánea L0 mediante la API pública
//         de serialización de USEARCH/RACORN.
//
//      3. Construir el subgrafo inducido por los nodos elegibles.
//
//      4. Para cada:
//
//             u -> x -> v
//
//         donde:
//
//             P(u) = true
//             P(x) = false
//             P(v) = true
//
//         proponer un shortcut persistente:
//
//             u <-> v
//
//      5. Mantener como máximo 16 propuestas 2-hop por nodo,
//         priorizadas por distancia L2.
//
//      6. Si quedan componentes desconectados, unirlas una vez
//         mediante un bosque mínimo entre componentes.
//
//  Importante:
//
//  - El HNSW original NO se modifica.
//  - RACORN upstream NO se modifica.
//  - El overlay almacena exclusivamente topología.
//  - Los vectores SIFT continúan siendo compartidos.
//  - El coste de materializar P es común a ambos planes y no
//    forma parte del break-even.
//  - Exact se utiliza únicamente como oráculo de Recall.
//
//  Costes:
//
//      Transient:
//
//          C_T(R) = R * L_T
//
//      Persistent:
//
//          C_P(R) = B_P + R * L_P
//
//      Break-even:
//
//          R* = B_P / (L_T - L_P)
//
//  Se informan dos B_P:
//
//      repair_build_ms
//
//          coste algorítmico de construir el overlay partiendo
//          de una vista L0 ya disponible.
//
//      total_persistent_build_ms
//
//          snapshot externo + construcción del overlay.
//
//  Esto permite separar el coste propio del algoritmo del coste
//  introducido por la actual API pública de RACORN.
//

#include "core/PersistentRepairBenchmark.hpp"

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

using Clock = std::chrono::steady_clock;

constexpr char const *kBaseFilename = "sift100k_base_f32le.bin";

constexpr char const *kQueryFilename = "sift10k_query_f32le.bin";

constexpr char const *kMetadataFilename = "sift100k_metadata_u32le_soa.bin";

constexpr std::size_t kVectorCount = 100'000U;

constexpr std::size_t kDimensions = 128U;

constexpr std::size_t kStoredQueryCount = 10'000U;

constexpr std::size_t kQueryCount = 1'000U;

constexpr std::size_t kMetadataColumnCount = 5U;

constexpr std::size_t kConnectivity = 16U;

constexpr std::size_t kEfConstruction = 100U;

constexpr std::size_t kEfSearch = 200U;

constexpr std::size_t kPersistentEfSearch = 200U;

constexpr std::size_t kNeighbors = 10U;

constexpr std::size_t kBitsPerWord = 32U;

constexpr std::size_t kShortcutProposalBudget = 16U;

constexpr std::size_t kWarmupRounds = 1U;

constexpr std::size_t kMeasuredRounds = 3U;

constexpr double kTwoHopThreshold = 0.30;

constexpr double kBridgeBudgetRatio = 1.00;

constexpr double kRecallFloor = 0.990;

constexpr double kRecallToleranceVsTransient = 0.005;

constexpr std::uint32_t kTenantRequired = 7U;

constexpr std::uint32_t kLanguageMin = 2U;

constexpr std::array<std::size_t, 8> kRepetitionPoints{
    1U, 10U, 25U, 50U, 100U, 250U, 500U, 1'000U};

using VectorKey = std::size_t;

using Slot = std::size_t;

using Index = index_gt<float, VectorKey, Slot>;

using SerializedLevel = std::int16_t;

using SerializedNeighborsCount = std::uint32_t;

struct FilterSpec final {

    std::uint32_t tenant_required{};
    std::uint32_t language_min{};
};

struct MetadataView final {

    std::uint32_t const *tenant{};
    std::uint32_t const *category{};
    std::uint32_t const *year{};
    std::uint32_t const *language{};
    std::uint32_t const *security_level{};
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

struct BaseTopologySnapshot final {

    std::vector<std::uint32_t> row_offsets;

    std::vector<std::uint32_t> neighbors;

    std::uint32_t entry_slot{};

    std::size_t serialized_bytes{};

    bool keys_are_identity{};
    bool valid{};
};

struct PersistentOverlay final {

    std::vector<std::uint32_t> global_ids;

    std::vector<std::uint32_t> row_offsets;

    std::vector<std::uint32_t> neighbors;

    std::uint32_t entry_local{};

    std::size_t induced_undirected_edges{};
    std::size_t two_hop_shortcut_edges{};
    std::size_t completion_edges{};

    //
    // Número total de aristas no dirigidas presentes en el
    // overlay después de:
    //
    //     subgrafo inducido
    //     + shortcuts persistentes 2-hop
    //     + reparación residual de componentes
    //
    std::size_t final_undirected_edges{};

    std::size_t components_induced{};
    std::size_t components_after_two_hop{};
    std::size_t components_final{};

    [[nodiscard]]
    std::size_t logical_bytes() const noexcept {

        return global_ids.size() * sizeof(std::uint32_t) +

               row_offsets.size() * sizeof(std::uint32_t) +

               neighbors.size() * sizeof(std::uint32_t) +

               sizeof(entry_local);
    }

    [[nodiscard]]
    std::size_t allocated_bytes() const noexcept {

        return global_ids.capacity() * sizeof(std::uint32_t) +

               row_offsets.capacity() * sizeof(std::uint32_t) +

               neighbors.capacity() * sizeof(std::uint32_t) +

               sizeof(entry_local);
    }
};

struct OverlayCandidate final {

    float distance{};
    std::uint32_t local{};
};

struct OverlaySearchResult final {

    std::array<std::uint32_t, kNeighbors> keys{};

    std::size_t count{};

    std::size_t computed_distances{};
    std::size_t visited_members{};
};

struct ShortcutCandidate final {

    float distance{};
    std::uint32_t local{};
};

struct ComponentEdge final {

    float distance{};

    std::uint32_t first{};
    std::uint32_t second{};

    std::uint32_t first_component{};
    std::uint32_t second_component{};
};

class DenseVectors final {

  public:
    DenseVectors(std::vector<float> values, std::size_t dimensions)
        : values_(std::move(values)), dimensions_(dimensions) {}

    [[nodiscard]]
    float const *row(std::size_t index) const noexcept {

        return values_.data() + index * dimensions_;
    }

  private:
    std::vector<float> values_;
    std::size_t dimensions_{};
};

struct L2Metric final {

    using IndexType = Index;

    using MemberRef = typename IndexType::member_cref_t;

    using MemberIterator = typename IndexType::member_citerator_t;

    DenseVectors const *vectors{};
    std::size_t dimensions{};

    [[nodiscard]]
    float const *row(std::size_t key) const noexcept {

        return vectors->row(key);
    }

    float operator()(MemberRef const &first, MemberRef const &second) const {

        return metric_l2sq_gt<float, float>{}(row(get_key(second)),
                                              row(get_key(first)), dimensions);
    }

    float operator()(float const *query, MemberRef const &member) const {

        return metric_l2sq_gt<float, float>{}(query, row(get_key(member)),
                                              dimensions);
    }

    float operator()(MemberIterator const &first,
                     MemberIterator const &second) const {

        return metric_l2sq_gt<float, float>{}(row(get_key(*second)),
                                              row(get_key(*first)), dimensions);
    }

    float operator()(float const *query, MemberIterator const &member) const {

        return metric_l2sq_gt<float, float>{}(query, row(get_key(*member)),
                                              dimensions);
    }
};

template <typename IndexType> class AlignedIndex final {

  public:
    using Allocator = aligned_allocator_gt<IndexType, 64>;

    explicit AlignedIndex(index_config_t const &configuration) {

        pointer_ = Allocator{}.allocate(1);

        new (pointer_) IndexType(configuration);
    }

    ~AlignedIndex() {

        if (pointer_ == nullptr) {

            return;
        }

        pointer_->~IndexType();

        Allocator{}.deallocate(pointer_, 1);
    }

    AlignedIndex(AlignedIndex const &) = delete;

    AlignedIndex &operator=(AlignedIndex const &) = delete;

    [[nodiscard]]
    IndexType &get() noexcept {

        return *pointer_;
    }

  private:
    IndexType *pointer_{nullptr};
};

class DisjointSet final {

  public:
    explicit DisjointSet(std::size_t count) : parent_(count), rank_(count, 0U) {

        for (std::size_t index = 0; index < count; ++index) {

            parent_[index] = static_cast<std::uint32_t>(index);
        }
    }

    std::uint32_t find(std::uint32_t value) {

        std::uint32_t &parent = parent_[value];

        if (parent != value) {

            parent = find(parent);
        }

        return parent;
    }

    bool unite(std::uint32_t first, std::uint32_t second) {

        first = find(first);

        second = find(second);

        if (first == second) {

            return false;
        }

        if (rank_[first] < rank_[second]) {

            std::swap(first, second);
        }

        parent_[second] = first;

        if (rank_[first] == rank_[second]) {

            ++rank_[first];
        }

        return true;
    }

  private:
    std::vector<std::uint32_t> parent_;

    std::vector<std::uint8_t> rank_;
};

class OverlaySearchContext final {

  public:
    explicit OverlaySearchContext(std::size_t node_count)
        : visited_epoch_(node_count, 0U) {

        candidates_.reserve(node_count);

        top_.reserve(node_count);
    }

    void begin_query() {

        ++epoch_;

        if (epoch_ == 0U) {

            std::fill(visited_epoch_.begin(), visited_epoch_.end(), 0U);

            epoch_ = 1U;
        }

        candidates_.clear();
        top_.clear();
    }

    [[nodiscard]]
    bool mark_if_new(std::uint32_t local) noexcept {

        std::uint32_t &value = visited_epoch_[local];

        if (value == epoch_) {

            return false;
        }

        value = epoch_;

        return true;
    }

    std::vector<OverlayCandidate> &candidates() noexcept { return candidates_; }

    std::vector<OverlayCandidate> &top() noexcept { return top_; }

  private:
    std::vector<std::uint32_t> visited_epoch_;

    std::uint32_t epoch_{};

    std::vector<OverlayCandidate> candidates_;

    std::vector<OverlayCandidate> top_;
};

struct BitsetPredicate final {

    std::uint32_t const *bitset{};

    [[nodiscard]]
    bool accepts_key(std::size_t key) const noexcept {

        return ((bitset[key >> 5U] >> (key & 31U)) & 1U) != 0U;
    }

    template <typename MemberLike>
    bool operator()(MemberLike &&member) const noexcept {

        return accepts_key(static_cast<std::size_t>(get_key(member)));
    }
};

void ensure_little_endian() {

    if constexpr (std::endian::native != std::endian::little) {

        throw std::runtime_error("G-REPAIR0 requiere little-endian.");
    }
}

double milliseconds_between(Clock::time_point first,
                            Clock::time_point second) noexcept {

    return std::chrono::duration<double, std::milli>(second - first).count();
}

void validate_file_size(std::filesystem::path const &path,
                        std::uintmax_t expected) {

    if (!std::filesystem::exists(path) ||
        !std::filesystem::is_regular_file(path)) {

        throw std::runtime_error("Archivo inexistente o invalido: " +
                                 path.string());
    }

    const std::uintmax_t actual = std::filesystem::file_size(path);

    if (actual != expected) {

        throw std::runtime_error("Tamaño inesperado: " +
                                 path.filename().string());
    }
}

template <typename T>
std::vector<T> read_binary_exact(std::filesystem::path const &path,
                                 std::size_t element_count) {

    std::ifstream input(path, std::ios::binary);

    if (!input) {

        throw std::runtime_error("No se pudo abrir: " + path.string());
    }

    std::vector<T> output(element_count);

    const std::size_t bytes = element_count * sizeof(T);

    input.read(reinterpret_cast<char *>(output.data()),
               static_cast<std::streamsize>(bytes));

    if (input.gcount() != static_cast<std::streamsize>(bytes)) {

        throw std::runtime_error("Lectura incompleta.");
    }

    return output;
}

std::vector<float> read_query_prefix(std::filesystem::path const &path) {

    std::ifstream input(path, std::ios::binary);

    if (!input) {

        throw std::runtime_error("No se pudo abrir queries.");
    }

    std::vector<float> output(kQueryCount * kDimensions);

    const std::size_t bytes = output.size() * sizeof(float);

    input.read(reinterpret_cast<char *>(output.data()),
               static_cast<std::streamsize>(bytes));

    if (input.gcount() != static_cast<std::streamsize>(bytes)) {

        throw std::runtime_error("Lectura incompleta de queries.");
    }

    return output;
}

std::size_t bitset_word_count_for(std::size_t count) noexcept {

    return (count + kBitsPerWord - 1U) / kBitsPerWord;
}

bool evaluate_filter(MetadataView const &metadata, std::size_t key,
                     FilterSpec const &filter) noexcept {

    const std::uint32_t category = metadata.category[key];

    const bool category_allowed =
        category == 3U || category == 7U || category == 9U;

    return metadata.tenant[key] == filter.tenant_required &&

           metadata.year[key] >= 2022U &&

           metadata.language[key] >= filter.language_min &&

           category_allowed &&

           metadata.security_level[key] <= 2U;
}

void materialize_bitset(MetadataView const &metadata, FilterSpec const &filter,
                        std::uint32_t *output) {

    const std::size_t word_count = bitset_word_count_for(kVectorCount);

    for (std::size_t word = 0; word < word_count; ++word) {

        std::uint32_t value = 0U;

        const std::size_t base = word * kBitsPerWord;

        for (std::size_t bit = 0; bit < kBitsPerWord; ++bit) {

            const std::size_t key = base + bit;

            if (key >= kVectorCount) {

                break;
            }

            if (evaluate_filter(metadata, key, filter)) {

                value |= (std::uint32_t{1U} << bit);
            }
        }

        output[word] = value;
    }
}

std::vector<std::uint32_t>
eligible_from_bitset(BitsetPredicate const &predicate) {

    std::vector<std::uint32_t> eligible;

    for (std::size_t key = 0; key < kVectorCount; ++key) {

        if (predicate.accepts_key(key)) {

            eligible.push_back(static_cast<std::uint32_t>(key));
        }
    }

    return eligible;
}

float l2_squared(float const *first, float const *second) noexcept {

    float sum = 0.0F;

    for (std::size_t dimension = 0; dimension < kDimensions; ++dimension) {

        const float delta = first[dimension] - second[dimension];

        sum += delta * delta;
    }

    return sum;
}

bool exact_candidate_less(ExactCandidate const &first,
                          ExactCandidate const &second) noexcept {

    if (first.distance != second.distance) {

        return first.distance < second.distance;
    }

    return first.key < second.key;
}

std::vector<std::uint32_t>
exact_top_k(DenseVectors const &base, float const *query,
            std::vector<std::uint32_t> const &eligible) {

    std::vector<ExactCandidate> candidates;

    candidates.reserve(eligible.size());

    for (std::uint32_t key : eligible) {

        candidates.push_back(
            ExactCandidate{.distance = l2_squared(query, base.row(key)),

                           .key = key});
    }

    if (candidates.size() < kNeighbors) {

        return {};
    }

    std::partial_sort(candidates.begin(),
                      candidates.begin() +
                          static_cast<std::ptrdiff_t>(kNeighbors),
                      candidates.end(), exact_candidate_less);

    std::vector<std::uint32_t> result;

    result.reserve(kNeighbors);

    for (std::size_t index = 0; index < kNeighbors; ++index) {

        result.push_back(candidates[index].key);
    }

    return result;
}

template <typename T>
T read_scalar(std::vector<std::uint8_t> const &bytes, std::size_t &offset) {

    if (offset + sizeof(T) > bytes.size()) {

        throw std::runtime_error("Snapshot HNSW truncado.");
    }

    T value{};

    std::memcpy(&value, bytes.data() + offset, sizeof(T));

    offset += sizeof(T);

    return value;
}

BaseTopologySnapshot extract_base_topology(Index const &index) {

    BaseTopologySnapshot output;

    const std::size_t expected_length = index.serialized_length();

    std::vector<std::uint8_t> serialized(expected_length);

    std::size_t written = 0U;

    auto serialization_result =
        index.save_to_stream([&](void *source, std::size_t length) noexcept {
            if (written + length > serialized.size()) {

                return false;
            }

            std::memcpy(serialized.data() + written, source, length);

            written += length;

            return true;
        });

    if (!serialization_result) {

        char const *message = serialization_result.error.release();

        throw std::runtime_error(
            message != nullptr ? message : "Fallo al serializar HNSW.");
    }

    if (written != expected_length) {

        throw std::runtime_error("Longitud serializada inconsistente.");
    }

    output.serialized_bytes = serialized.size();

    std::size_t offset = 0U;

    const index_serialized_header_t header =
        read_scalar<index_serialized_header_t>(serialized, offset);

    if (header.size != kVectorCount || header.connectivity != kConnectivity ||
        header.connectivity_base != kConnectivity * 2U ||
        header.entry_slot >= header.size) {

        throw std::runtime_error("Header serializado HNSW inesperado.");
    }

    std::vector<SerializedLevel> levels(header.size);

    for (std::size_t slot = 0; slot < header.size; ++slot) {

        levels[slot] = read_scalar<SerializedLevel>(serialized, offset);

        if (levels[slot] < 0) {

            throw std::runtime_error("Nivel HNSW negativo.");
        }
    }

    output.row_offsets.resize(header.size + 1U);

    output.keys_are_identity = true;

    for (std::size_t slot = 0; slot < header.size; ++slot) {

        output.row_offsets[slot] =
            static_cast<std::uint32_t>(output.neighbors.size());

        const VectorKey key = read_scalar<VectorKey>(serialized, offset);

        const SerializedLevel stored_level =
            read_scalar<SerializedLevel>(serialized, offset);

        if (stored_level != levels[slot]) {

            throw std::runtime_error("Nivel duplicado inconsistente.");
        }

        if (key != slot) {

            output.keys_are_identity = false;
        }

        const SerializedNeighborsCount base_count =
            read_scalar<SerializedNeighborsCount>(serialized, offset);

        if (base_count > header.connectivity_base) {

            throw std::runtime_error("Grado L0 mayor que M0.");
        }

        for (std::size_t index = 0; index < header.connectivity_base; ++index) {

            const Slot neighbor = read_scalar<Slot>(serialized, offset);

            if (index < base_count) {

                if (neighbor >= header.size) {

                    throw std::runtime_error("Neighbor L0 fuera de rango.");
                }

                output.neighbors.push_back(
                    static_cast<std::uint32_t>(neighbor));
            }
        }

        for (SerializedLevel level = 1; level <= stored_level; ++level) {

            const SerializedNeighborsCount count =
                read_scalar<SerializedNeighborsCount>(serialized, offset);

            if (count > header.connectivity) {

                throw std::runtime_error("Grado superior mayor que M.");
            }

            for (std::size_t index = 0; index < header.connectivity; ++index) {

                static_cast<void>(read_scalar<Slot>(serialized, offset));
            }
        }
    }

    output.row_offsets[header.size] =
        static_cast<std::uint32_t>(output.neighbors.size());

    if (offset != serialized.size()) {

        throw std::runtime_error("Snapshot HNSW contiene bytes residuales.");
    }

    if (!output.keys_are_identity) {

        throw std::runtime_error("G-REPAIR0 exige key == slot.");
    }

    output.entry_slot = static_cast<std::uint32_t>(header.entry_slot);

    output.valid = true;

    return output;
}

void normalize_adjacency(std::vector<std::vector<std::uint32_t>> &adjacency) {

    for (auto &row : adjacency) {

        std::sort(row.begin(), row.end());

        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
}

void add_undirected_edge(std::vector<std::vector<std::uint32_t>> &adjacency,
                         std::uint32_t first, std::uint32_t second) {

    if (first == second) {

        return;
    }

    adjacency[first].push_back(second);

    adjacency[second].push_back(first);
}

std::size_t
directed_edge_count(std::vector<std::vector<std::uint32_t>> const &adjacency) {

    std::size_t total = 0U;

    for (auto const &row : adjacency) {

        total += row.size();
    }

    return total;
}

std::pair<std::size_t, std::vector<std::uint32_t>>
component_labels(std::vector<std::vector<std::uint32_t>> const &adjacency) {

    const std::size_t count = adjacency.size();

    std::vector<std::uint32_t> labels(
        count, std::numeric_limits<std::uint32_t>::max());

    std::vector<std::uint32_t> stack;

    stack.reserve(count);

    std::uint32_t component = 0U;

    for (std::size_t start = 0; start < count; ++start) {

        if (labels[start] != std::numeric_limits<std::uint32_t>::max()) {

            continue;
        }

        labels[start] = component;

        stack.clear();

        stack.push_back(static_cast<std::uint32_t>(start));

        while (!stack.empty()) {

            const std::uint32_t current = stack.back();

            stack.pop_back();

            for (std::uint32_t neighbor : adjacency[current]) {

                if (labels[neighbor] ==
                    std::numeric_limits<std::uint32_t>::max()) {

                    labels[neighbor] = component;

                    stack.push_back(neighbor);
                }
            }
        }

        ++component;
    }

    return {component, std::move(labels)};
}

PersistentOverlay
build_persistent_overlay(BaseTopologySnapshot const &topology,
                         DenseVectors const &base,
                         std::vector<std::uint32_t> const &eligible) {

    PersistentOverlay overlay;

    overlay.global_ids = eligible;

    if (eligible.empty()) {

        throw std::runtime_error("No existen nodos elegibles.");
    }

    std::vector<std::int32_t> local_of_global(kVectorCount, -1);

    for (std::size_t local = 0; local < eligible.size(); ++local) {

        local_of_global[eligible[local]] = static_cast<std::int32_t>(local);
    }

    std::vector<std::vector<std::uint32_t>> adjacency(eligible.size());

    //
    // 1. Subgrafo inducido L0.
    //

    for (std::size_t local = 0; local < eligible.size(); ++local) {

        const std::uint32_t global = eligible[local];

        const std::uint32_t begin = topology.row_offsets[global];

        const std::uint32_t end = topology.row_offsets[global + 1U];

        for (std::uint32_t offset = begin; offset < end; ++offset) {

            const std::uint32_t neighbor_global = topology.neighbors[offset];

            const std::int32_t neighbor_local =
                local_of_global[neighbor_global];

            if (neighbor_local >= 0) {

                add_undirected_edge(adjacency,
                                    static_cast<std::uint32_t>(local),
                                    static_cast<std::uint32_t>(neighbor_local));
            }
        }
    }

    normalize_adjacency(adjacency);

    overlay.induced_undirected_edges = directed_edge_count(adjacency) / 2U;

    const auto induced_components = component_labels(adjacency);

    overlay.components_induced = induced_components.first;

    //
    // 2. Contracción persistente de caminos 2-hop:
    //
    //      elegible -> fallido -> elegible.
    //
    // Cada nodo propone como máximo B shortcuts.
    //

    std::vector<std::pair<std::uint32_t, std::uint32_t>> selected_shortcuts;

    for (std::size_t local = 0; local < eligible.size(); ++local) {

        const std::uint32_t global = eligible[local];

        std::vector<std::uint8_t> seen(eligible.size(), 0U);

        seen[local] = 1U;

        for (std::uint32_t direct : adjacency[local]) {

            seen[direct] = 1U;
        }

        std::vector<ShortcutCandidate> candidates;

        const std::uint32_t first_begin = topology.row_offsets[global];

        const std::uint32_t first_end = topology.row_offsets[global + 1U];

        for (std::uint32_t first_offset = first_begin; first_offset < first_end;
             ++first_offset) {

            const std::uint32_t bridge_global =
                topology.neighbors[first_offset];

            //
            // El intermedio debe fallar P.
            //

            if (local_of_global[bridge_global] >= 0) {

                continue;
            }

            const std::uint32_t second_begin =
                topology.row_offsets[bridge_global];

            const std::uint32_t second_end =
                topology.row_offsets[bridge_global + 1U];

            for (std::uint32_t second_offset = second_begin;
                 second_offset < second_end; ++second_offset) {

                const std::uint32_t target_global =
                    topology.neighbors[second_offset];

                const std::int32_t target_local =
                    local_of_global[target_global];

                if (target_local < 0) {

                    continue;
                }

                const std::uint32_t target =
                    static_cast<std::uint32_t>(target_local);

                if (seen[target] != 0U) {

                    continue;
                }

                seen[target] = 1U;

                candidates.push_back(ShortcutCandidate{
                    .distance =
                        l2_squared(base.row(global), base.row(target_global)),

                    .local = target});
            }
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](ShortcutCandidate const &first,
                     ShortcutCandidate const &second) {
                      if (first.distance != second.distance) {

                          return first.distance < second.distance;
                      }

                      return first.local < second.local;
                  });

        const std::size_t selected_count =
            std::min(kShortcutProposalBudget, candidates.size());

        for (std::size_t index = 0; index < selected_count; ++index) {

            selected_shortcuts.emplace_back(static_cast<std::uint32_t>(local),
                                            candidates[index].local);
        }
    }

    const std::size_t before_shortcuts = overlay.induced_undirected_edges;

    for (auto const &edge : selected_shortcuts) {

        add_undirected_edge(adjacency, edge.first, edge.second);
    }

    normalize_adjacency(adjacency);

    const std::size_t after_shortcuts = directed_edge_count(adjacency) / 2U;

    overlay.two_hop_shortcut_edges = after_shortcuts - before_shortcuts;

    const auto after_two_hop_components = component_labels(adjacency);

    overlay.components_after_two_hop = after_two_hop_components.first;

    //
    // 3. Reparación de conectividad residual.
    //
    // Si el overlay sigue fragmentado, se construye un MST
    // entre componentes usando la menor distancia L2 disponible.
    //

    if (after_two_hop_components.first > 1U) {

        std::vector<ComponentEdge> component_edges;

        const auto &labels = after_two_hop_components.second;

        for (std::size_t first = 0; first < eligible.size(); ++first) {

            for (std::size_t second = first + 1U; second < eligible.size();
                 ++second) {

                if (labels[first] == labels[second]) {

                    continue;
                }

                component_edges.push_back(ComponentEdge{
                    .distance = l2_squared(base.row(eligible[first]),
                                           base.row(eligible[second])),

                    .first = static_cast<std::uint32_t>(first),

                    .second = static_cast<std::uint32_t>(second),

                    .first_component = labels[first],

                    .second_component = labels[second]});
            }
        }

        std::sort(component_edges.begin(), component_edges.end(),
                  [](ComponentEdge const &first, ComponentEdge const &second) {
                      if (first.distance != second.distance) {

                          return first.distance < second.distance;
                      }

                      if (first.first != second.first) {

                          return first.first < second.first;
                      }

                      return first.second < second.second;
                  });

        DisjointSet components(after_two_hop_components.first);

        std::size_t remaining = after_two_hop_components.first;

        for (ComponentEdge const &edge : component_edges) {

            if (components.unite(edge.first_component, edge.second_component)) {

                add_undirected_edge(adjacency, edge.first, edge.second);

                ++overlay.completion_edges;

                --remaining;

                if (remaining == 1U) {

                    break;
                }
            }
        }
    }

    normalize_adjacency(adjacency);

    const auto final_components = component_labels(adjacency);

    overlay.components_final = final_components.first;

    overlay.final_undirected_edges = directed_edge_count(adjacency) / 2U;

    //
    // Entry point persistente:
    //
    // nodo elegible más cercano al entry point original HNSW.
    //

    float best_entry_distance = std::numeric_limits<float>::max();

    std::uint32_t best_entry = 0U;

    for (std::size_t local = 0; local < eligible.size(); ++local) {

        const float distance = l2_squared(base.row(topology.entry_slot),
                                          base.row(eligible[local]));

        if (distance < best_entry_distance ||
            (distance == best_entry_distance && local < best_entry)) {

            best_entry_distance = distance;

            best_entry = static_cast<std::uint32_t>(local);
        }
    }

    overlay.entry_local = best_entry;

    //
    // CSR compacto final.
    //

    overlay.row_offsets.resize(eligible.size() + 1U);

    for (std::size_t local = 0; local < adjacency.size(); ++local) {

        overlay.row_offsets[local] =
            static_cast<std::uint32_t>(overlay.neighbors.size());

        for (std::uint32_t neighbor : adjacency[local]) {

            overlay.neighbors.push_back(neighbor);
        }
    }

    overlay.row_offsets[eligible.size()] =
        static_cast<std::uint32_t>(overlay.neighbors.size());

    return overlay;
}

bool candidate_ascending(OverlayCandidate const &first,
                         OverlayCandidate const &second) noexcept {

    if (first.distance != second.distance) {

        return first.distance < second.distance;
    }

    return first.local < second.local;
}

struct CandidateGreater final {

    bool operator()(OverlayCandidate const &first,
                    OverlayCandidate const &second) const noexcept {

        return candidate_ascending(second, first);
    }
};

struct CandidateLess final {

    bool operator()(OverlayCandidate const &first,
                    OverlayCandidate const &second) const noexcept {

        return candidate_ascending(first, second);
    }
};

OverlaySearchResult search_overlay(PersistentOverlay const &overlay,
                                   DenseVectors const &base, float const *query,
                                   OverlaySearchContext &context) {

    OverlaySearchResult output;

    if (overlay.global_ids.empty()) {

        return output;
    }

    context.begin_query();

    auto &candidates = context.candidates();

    auto &top = context.top();

    const std::uint32_t start = overlay.entry_local;

    const float start_distance =
        l2_squared(query, base.row(overlay.global_ids[start]));

    ++output.computed_distances;
    ++output.visited_members;

    context.mark_if_new(start);

    candidates.push_back(OverlayCandidate{.distance = start_distance,

                                          .local = start});

    std::push_heap(candidates.begin(), candidates.end(), CandidateGreater{});

    top.push_back(OverlayCandidate{.distance = start_distance,

                                   .local = start});

    std::push_heap(top.begin(), top.end(), CandidateLess{});

    while (!candidates.empty()) {

        std::pop_heap(candidates.begin(), candidates.end(), CandidateGreater{});

        const OverlayCandidate current = candidates.back();

        candidates.pop_back();

        if (top.size() >= kPersistentEfSearch &&
            current.distance > top.front().distance) {

            break;
        }

        const std::uint32_t begin = overlay.row_offsets[current.local];

        const std::uint32_t end = overlay.row_offsets[current.local + 1U];

        for (std::uint32_t offset = begin; offset < end; ++offset) {

            const std::uint32_t neighbor = overlay.neighbors[offset];

            if (!context.mark_if_new(neighbor)) {

                continue;
            }

            ++output.visited_members;

            const float distance =
                l2_squared(query, base.row(overlay.global_ids[neighbor]));

            ++output.computed_distances;

            if (top.size() < kPersistentEfSearch ||
                distance < top.front().distance) {

                candidates.push_back(OverlayCandidate{.distance = distance,

                                                      .local = neighbor});

                std::push_heap(candidates.begin(), candidates.end(),
                               CandidateGreater{});

                top.push_back(OverlayCandidate{.distance = distance,

                                               .local = neighbor});

                std::push_heap(top.begin(), top.end(), CandidateLess{});

                if (top.size() > kPersistentEfSearch) {

                    std::pop_heap(top.begin(), top.end(), CandidateLess{});

                    top.pop_back();
                }
            }
        }
    }

    std::sort(top.begin(), top.end(), candidate_ascending);

    output.count = std::min(kNeighbors, top.size());

    for (std::size_t index = 0; index < output.count; ++index) {

        output.keys[index] = overlay.global_ids[top[index].local];
    }

    return output;
}

index_search_config_t make_racorn_configuration() {

    index_search_config_t configuration;

    configuration.expansion = kEfSearch;

    configuration.thread = 0;

    configuration.two_hop_threshold = kTwoHopThreshold;

    configuration.racorn = true;

    configuration.bridge_budget_ratio = kBridgeBudgetRatio;

    configuration.three_hop_threshold = -1.0;

    configuration.fallback_exact_threshold = 0.0;

    configuration.racorn_stride_sampling = true;

    return configuration;
}

RacornSnapshot run_racorn(Index &index, float const *query,
                          L2Metric const &metric,
                          BitsetPredicate const &predicate) {

    auto configuration = make_racorn_configuration();

    auto result =
        index.search(query, kNeighbors, metric, configuration, predicate);

    RacornSnapshot output;

    output.keys.reserve(result.size());

    for (std::size_t index = 0; index < result.size(); ++index) {

        output.keys.push_back(
            static_cast<std::uint32_t>(result[index].member.key));
    }

    output.computed_distances = result.computed_distances;

    output.visited_members = result.visited_members;

    output.predicate_evaluations = result.predicate_evaluations;

    output.fallback_count = result.fallback_count;

    return output;
}

void run_racorn_sink(Index &index, float const *query, L2Metric const &metric,
                     BitsetPredicate const &predicate, std::size_t &sink) {

    auto configuration = make_racorn_configuration();

    auto result =
        index.search(query, kNeighbors, metric, configuration, predicate);

    sink ^= result.size() + result.computed_distances + result.visited_members +
            result.predicate_evaluations + result.fallback_count;
}

double recall_at_k(std::vector<std::uint32_t> const &approximate,
                   std::vector<std::uint32_t> const &exact) {

    if (exact.empty()) {

        return 0.0;
    }

    std::size_t hits = 0U;

    for (std::uint32_t key : approximate) {

        if (std::find(exact.begin(), exact.end(), key) != exact.end()) {

            ++hits;
        }
    }

    return static_cast<double>(hits) / static_cast<double>(exact.size());
}

double recall_at_k(OverlaySearchResult const &approximate,
                   std::vector<std::uint32_t> const &exact) {

    if (exact.empty()) {

        return 0.0;
    }

    std::size_t hits = 0U;

    for (std::size_t index = 0; index < approximate.count; ++index) {

        if (std::find(exact.begin(), exact.end(), approximate.keys[index]) !=
            exact.end()) {

            ++hits;
        }
    }

    return static_cast<double>(hits) / static_cast<double>(exact.size());
}

double mean(std::vector<double> const &values) {

    if (values.empty()) {

        return 0.0;
    }

    double sum = 0.0;

    for (double value : values) {

        sum += value;
    }

    return sum / static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double probability) {

    if (values.empty()) {

        return 0.0;
    }

    std::sort(values.begin(), values.end());

    if (values.size() == 1U) {

        return values.front();
    }

    const double position =
        probability * static_cast<double>(values.size() - 1U);

    const std::size_t lower = static_cast<std::size_t>(std::floor(position));

    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));

    if (lower == upper) {

        return values[lower];
    }

    const double fraction = position - static_cast<double>(lower);

    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

void fill_latency_stats(RepairLatencyStats &output,
                        std::vector<double> const &samples) {

    output.mean_ms = mean(samples);

    output.p50_ms = percentile(samples, 0.50);

    output.p95_ms = percentile(samples, 0.95);

    output.p99_ms = percentile(samples, 0.99);

    if (output.mean_ms > 0.0) {

        output.qps_mean = 1000.0 / output.mean_ms;
    }
}

void print_progress(std::string const &message) {

    std::cout << message << '\n' << std::flush;
}

} // namespace

PersistentRepairBenchmarkResult
PersistentRepairBenchmark::run(std::filesystem::path const &data_directory) {

    ensure_little_endian();

    PersistentRepairBenchmarkResult output;

    output.data_directory = data_directory;

    output.vector_count = kVectorCount;

    output.dimensions = kDimensions;

    output.query_count = kQueryCount;

    output.k = kNeighbors;

    output.connectivity = kConnectivity;

    output.connectivity_base = kConnectivity * 2U;

    output.ef_construction = kEfConstruction;

    output.ef_search = kEfSearch;

    output.persistent_ef_search = kPersistentEfSearch;

    output.bridge_budget_ratio = kBridgeBudgetRatio;

    output.two_hop_threshold = kTwoHopThreshold;

    output.tenant_required = kTenantRequired;

    output.language_min = kLanguageMin;

    output.shortcut_proposal_budget = kShortcutProposalBudget;

    output.warmup_rounds = kWarmupRounds;

    output.measured_rounds = kMeasuredRounds;

    output.racorn_modified = false;

    const auto base_path = data_directory / kBaseFilename;

    const auto query_path = data_directory / kQueryFilename;

    const auto metadata_path = data_directory / kMetadataFilename;

    print_progress("[G-REPAIR0] Validando archivos...");

    validate_file_size(base_path, static_cast<std::uintmax_t>(kVectorCount) *
                                      kDimensions * sizeof(float));

    output.base_file_valid = true;

    validate_file_size(query_path,
                       static_cast<std::uintmax_t>(kStoredQueryCount) *
                           kDimensions * sizeof(float));

    output.query_file_valid = true;

    validate_file_size(metadata_path,
                       static_cast<std::uintmax_t>(kMetadataColumnCount) *
                           kVectorCount * sizeof(std::uint32_t));

    output.metadata_file_valid = true;

    print_progress("[G-REPAIR0] Cargando SIFT100K...");

    DenseVectors base(
        read_binary_exact<float>(base_path, kVectorCount * kDimensions),
        kDimensions);

    print_progress("[G-REPAIR0] Cargando 1000 queries...");

    DenseVectors queries(read_query_prefix(query_path), kDimensions);

    print_progress("[G-REPAIR0] Cargando metadata...");

    auto metadata_storage = read_binary_exact<std::uint32_t>(
        metadata_path, kMetadataColumnCount * kVectorCount);

    const MetadataView metadata{
        .tenant = metadata_storage.data() + 0U * kVectorCount,

        .category = metadata_storage.data() + 1U * kVectorCount,

        .year = metadata_storage.data() + 2U * kVectorCount,

        .language = metadata_storage.data() + 3U * kVectorCount,

        .security_level = metadata_storage.data() + 4U * kVectorCount};

    const FilterSpec filter{.tenant_required = kTenantRequired,

                            .language_min = kLanguageMin};

    //
    // Predicado común materializado una sola vez.
    // Su coste es común a ambos planes y no entra en R*.
    //

    std::vector<std::uint32_t> predicate_bitset(
        bitset_word_count_for(kVectorCount));

    materialize_bitset(metadata, filter, predicate_bitset.data());

    output.predicate_bitset_bytes =
        predicate_bitset.size() * sizeof(std::uint32_t);

    const BitsetPredicate predicate{.bitset = predicate_bitset.data()};

    const auto eligible = eligible_from_bitset(predicate);

    output.eligible_count = eligible.size();

    output.selectivity = static_cast<double>(eligible.size()) /
                         static_cast<double>(kVectorCount);

    if (eligible.size() < kNeighbors) {

        throw std::runtime_error("Predicado produce menos de k elegibles.");
    }

    //
    // HNSW base.
    //

    print_progress("[G-REPAIR0] Construyendo HNSW base...");

    L2Metric metric{.vectors = &base,

                    .dimensions = kDimensions};

    index_config_t index_configuration(kConnectivity);

    AlignedIndex<Index> aligned_index(index_configuration);

    Index &index = aligned_index.get();

    index.reserve(index_limits_t(kVectorCount, 1));

    index_update_config_t update_configuration;

    update_configuration.expansion = kEfConstruction;

    update_configuration.thread = 0;

    constexpr std::size_t progress_step = kVectorCount / 10U;

    for (std::size_t key = 0; key < kVectorCount; ++key) {

        index.add(static_cast<VectorKey>(key), base.row(key), metric,
                  update_configuration);

        if ((key + 1U) % progress_step == 0U) {

            std::cout << "[G-REPAIR0] HNSW "
                      << ((key + 1U) * 100U / kVectorCount) << "%\n"
                      << std::flush;
        }
    }

    //
    // Snapshot topológico público.
    //

    print_progress("[G-REPAIR0] Extrayendo snapshot L0...");

    const auto snapshot_start = Clock::now();

    const BaseTopologySnapshot topology = extract_base_topology(index);

    const auto snapshot_end = Clock::now();

    output.topology_snapshot_ms =
        milliseconds_between(snapshot_start, snapshot_end);

    output.serialized_index_bytes = topology.serialized_bytes;

    output.parsed_l0_directed_edges = topology.neighbors.size();

    const auto l0_stats = index.stats(0U);

    output.index_l0_directed_edges = l0_stats.edges;

    output.topology_snapshot_valid =
        topology.valid && topology.keys_are_identity &&
        output.parsed_l0_directed_edges == output.index_l0_directed_edges;

    if (!output.topology_snapshot_valid) {

        throw std::runtime_error("Snapshot L0 no coincide con stats(0).");
    }

    //
    // Construcción one-shot de reparación persistente.
    //

    print_progress("[G-REPAIR0] Construyendo reparación persistente...");

    const auto repair_start = Clock::now();

    const PersistentOverlay overlay =
        build_persistent_overlay(topology, base, eligible);

    const auto repair_end = Clock::now();

    output.repair_build_ms = milliseconds_between(repair_start, repair_end);

    output.total_persistent_build_ms =
        output.topology_snapshot_ms + output.repair_build_ms;

    output.induced_undirected_edges = overlay.induced_undirected_edges;

    output.two_hop_shortcut_edges = overlay.two_hop_shortcut_edges;

    output.completion_edges = overlay.completion_edges;

    output.final_undirected_edges = overlay.final_undirected_edges;

    output.components_induced = overlay.components_induced;

    output.components_after_two_hop = overlay.components_after_two_hop;

    output.components_final = overlay.components_final;

    output.overlay_connected = overlay.components_final == 1U;

    output.overlay_logical_bytes = overlay.logical_bytes();

    output.overlay_allocated_bytes = overlay.allocated_bytes();

    if (!output.overlay_connected) {

        throw std::runtime_error(
            "La reparación persistente no quedó conectada.");
    }

    //
    // Ground truth exacto.
    //

    print_progress("[G-REPAIR0] Construyendo ground truth...");

    std::vector<std::vector<std::uint32_t>> ground_truth;

    ground_truth.reserve(kQueryCount);

    output.ground_truth_complete = true;

    for (std::size_t query_index = 0; query_index < kQueryCount;
         ++query_index) {

        auto exact = exact_top_k(base, queries.row(query_index), eligible);

        if (exact.size() != kNeighbors) {

            output.ground_truth_complete = false;
        }

        ground_truth.push_back(std::move(exact));

        if ((query_index + 1U) % 100U == 0U) {

            std::cout << "[G-REPAIR0] GT " << (query_index + 1U) << "/"
                      << kQueryCount << '\n'
                      << std::flush;
        }
    }

    //
    // Validación de calidad y trabajo.
    //

    print_progress("[G-REPAIR0] Validando Transient vs Persistent...");

    OverlaySearchContext overlay_context(overlay.global_ids.size());

    double transient_recall_sum = 0.0;

    double persistent_recall_sum = 0.0;

    double transient_min_recall = 1.0;

    double persistent_min_recall = 1.0;

    double transient_distances_sum = 0.0;

    double transient_visited_sum = 0.0;

    double transient_predicates_sum = 0.0;

    double transient_fallback_sum = 0.0;

    double persistent_distances_sum = 0.0;

    double persistent_visited_sum = 0.0;

    for (std::size_t query_index = 0; query_index < kQueryCount;
         ++query_index) {

        const RacornSnapshot transient =
            run_racorn(index, queries.row(query_index), metric, predicate);

        const OverlaySearchResult persistent = search_overlay(
            overlay, base, queries.row(query_index), overlay_context);

        const double transient_recall =
            recall_at_k(transient.keys, ground_truth[query_index]);

        const double persistent_recall =
            recall_at_k(persistent, ground_truth[query_index]);

        transient_recall_sum += transient_recall;

        persistent_recall_sum += persistent_recall;

        transient_min_recall = std::min(transient_min_recall, transient_recall);

        persistent_min_recall =
            std::min(persistent_min_recall, persistent_recall);

        if (transient.keys.size() == kNeighbors &&
            std::abs(transient_recall - 1.0) <= 1e-12) {

            ++output.transient_perfect_queries;
        }

        if (persistent.count == kNeighbors &&
            std::abs(persistent_recall - 1.0) <= 1e-12) {

            ++output.persistent_perfect_queries;
        }

        if (transient.keys.size() < kNeighbors) {

            ++output.transient_underfilled_queries;
        }

        if (persistent.count < kNeighbors) {

            ++output.persistent_underfilled_queries;
        }

        for (std::uint32_t key : transient.keys) {

            if (!predicate.accepts_key(key)) {

                ++output.invalid_transient_results;

                break;
            }
        }

        for (std::size_t index_result = 0; index_result < persistent.count;
             ++index_result) {

            if (!predicate.accepts_key(persistent.keys[index_result])) {

                ++output.invalid_persistent_results;

                break;
            }
        }

        transient_distances_sum +=
            static_cast<double>(transient.computed_distances);

        transient_visited_sum += static_cast<double>(transient.visited_members);

        transient_predicates_sum +=
            static_cast<double>(transient.predicate_evaluations);

        transient_fallback_sum += static_cast<double>(transient.fallback_count);

        persistent_distances_sum +=
            static_cast<double>(persistent.computed_distances);

        persistent_visited_sum +=
            static_cast<double>(persistent.visited_members);
    }

    output.transient_mean_recall =
        transient_recall_sum / static_cast<double>(kQueryCount);

    output.persistent_mean_recall =
        persistent_recall_sum / static_cast<double>(kQueryCount);

    output.transient_min_recall = transient_min_recall;

    output.persistent_min_recall = persistent_min_recall;

    output.transient.mean_computed_distances =
        transient_distances_sum / static_cast<double>(kQueryCount);

    output.transient.mean_visited_members =
        transient_visited_sum / static_cast<double>(kQueryCount);

    output.transient_mean_predicate_evaluations =
        transient_predicates_sum / static_cast<double>(kQueryCount);

    output.transient_mean_fallback_count =
        transient_fallback_sum / static_cast<double>(kQueryCount);

    output.persistent.mean_computed_distances =
        persistent_distances_sum / static_cast<double>(kQueryCount);

    output.persistent.mean_visited_members =
        persistent_visited_sum / static_cast<double>(kQueryCount);

    //
    // Criterio de calidad definido antes del benchmark.
    //

    const double minimum_admissible_recall =
        std::max(kRecallFloor,
                 output.transient_mean_recall - kRecallToleranceVsTransient);

    output.persistent_quality_admissible =
        output.persistent_mean_recall >= minimum_admissible_recall &&
        output.persistent_underfilled_queries == 0U;

    //
    // Warm-up.
    //

    print_progress("[G-REPAIR0] Warm-up...");

    for (std::size_t round = 0; round < kWarmupRounds; ++round) {

        for (std::size_t query_index = 0; query_index < kQueryCount;
             ++query_index) {

            run_racorn_sink(index, queries.row(query_index), metric, predicate,
                            output.benchmark_sink);

            const auto persistent = search_overlay(
                overlay, base, queries.row(query_index), overlay_context);

            output.benchmark_sink ^= persistent.count +
                                     persistent.computed_distances +
                                     persistent.visited_members;

            if (persistent.count > 0U) {

                output.benchmark_sink ^= persistent.keys[0];
            }
        }
    }

    //
    // Benchmark emparejado.
    //

    print_progress(
        "[G-REPAIR0] Midiendo reparación transitoria vs persistente...");

    std::vector<double> transient_samples;

    std::vector<double> persistent_samples;

    transient_samples.reserve(kQueryCount * kMeasuredRounds);

    persistent_samples.reserve(kQueryCount * kMeasuredRounds);

    for (std::size_t round = 0; round < kMeasuredRounds; ++round) {

        std::cout << "[G-REPAIR0] Ronda " << (round + 1U) << "/"
                  << kMeasuredRounds << '\n'
                  << std::flush;

        for (std::size_t query_index = 0; query_index < kQueryCount;
             ++query_index) {

            auto measure_transient = [&]() {
                const auto start = Clock::now();

                run_racorn_sink(index, queries.row(query_index), metric,
                                predicate, output.benchmark_sink);

                const auto end = Clock::now();

                transient_samples.push_back(milliseconds_between(start, end));
            };

            auto measure_persistent = [&]() {
                const auto start = Clock::now();

                const auto result = search_overlay(
                    overlay, base, queries.row(query_index), overlay_context);

                const auto end = Clock::now();

                persistent_samples.push_back(milliseconds_between(start, end));

                output.benchmark_sink ^= result.count +
                                         result.computed_distances +
                                         result.visited_members;

                if (result.count > 0U) {

                    output.benchmark_sink ^= result.keys[0];
                }
            };

            if ((round + query_index) % 2U == 0U) {

                measure_transient();
                measure_persistent();

            } else {

                measure_persistent();
                measure_transient();
            }
        }
    }

    fill_latency_stats(output.transient, transient_samples);

    fill_latency_stats(output.persistent, persistent_samples);

    output.persistent_query_faster =
        output.persistent.mean_ms < output.transient.mean_ms;

    if (output.persistent.mean_ms > 0.0) {

        output.persistent_speedup_vs_transient =
            output.transient.mean_ms / output.persistent.mean_ms;
    }

    if (output.persistent_query_faster) {

        const double saving =
            output.transient.mean_ms - output.persistent.mean_ms;

        output.break_even_exists = saving > 0.0;

        if (output.break_even_exists) {

            output.break_even_total_repetitions =
                output.total_persistent_build_ms / saving;

            output.break_even_repair_only_repetitions =
                output.repair_build_ms / saving;
        }
    }

    for (std::size_t repetitions : kRepetitionPoints) {

        const double transient_cost =
            static_cast<double>(repetitions) * output.transient.mean_ms;

        const double persistent_cost =
            output.total_persistent_build_ms +
            static_cast<double>(repetitions) * output.persistent.mean_ms;

        output.cumulative_costs.push_back(RepairCostPoint{
            .repetitions = repetitions,

            .transient_cumulative_ms = transient_cost,

            .persistent_cumulative_ms = persistent_cost,

            .winner = persistent_cost < transient_cost ? "Persistent-Repair"
                                                       : "RACORN-Transient"});
    }

    if (!output.persistent_quality_admissible) {

        output.decision = "STOP_PERSISTENT_REPAIR_QUALITY";

    } else if (!output.persistent_query_faster) {

        output.decision = "STOP_PERSISTENT_REPAIR_LATENCY";

    } else {

        output.decision = "PERSISTENT_REPAIR_HAS_BREAK_EVEN";
    }

    output.valid = output.base_file_valid && output.query_file_valid &&
                   output.metadata_file_valid &&
                   output.topology_snapshot_valid &&
                   output.ground_truth_complete && output.overlay_connected &&
                   output.invalid_transient_results == 0U &&
                   output.invalid_persistent_results == 0U &&
                   transient_samples.size() == kQueryCount * kMeasuredRounds &&
                   persistent_samples.size() == kQueryCount * kMeasuredRounds &&
                   !output.racorn_modified;

    print_progress("[G-REPAIR0] Experimento completado.");

    return output;
}

} // namespace racorn_uma::core

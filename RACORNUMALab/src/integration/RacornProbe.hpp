//
//  RacornProbe.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir la primera frontera explícita entre RACORNUMALab y el artefacto
//  científico RACORN-1.
//
//  Gate:
//  G2 — Proveniencia e integración de headers RACORN.
//
//  Responsabilidades:
//  - Exponer la procedencia exacta del código RACORN utilizado.
//  - Permitir comprobar que los headers RACORN/USEARCH son aceptados por
//    Apple Clang arm64.
//  - Mantener aislado el código de terceros del resto del sistema.
//
//  Invariantes:
//  - No construye índices.
//  - No realiza búsquedas.
//  - No asigna memoria RACORN.
//  - No modifica ninguna estructura upstream.
//  - No ejecuta código Metal.
//
//  Seguridad:
//  - Esta unidad es exclusivamente de diagnóstico.
//  - No accede a red, archivos ni recursos externos durante ejecución.
//

#pragma once

#include <string_view>

namespace racorn_uma::integration {

struct RacornProbe final {

    [[nodiscard]]
    static constexpr std::string_view version() noexcept {
        return "1.0.0";
    }

    [[nodiscard]]
    static constexpr std::string_view commit() noexcept {
        return "04223a28e09d46e7e7708038a0083bdd34731edd";
    }

    [[nodiscard]]
    static constexpr std::string_view repository() noexcept {
        return "naver/racorn";
    }

    [[nodiscard]]
    static bool headers_available() noexcept;
};

} // namespace racorn_uma::integration

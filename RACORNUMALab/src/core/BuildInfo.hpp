//
//  BuildInfo.hpp
//  RACORNUMALab
//
//  Propósito:
//  Proporcionar información reproducible sobre el entorno de compilación
//  utilizado para construir el motor RACORNUMALab.
//
//  Responsabilidades:
//  - Identificar el compilador.
//  - Identificar la arquitectura objetivo.
//  - Identificar el estándar C++.
//  - Identificar el modo de construcción.
//
//  Invariantes:
//  - No realiza asignaciones globales.
//  - No depende de RACORN, USEARCH, Metal ni frameworks externos.
//  - Toda la información procede del toolchain utilizado durante build.
//
//  Seguridad:
//  - No accede al sistema de archivos.
//  - No ejecuta procesos externos.
//  - No modifica estado global.
//

#pragma once

#include <string>

namespace racorn_uma::core {

struct BuildInfo final {

    [[nodiscard]]
    static std::string compiler();

    [[nodiscard]]
    static std::string architecture();

    [[nodiscard]]
    static std::string cpp_standard();

    [[nodiscard]]
    static std::string build_mode();
};

} // namespace racorn_uma::core

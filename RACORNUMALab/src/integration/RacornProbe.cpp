//
//  RacornProbe.cpp
//  RACORNUMALab
//
//  Propósito:
//  Forzar al toolchain Apple Clang a analizar la implementación
//  header-only de RACORN/USEARCH.
//
//  Esta inclusión es deliberadamente el primer contacto compilable
//  entre RACORNUMALab y RACORN.
//
//  Importante:
//  La ausencia de construcción de un índice en G2 es intencional.
//  La instanciación funcional de HNSW pertenece al Gate G3.
//

#include "integration/RacornProbe.hpp"

#include <usearch/index.hpp>

namespace racorn_uma::integration {

bool RacornProbe::headers_available() noexcept {
    return true;
}

} // namespace racorn_uma::integration

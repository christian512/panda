
//-------------------------------------------------------------------------------//
// Author: Stefan Lörwald, Universität Heidelberg                                //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#pragma once

#include <optional>
#include <tuple>
#include <utility>

#include "maps.h"
#include "matrix.h"
#include "names.h"
#include "row.h"
#include "vertex_group.h"

namespace panda
{
   namespace input
   {
      /// Reads a conical/convex hull description with optional names and maps.
      template <typename Integer>
      std::tuple<Vertices<Integer>, Names, Maps, Inequalities<Integer>, std::optional<VertexGroup>> vertices(int, char**);
      /// Reads an inequality description with optional names and maps.
      template <typename Integer>
      std::tuple<Inequalities<Integer>, Names, Maps, Vertices<Integer>, std::optional<VertexGroup>> inequalities(int, char**);

      // explicit template instantiations
      template <>
      std::tuple<Vertices<int>, Names, Maps, Inequalities<int>, std::optional<VertexGroup>> vertices<int>(int, char**);
      template <>
      std::tuple<Inequalities<int>, Names, Maps, Vertices<int>, std::optional<VertexGroup>> inequalities<int>(int, char**);
   }
}

#include "input.tpp"


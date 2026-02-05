
//-------------------------------------------------------------------------------//
// Author: Christian Staufenbiel                                                 //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#pragma once

#include <set>
#include <vector>

#include "maps.h"
#include "matrix.h"
#include "row.h"
#include "vertex_group.h"

namespace panda
{
   namespace algorithm
   {
      /// Check if maps are pure vertex permutations (no scaling, only index swapping)
      bool arePurePermutations(const Maps& maps);

      /// Compute induced vertex permutations from coordinate maps.
      /// For each map in Maps, applies the coordinate transformation to each vertex
      /// and finds which vertex index the result maps to.
      /// Returns empty vector if:
      /// - maps are not pure permutations
      /// - some transformed vertex is not found in the vertex list
      /// - maps are empty
      template <typename Integer>
      std::vector<std::vector<std::size_t>> computeVertexPermutations(const Maps& maps, const Matrix<Integer>& vertices);

      /// Reduce a set of facets to equivalence class representatives
      /// using vertex-support-based canonical forms under a vertex group.
      template <typename Integer, typename TagType>
      Matrix<Integer> classesVertexSupport(std::set<Row<Integer>> rows,
                                           const Matrix<Integer>& vertices,
                                           const Maps& maps,
                                           const VertexGroup& group,
                                           TagType tag);
   }
}

#include "algorithm_classes_vertex_support.eti"

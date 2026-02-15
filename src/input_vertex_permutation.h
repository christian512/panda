
//-------------------------------------------------------------------------------//
// Author: Christian Staufenbiel                                                 //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#pragma once

#include <cstddef>
#include <istream>
#include <vector>

namespace panda
{
   namespace input
   {
      namespace implementation
      {
         /// Reads vertex permutations in disjoint cycle notation.
         /// Each line represents one permutation generator.
         /// Format: (a,b,c)(d,e) where a,b,c,d,e are 0-based vertex indices.
         /// Returns a list of permutation generators, where each generator is a
         /// permutation array: generator[i] = index of vertex that i maps to.
         std::vector<std::vector<std::size_t>> vertexPermutations(std::istream&, std::size_t n_vertices);
      }
   }
}

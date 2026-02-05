
//-------------------------------------------------------------------------------//
// Author: Christian Staufenbiel                                                 //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "maps.h"
#include "matrix.h"

namespace panda
{
   /// Wrapper around a permutalib group acting on vertex indices.
   /// All permutalib types are hidden behind the pimpl firewall.
   /// Vertex supports are represented as sorted vectors of vertex indices.
   class VertexGroup
   {
   public:
      /// Build a VertexGroup from original (pre-normalization) maps and vertices.
      /// Returns std::nullopt if maps are empty or not pure permutations.
      template <typename Integer>
      static std::optional<VertexGroup> create(const Maps& original_maps, const Matrix<Integer>& vertices);

      /// Construct from generator permutations on vertex indices.
      VertexGroup(const std::vector<std::vector<std::size_t>>& generators,
                  std::size_t n_vertices);

      // Allow copy via shared pimpl
      VertexGroup(const VertexGroup&) = default;
      VertexGroup& operator=(const VertexGroup&) = default;
      VertexGroup(VertexGroup&&) = default;
      VertexGroup& operator=(VertexGroup&&) = default;

      /// Compute the canonical form of a vertex support under the group.
      /// Input/output: sorted vector of vertex indices that lie on the face.
      /// Two supports are equivalent iff they have the same canonical form.
      std::vector<std::size_t> canonicalSupport(const std::vector<std::size_t>& support) const;

      /// Number of vertices the group acts on.
      std::size_t size() const;

   private:
      struct Impl;
      std::shared_ptr<const Impl> impl_;
   };
}

#include "vertex_group.eti"

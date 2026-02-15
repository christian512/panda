
//-------------------------------------------------------------------------------//
// Author: Christian Staufenbiel                                                 //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#define COMPILE_TEMPLATE_VERTEX_GROUP
#include "vertex_group.h"
#undef COMPILE_TEMPLATE_VERTEX_GROUP

#include <cassert>
#include <cstddef>
#include <optional>
#include <vector>

#ifdef DEBUG
#include <iostream>
#endif

#include "../permutalib/src/Group.h"
#include "../permutalib/src/Permutation.h"
#include "algorithm_classes_vertex_support.h"
#include "gmpxx.h"
#include "maps.h"
#include "matrix.h"

using namespace panda;

template <typename Integer>
std::optional<VertexGroup> VertexGroup::create(const Maps& original_maps, const Matrix<Integer>& vertices)
{
   if (original_maps.empty() || vertices.empty())
   {
      return std::nullopt;
   }

   auto vertex_perms = algorithm::computeVertexPermutations(original_maps, vertices);
   if (vertex_perms.empty())
   {
      return std::nullopt;
   }

   return VertexGroup(vertex_perms, vertices.size());
}

// VertexGroup implementation (must be in this translation unit because
// permutalib headers define non-inline functions that cause ODR violations
// if included in multiple translation units).

struct panda::VertexGroup::Impl
{
   permutalib::Group<permutalib::SingleSidedPerm<uint32_t>, mpz_class> group;
   std::size_t n_vertices;

   Impl(permutalib::Group<permutalib::SingleSidedPerm<uint32_t>, mpz_class>&& g, std::size_t n)
      : group(std::move(g)), n_vertices(n)
   {}
};

panda::VertexGroup::VertexGroup(const std::vector<std::vector<std::size_t>>& generators,
                                std::size_t n_vertices)
{
   assert(!generators.empty());
   assert(n_vertices > 0);

   #ifdef DEBUG
   std::cerr << "[DEBUG] VertexGroup: building group with " << generators.size()
             << " generators on " << n_vertices << " vertices" << std::endl;
   #endif

   using Tidx = uint32_t;
   using Telt = permutalib::SingleSidedPerm<Tidx>;
   using Tint = mpz_class;

   std::vector<Telt> gen_elts;
   gen_elts.reserve(generators.size());

   for (const auto& gen : generators)
   {
      assert(gen.size() == n_vertices);
      std::vector<Tidx> perm(n_vertices);
      for (std::size_t i = 0; i < n_vertices; ++i)
      {
         perm[i] = static_cast<Tidx>(gen[i]);
      }
      gen_elts.emplace_back(perm);
   }

   Telt id(static_cast<Tidx>(n_vertices));
   auto group = permutalib::Group<Telt, Tint>(gen_elts, id);

   #ifdef DEBUG
   std::cerr << "[DEBUG] VertexGroup: group order = " << group.size() << std::endl;
   #endif

   impl_ = std::make_shared<const Impl>(std::move(group), n_vertices);
}

std::vector<std::size_t> panda::VertexGroup::canonicalSupport(const std::vector<std::size_t>& support) const
{
   // Convert sorted vector of indices to Face (bitset)
   permutalib::Face face(impl_->n_vertices);
   for (const auto& idx : support)
   {
      assert(idx < impl_->n_vertices);
      face.set(idx);
   }

   // Compute canonical form
   auto canonical = impl_->group.CanonicalImage(face);

   // Convert Face back to sorted vector of indices
   std::vector<std::size_t> result;
   result.reserve(canonical.count());
   auto pos = canonical.find_first();
   while (pos != boost::dynamic_bitset<>::npos)
   {
      result.push_back(pos);
      pos = canonical.find_next(pos);
   }
   return result;
}

std::size_t panda::VertexGroup::size() const
{
   return impl_->n_vertices;
}

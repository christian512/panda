
//-------------------------------------------------------------------------------//
// Author: Stefan Lörwald, Universität Heidelberg                                //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#define COMPILE_TEMPLATE_ALGORITHM_ROTATION
#include "algorithm_rotation.h"
#undef COMPILE_TEMPLATE_ALGORITHM_ROTATION

#include <algorithm>
#include <cassert>
#include <list>
#include <set>

#ifdef DEBUG
#include <iostream>
#endif

#include "algorithm_classes.h"
#include "algorithm_classes_vertex_support.h"
#include "algorithm_fourier_motzkin_elimination.h"
#include "algorithm_inequality_operations.h"
#include "algorithm_integer_operations.h"
#include "algorithm_row_operations.h"

using namespace panda;

namespace
{
   /// Rotates a facet around a ridge. It's the exact same algorithm as for vertices.
   template <typename Integer>
   Facet<Integer> rotate(const Vertices<Integer>&, Vertex<Integer>, const Facet<Integer>&, Facet<Integer>);
   /// Returns all ridges on a facet (equivalent to all facets of the facet).
   template <typename Integer>
   Inequalities<Integer> getRidges(const Vertices<Integer>&, const Facet<Integer>&);
   /// Returns all vertices that lie on the facet (satisfy the inequality with equality).
   template <typename Integer>
   Vertices<Integer> verticesWithZeroDistance(const Vertices<Integer>&, const Facet<Integer>&);
   /// Returns ridges using single-threaded adjacency decomposition on the sub-polytope.
   template <typename Integer, typename TagType>
   Inequalities<Integer> getRidgesRecursive(const Vertices<Integer>&, const Facet<Integer>&, TagType, int, int, bool);
   /// Performs single-threaded adjacency decomposition, returning all facets found.
   template <typename Integer, typename TagType>
   Matrix<Integer> singleThreadedAD(const Matrix<Integer>&, TagType, int, int, bool);
}

template <typename Integer, typename TagType>
Matrix<Integer> panda::algorithm::rotation(const Matrix<Integer>& matrix,
                                    const Row<Integer>& input,
                                    const Maps& maps,
                                    const std::optional<VertexGroup>& vertex_group,
                                    TagType tag)
{
   // as the first step of the rotation, the furthest Vertex w.r.t. the input facet is calculated.
   // this will be the same vertex for all neighbouring ridges, hence, only needs to be computed once.
   const auto furthest_vertex = furthestVertex(matrix, input);
   const auto ridges = getRidges(matrix, input);
   std::set<Row<Integer>> output;
   for ( const auto& ridge : ridges )
   {
      const auto new_row = rotate(matrix, furthest_vertex, input, ridge);
      output.insert(new_row);
   }
   // When vertex group is available, skip equivalence reduction here;
   // canonical support dedup happens at put() time in the List.
   if ( vertex_group.has_value() )
   {
      return Matrix<Integer>(output.begin(), output.end());
   }
   return classes(output, maps, tag);
}

template <typename Integer, typename TagType>
Matrix<Integer> panda::algorithm::rotationRecursive(const Matrix<Integer>& matrix,
                                    const Row<Integer>& input,
                                    const Maps& maps,
                                    const std::optional<VertexGroup>& vertex_group,
                                    TagType tag,
                                    int recursion_depth,
                                    int min_vertices,
                                    bool sampling)
{
   const auto furthest_vertex = furthestVertex(matrix, input);
   const auto ridges = getRidgesRecursive(matrix, input, tag, recursion_depth, min_vertices, sampling);
   std::set<Row<Integer>> output;
   for ( const auto& ridge : ridges )
   {
      const auto new_row = rotate(matrix, furthest_vertex, input, ridge);
      output.insert(new_row);
   }
   #ifdef DEBUG
   std::cerr << "[DEBUG] Equivalence check: " << output.size() << " facets\n";
   #endif
   // When vertex group is available, skip equivalence reduction here;
   // canonical support dedup happens at put() time in the List.
   if ( vertex_group.has_value() )
   {
      return Matrix<Integer>(output.begin(), output.end());
   }
   return classes(output, maps, tag);
}

namespace
{
   template <typename Integer>
   Facet<Integer> rotate(const Vertices<Integer>& vertices, Vertex<Integer> vertex, const Facet<Integer>& facet, Facet<Integer> ridge)
   {
      // the calculation of the initial vertex, which has to be the furthest vertex w.r.t. "facet", is calculated outside of this function as it is the same for all rotations.
      auto d_f = algorithm::distance(facet, vertex);
      auto d_r = algorithm::distance(ridge, vertex);
      do
      {
         const auto gcd_ds = algorithm::gcd(d_f, d_r);
         if ( gcd_ds > 1 )
         {
            d_f /= gcd_ds;
            d_r /= gcd_ds;
         }
         ridge = d_f * ridge - d_r * facet;
         const auto gcd_value = algorithm::gcd(ridge);
         assert( gcd_value != 0 );
         if ( gcd_value > 1 )
         {
            ridge /= gcd_value;
         }
         vertex = algorithm::nearestVertex(vertices, ridge);
         d_f = algorithm::distance(facet, vertex);
         d_r = algorithm::distance(ridge, vertex);
      }
      while ( d_r != 0 );
      return ridge;
   }

   template <typename Integer>
   Inequalities<Integer> getRidges(const Vertices<Integer>& vertices, const Facet<Integer>& facet)
   {
      const auto vertices_on_facet = verticesWithZeroDistance(vertices, facet);
      assert( !vertices_on_facet.empty() );
      return algorithm::fourierMotzkinElimination(vertices_on_facet);
   }

   template <typename Integer>
   Vertices<Integer> verticesWithZeroDistance(const Vertices<Integer>& vertices, const Facet<Integer>& facet)
   {
      Vertices<Integer> selection;
      std::copy_if(vertices.cbegin(), vertices.cend(), std::back_inserter(selection), [&facet](const Vertex<Integer>& vertex)
      {
         return (algorithm::distance(facet, vertex) == 0);
      });
      return selection;
   }

   template <typename Integer, typename TagType>
   Inequalities<Integer> getRidgesRecursive(const Vertices<Integer>& vertices, const Facet<Integer>& facet, TagType tag, int recursion_depth, int min_vertices, bool sampling)
   {
      const auto vertices_on_facet = verticesWithZeroDistance(vertices, facet);
      assert( !vertices_on_facet.empty() );
      const auto num_vertices = static_cast<int>(vertices_on_facet.size());
      const auto effective_min = (min_vertices < 2) ? 2 : min_vertices;
      if ( recursion_depth > 0 && num_vertices >= effective_min )
      {
         #ifdef DEBUG
         std::cerr << "[DEBUG] Recursing down: depth=" << recursion_depth << " vertices=" << num_vertices << "\n";
         #endif
         auto result = singleThreadedAD(vertices_on_facet, tag, recursion_depth - 1, min_vertices, sampling);
         #ifdef DEBUG
         std::cerr << "[DEBUG] Recursing up: depth=" << recursion_depth << " ridges=" << result.size() << "\n";
         #endif
         return result;
      }
      #ifdef DEBUG
      std::cerr << "[DEBUG] FME: vertices=" << num_vertices << "\n";
      #endif
      return algorithm::fourierMotzkinElimination(vertices_on_facet);
   }

   template <typename Integer, typename TagType>
   Matrix<Integer> singleThreadedAD(const Matrix<Integer>& vertices, TagType tag, int recursion_depth, int min_vertices, bool sampling)
   {
      // Get initial facets via FME heuristic
      auto initial_facets = algorithm::fourierMotzkinEliminationHeuristic(vertices);
      if ( initial_facets.empty() )
      {
         return initial_facets;
      }
      // BFS rotation loop to find all facets
      std::set<Row<Integer>> all_facets(initial_facets.begin(), initial_facets.end());
      std::list<Row<Integer>> queue;
      if ( sampling )
      {
         queue.push_back(initial_facets.front());
      }
      else
      {
         queue.assign(initial_facets.begin(), initial_facets.end());
      }
      const Maps empty_maps;
      while ( !queue.empty() )
      {
         auto current = queue.front();
         queue.pop_front();
         const auto furthest = algorithm::furthestVertex(vertices, current);
         Inequalities<Integer> ridges;
         const auto effective_min = (min_vertices < 2) ? 2 : min_vertices;
         if ( recursion_depth > 0 && static_cast<int>(vertices.size()) >= effective_min )
         {
            ridges = getRidgesRecursive(vertices, current, tag, recursion_depth, min_vertices, sampling);
         }
         else
         {
            ridges = getRidges(vertices, current);
         }
         for ( const auto& ridge : ridges )
         {
            const auto adjacent = rotate(vertices, furthest, current, ridge);
            if ( all_facets.find(adjacent) == all_facets.end() )
            {
               all_facets.insert(adjacent);
               if ( !sampling )
               {
                  queue.push_back(adjacent);
               }
            }
         }
      }
      return Matrix<Integer>(all_facets.begin(), all_facets.end());
   }
}

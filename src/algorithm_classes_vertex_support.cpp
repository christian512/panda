
//-------------------------------------------------------------------------------//
// Author: Christian Staufenbiel                                                 //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#define COMPILE_TEMPLATE_ALGORITHM_CLASSES_VERTEX_SUPPORT
#include "algorithm_classes_vertex_support.h"
#undef COMPILE_TEMPLATE_ALGORITHM_CLASSES_VERTEX_SUPPORT

#include <algorithm>
#include <cassert>
#include <map>

#ifdef DEBUG
#include <iostream>
#endif

#include "algorithm_classes.h"
#include "algorithm_inequality_operations.h"
#include "algorithm_map_operations.h"
#include "tags.h"

using namespace panda;

bool panda::algorithm::arePurePermutations(const Maps& maps)
{
   #ifdef DEBUG
   std::cerr << "[DEBUG] Checking if maps are pure permutations..." << std::endl;
   std::cerr << "[DEBUG] Number of maps: " << maps.size() << std::endl;
   #endif

   for (std::size_t map_idx = 0; map_idx < maps.size(); ++map_idx)
   {
      const auto& map = maps[map_idx];
      #ifdef DEBUG
      std::cerr << "[DEBUG] Checking map " << map_idx << " (size: " << map.size() << ")" << std::endl;
      #endif

      // Check if map is a permutation matrix (with possible sign changes)
      // Each row (image) should have exactly one term with factor ±1
      for (std::size_t i = 0; i < map.size(); ++i)
      {
         const auto& image = map[i];
         if (image.size() != 1)
         {
            #ifdef DEBUG
            std::cerr << "[DEBUG] Map " << map_idx << ", position " << i
                      << ": image has " << image.size() << " terms (expected 1)" << std::endl;
            std::cerr << "[DEBUG] -> Maps are NOT pure permutations" << std::endl;
            #endif
            return false;
         }
         if (std::abs(image[0].second) != 1)
         {
            #ifdef DEBUG
            std::cerr << "[DEBUG] Map " << map_idx << ", position " << i
                      << ": factor is " << image[0].second << " (expected ±1)" << std::endl;
            std::cerr << "[DEBUG] -> Maps are NOT pure permutations" << std::endl;
            #endif
            return false;
         }
      }
   }

   #ifdef DEBUG
   std::cerr << "[DEBUG] -> Maps ARE pure permutations" << std::endl;
   #endif
   return true;
}

template <typename Integer>
std::vector<std::vector<std::size_t>> panda::algorithm::computeVertexPermutations(const Maps& maps, const Matrix<Integer>& vertices)
{
   #ifdef DEBUG
   std::cerr << "[DEBUG] computeVertexPermutations: maps.size()=" << maps.size()
             << ", vertices.size()=" << vertices.size() << std::endl;
   #endif

   if (maps.empty() || vertices.empty())
   {
      #ifdef DEBUG
      std::cerr << "[DEBUG] Empty maps or vertices, returning empty" << std::endl;
      #endif
      return {};
   }

   // Check if maps are pure permutations
   if (!arePurePermutations(maps))
   {
      #ifdef DEBUG
      std::cerr << "[DEBUG] Maps are not pure permutations, returning empty" << std::endl;
      #endif
      return {};
   }

   std::vector<std::vector<std::size_t>> vertex_permutations;
   vertex_permutations.reserve(maps.size());

   // For each coordinate map, compute the induced vertex permutation
   for (std::size_t map_idx = 0; map_idx < maps.size(); ++map_idx)
   {
      const auto& map = maps[map_idx];
      std::vector<std::size_t> vertex_perm(vertices.size());

      #ifdef DEBUG
      std::cerr << "[DEBUG] Processing map " << map_idx << std::endl;
      #endif

      // For each vertex, apply the map and find which vertex it maps to
      bool valid = true;
      for (std::size_t v_idx = 0; v_idx < vertices.size(); ++v_idx)
      {
         const auto& vertex = vertices[v_idx];

         // Apply coordinate map to vertex using facet tag
         // (We're transforming a point in coordinate space, which uses facet semantics)
         const auto transformed = apply(map, vertex, tag::facet{});

         // Find which vertex index this corresponds to
         auto it = std::find(vertices.cbegin(), vertices.cend(), transformed);
         if (it == vertices.cend())
         {
            #ifdef DEBUG
            std::cerr << "[DEBUG] Map " << map_idx << ": vertex " << v_idx
                      << " transforms to a point not in vertex list" << std::endl;
            #endif
            valid = false;
            break;
         }

         vertex_perm[v_idx] = static_cast<std::size_t>(it - vertices.cbegin());

         #ifdef DEBUG
         std::cerr << "[DEBUG]   vertex " << v_idx << " -> " << vertex_perm[v_idx] << std::endl;
         #endif
      }

      if (!valid)
      {
         #ifdef DEBUG
         std::cerr << "[DEBUG] Map " << map_idx << " does not induce a valid vertex permutation" << std::endl;
         #endif
         return {};
      }

      vertex_permutations.push_back(vertex_perm);
   }

   #ifdef DEBUG
   std::cerr << "[DEBUG] Successfully computed " << vertex_permutations.size()
             << " vertex permutations" << std::endl;
   #endif

   return vertex_permutations;
}

namespace
{
   /// Convert a facet to its vertex support: sorted vector of vertex indices
   /// where the inequality is satisfied with equality (distance == 0).
   template <typename Integer>
   std::vector<std::size_t> facetToVertexSupport(const Row<Integer>& facet,
                                                  const Matrix<Integer>& vertices)
   {
      std::vector<std::size_t> support;
      for (std::size_t i = 0; i < vertices.size(); ++i)
      {
         if (algorithm::distance(facet, vertices[i]) == 0)
         {
            support.push_back(i);
         }
      }
      return support;
   }
}

template <typename Integer, typename TagType>
Matrix<Integer> panda::algorithm::classesVertexSupport(std::set<Row<Integer>> rows,
                                                       const Matrix<Integer>& vertices,
                                                       const Maps& maps,
                                                       const VertexGroup& group,
                                                       TagType tag)
{
   if (rows.empty())
   {
      return Matrix<Integer>();
   }

   #ifdef DEBUG
   std::cerr << "[DEBUG] classesVertexSupport: " << rows.size() << " rows, "
             << vertices.size() << " vertices" << std::endl;
   #endif

   // Map from canonical support to representative facet
   std::map<std::vector<std::size_t>, Row<Integer>> canonical_to_representative;

   for (const auto& row : rows)
   {
      auto support = facetToVertexSupport(row, vertices);
      auto canonical = group.canonicalSupport(support);

      if (canonical_to_representative.find(canonical) == canonical_to_representative.end())
      {
         canonical_to_representative[canonical] = row;
      }
   }
   #ifdef DEBUG
   std::cerr << "[DEBUG] classesVertexSupport: calculating representative " << std::endl;
   #endif
   Matrix<Integer> result;
   result.reserve(canonical_to_representative.size());
   for (auto& pair : canonical_to_representative)
   {
      result.push_back(std::move(classRepresentative(pair.second, maps, tag)));
   }

   #ifdef DEBUG
   std::cerr << "[DEBUG] classesVertexSupport: reduced " << rows.size()
             << " rows to " << result.size() << " classes" << std::endl;
   #endif

   return result;
}

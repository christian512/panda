
//-------------------------------------------------------------------------------//
// Author: Christian Staufenbiel                                                 //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#include "input_vertex_permutation.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef DEBUG
#include <iostream>
#endif

#include "input_common.h"
#include "input_keywords.h"

using namespace panda;

namespace
{
   /// Parse a single permutation from a space-separated list of indices.
   /// Example: "1 0 3 2" for 4 vertices means 0->1, 1->0, 2->3, 3->2.
   std::vector<std::size_t> parsePermutation(const std::string& line, std::size_t n_vertices);
}

std::vector<std::vector<std::size_t>> panda::input::implementation::vertexPermutations(std::istream& stream, std::size_t n_vertices)
{
   std::vector<std::vector<std::size_t>> generators;

   // First, consume the keyword line (like maps() does)
   std::string token;
   if ( !std::getline(stream, token) || !isKeywordVertexPermutations(trimWhitespace(token)) )
   {
      throw std::invalid_argument("Cannot read vertex permutations: file is at an invalid position.");
   }

   // Now read the permutation lines
   for ( std::string line; std::getline(stream, line); )
   {
      line = trimWhitespace(line);

      // Stop at empty line or next keyword
      if ( line.empty() || isKeyword(line) )
      {
         break;
      }

      // Parse this permutation generator
      auto permutation = ::parsePermutation(line, n_vertices);
      generators.push_back(std::move(permutation));
   }

   #ifdef DEBUG
   std::cerr << "[DEBUG] Parsed " << generators.size() << " vertex permutation generators for " << n_vertices << " vertices\n";
   #endif

   return generators;
}

namespace
{
   std::vector<std::size_t> parsePermutation(const std::string& line, std::size_t n_vertices)
   {
      std::vector<std::size_t> permutation;
      std::istringstream ss(line);
      std::size_t value;

      while ( ss >> value )
      {
         if ( value >= n_vertices )
         {
            throw std::invalid_argument("Vertex index " + std::to_string(value) +
                                       " out of range [0, " + std::to_string(n_vertices - 1) +
                                       "] in permutation \"" + line + "\"");
         }
         permutation.push_back(value);
      }

      if ( permutation.size() != n_vertices )
      {
         throw std::invalid_argument("Permutation has " + std::to_string(permutation.size()) +
                                    " entries but expected " + std::to_string(n_vertices) +
                                    " (one per vertex).");
      }

      return permutation;
   }
}

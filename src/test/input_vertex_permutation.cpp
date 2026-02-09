
//-------------------------------------------------------------------------------//
// Author: Christian Staufenbiel                                                 //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#include "testing_gear.h"

#include "input_vertex_permutation.h"

#include <sstream>
#include <stdexcept>
#include <vector>

using namespace panda;

namespace
{
   void single_generator();
   void multiple_generators();
   void identity_permutation();
   void wrong_count();
   void out_of_range();
   void invalid_position();
   void empty_section();
}

int main()
try
{
   single_generator();
   multiple_generators();
   identity_permutation();
   wrong_count();
   out_of_range();
   invalid_position();
   empty_section();
}
catch ( const TestingGearException& e )
{
   std::cerr << e.what() << "\n";
   return 1;
}

namespace
{
   void single_generator()
   {
      std::istringstream stream("VERTEX_PERMUTATIONS:\n1 0 3 2\n");
      const auto generators = input::implementation::vertexPermutations(stream, 4);
      ASSERT(generators.size() == 1, "Expected exactly one generator.");
      ASSERT((generators[0] == std::vector<std::size_t>{1, 0, 3, 2}), "Generator mismatch.");
   }

   void multiple_generators()
   {
      std::istringstream stream("VERTEX_PERMUTATIONS:\n1 0 2 3\n0 1 3 2\n");
      const auto generators = input::implementation::vertexPermutations(stream, 4);
      ASSERT(generators.size() == 2, "Expected two generators.");
      ASSERT((generators[0] == std::vector<std::size_t>{1, 0, 2, 3}), "First generator mismatch.");
      ASSERT((generators[1] == std::vector<std::size_t>{0, 1, 3, 2}), "Second generator mismatch.");
   }

   void identity_permutation()
   {
      std::istringstream stream("VERTEX_PERMUTATIONS:\n0 1 2\n");
      const auto generators = input::implementation::vertexPermutations(stream, 3);
      ASSERT(generators.size() == 1, "Expected exactly one generator.");
      ASSERT((generators[0] == std::vector<std::size_t>{0, 1, 2}), "Identity permutation mismatch.");
   }

   void wrong_count()
   {
      std::istringstream stream("VERTEX_PERMUTATIONS:\n1 0 3\n");
      ASSERT_EXCEPTION(input::implementation::vertexPermutations(stream, 4),
                       std::invalid_argument,
                       "Permutation with wrong number of entries must throw.");
   }

   void out_of_range()
   {
      std::istringstream stream("VERTEX_PERMUTATIONS:\n1 0 10 3\n");
      ASSERT_EXCEPTION(input::implementation::vertexPermutations(stream, 4),
                       std::invalid_argument,
                       "Permutation with out-of-range index must throw.");
   }

   void invalid_position()
   {
      std::istringstream stream("NOT_A_KEYWORD\n1 0 3 2\n");
      ASSERT_EXCEPTION(input::implementation::vertexPermutations(stream, 4),
                       std::invalid_argument,
                       "Reading from wrong file position must throw.");
   }

   void empty_section()
   {
      std::istringstream stream("VERTEX_PERMUTATIONS:\n\n");
      const auto generators = input::implementation::vertexPermutations(stream, 4);
      ASSERT(generators.size() == 0, "Empty section should yield zero generators.");
   }
}

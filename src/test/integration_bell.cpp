
//-------------------------------------------------------------------------------//
// Author: Christian Staufenbiel                                                 //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#include "testing_gear.h"

#include "method_facet_enumeration.h"

#include <sstream>
#include <string>

using namespace panda;

namespace
{
   void bell_2222();
   void bell_2233();
   void bell_3322();
   void bell_4322();

   int countLines(const std::string& output, const std::string& after_header)
   {
      int count = 0;
      std::istringstream iss(output);
      std::string line;
      bool counting = false;
      while ( std::getline(iss, line) )
      {
         if ( line.find(after_header) != std::string::npos )
         {
            counting = true;
            continue;
         }
         if ( counting && !line.empty() && line.find_first_not_of(" \t") != std::string::npos )
         {
            count++;
         }
      }
      return count;
   }

   std::string runFacetEnumeration(const char* file)
   {
      char* argv[] = {
         (char*)"panda",
         (char*)file,
         (char*)"-m", (char*)"ad",
         (char*)"-t", (char*)"1"
      };
      int argc = 6;

      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());
      int result = panda::method::facetEnumeration(argc, argv);
      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, std::string("Facet enumeration failed for ") + file);
      return output.str();
   }
}

int main()
try
{
   bell_2222();
   bell_2233();
   bell_3322();
   bell_4322();
}
catch ( const TestingGearException& e )
{
   std::cerr << e.what() << "\n";
   return 1;
}

namespace
{
   /// CHSH scenario (2 inputs, 2 outputs per party)
   /// Known result: 2 classes of facet inequalities
   ///   - non-negativity: -p(0,0|x,y) <= 0
   ///   - CHSH inequality
   void bell_2222()
   {
      SILENCE_CERR();
      const auto output = runFacetEnumeration("../samples/panda_format/bell/2222");
      ASSERT(output.find("Inequalities:") != std::string::npos, "bell_2222: missing 'Inequalities:' header");
      ASSERT(countLines(output, "Inequalities:") == 2, "bell_2222: expected 2 facet classes");
   }

   /// 2 inputs for A, 2 inputs for B, 2 outputs for A, 3 outputs for B
   /// Known result: 4 classes of facet inequalities
   void bell_2233()
   {
      SILENCE_CERR();
      const auto output = runFacetEnumeration("../samples/panda_format/bell/2233");
      ASSERT(output.find("Inequalities:") != std::string::npos, "bell_2233: missing 'Inequalities:' header");
      ASSERT(countLines(output, "Inequalities:") == 4, "bell_2233: expected 4 facet classes");
   }

   /// 3 inputs for A, 3 inputs for B, 2 outputs per party
   /// Known result: 3 classes of facet inequalities (includes I3322)
   void bell_3322()
   {
      SILENCE_CERR();
      const auto output = runFacetEnumeration("../samples/panda_format/bell/3322");
      ASSERT(output.find("Inequalities:") != std::string::npos, "bell_3322: missing 'Inequalities:' header");
      ASSERT(countLines(output, "Inequalities:") == 3, "bell_3322: expected 3 facet classes");
   }

   /// 4 inputs for A, 3 inputs for B, 2 outputs per party
   /// Known result: 6 classes of facet inequalities
   void bell_4322()
   {
      SILENCE_CERR();
      const auto output = runFacetEnumeration("../samples/panda_format/bell/4322");
      ASSERT(output.find("Inequalities:") != std::string::npos, "bell_4322: missing 'Inequalities:' header");
      ASSERT(countLines(output, "Inequalities:") == 6, "bell_4322: expected 6 facet classes");
   }
}

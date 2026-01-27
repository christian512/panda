
//-------------------------------------------------------------------------------//
// Author: Christian Staufenbiel                                                 //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#include "testing_gear.h"

#include "method_facet_enumeration.h"
#include "method_vertex_enumeration.h"

#include <sstream>

using namespace panda;

namespace
{
   void sample1_facetEnumeration_AD();
   void sample1_facetEnumeration_DD();
   void sample3_vertexEnumeration_AD();
   void sample3_vertexEnumeration_DD();
   void sample4_vertexEnumeration_AD();
   void sample5_facetEnumeration_AD();
   void porta_sample1_facetEnumeration_AD();
   void sample1_facetEnumeration_AD_r1();
   void sample1_facetEnumeration_AD_r2();
   void sample1_facetEnumeration_AD_r1_minv5();
   void sample3_vertexEnumeration_AD_r1();
   void sample5_facetEnumeration_AD_r1();
   void sample5_facetEnumeration_AD_r1_minv3();
}

int main()
try
{
   sample1_facetEnumeration_AD();
   sample1_facetEnumeration_DD();
   sample3_vertexEnumeration_AD();
   sample3_vertexEnumeration_DD();
   sample4_vertexEnumeration_AD();
   sample5_facetEnumeration_AD();
   porta_sample1_facetEnumeration_AD();
   sample1_facetEnumeration_AD_r1();
   sample1_facetEnumeration_AD_r2();
   sample1_facetEnumeration_AD_r1_minv5();
   sample3_vertexEnumeration_AD_r1();
   sample5_facetEnumeration_AD_r1();
   sample5_facetEnumeration_AD_r1_minv3();
}
catch ( const TestingGearException& e )
{
   std::cerr << e.what() << "\n";
   return 1;
}

namespace
{
   /// Sample 1: Facet enumeration with Adjacency Decomposition
   /// Input: samples/panda_format/sample_1 (unit cube vertices)
   /// Expected: 6 facet inequalities
   void sample1_facetEnumeration_AD()
   {
      SILENCE_CERR();

      char* argv[] = {
         (char*)"panda",
         (char*)"../samples/panda_format/sample_1",
         (char*)"-m", (char*)"ad",  // adjacency decomposition
         (char*)"-t", (char*)"1"     // single thread for reproducibility
      };
      int argc = 6;

      // Redirect stdout to capture output
      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

      int result = panda::method::facetEnumeration(argc, argv);

      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, "Sample 1 (AD): Facet enumeration failed");

      // Verify output contains expected facets
      std::string output_str = output.str();
      ASSERT(output_str.find("Inequalities:") != std::string::npos,
             "Sample 1 (AD): Output missing 'Inequalities:' header");
      ASSERT(output_str.find("-x <= 0") != std::string::npos,
             "Sample 1 (AD): Missing facet -x <= 0");
      ASSERT(output_str.find("-y <= 0") != std::string::npos,
             "Sample 1 (AD): Missing facet -y <= 0");
      ASSERT(output_str.find("-z <= 0") != std::string::npos,
             "Sample 1 (AD): Missing facet -z <= 0");
      ASSERT(output_str.find("x <= 1") != std::string::npos,
             "Sample 1 (AD): Missing facet x <= 1");
      ASSERT(output_str.find("y <= 1") != std::string::npos,
             "Sample 1 (AD): Missing facet y <= 1");
      ASSERT(output_str.find("z <= 1") != std::string::npos,
             "Sample 1 (AD): Missing facet z <= 1");
   }

   /// Sample 1: Facet enumeration with Double Description
   /// Input: samples/panda_format/sample_1 (unit cube vertices)
   /// Expected: 6 facet inequalities
   void sample1_facetEnumeration_DD()
   {
      SILENCE_CERR();

      char* argv[] = {
         (char*)"panda",
         (char*)"../samples/panda_format/sample_1",
         (char*)"-m", (char*)"dd",  // double description
         (char*)"-t", (char*)"1"
      };
      int argc = 6;

      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

      int result = panda::method::facetEnumeration(argc, argv);

      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, "Sample 1 (DD): Facet enumeration failed");

      std::string output_str = output.str();
      ASSERT(output_str.find("Inequalities:") != std::string::npos,
             "Sample 1 (DD): Output missing 'Inequalities:' header");
      ASSERT(output_str.find("-x <= 0") != std::string::npos,
             "Sample 1 (DD): Missing facet -x <= 0");
      ASSERT(output_str.find("-y <= 0") != std::string::npos,
             "Sample 1 (DD): Missing facet -y <= 0");
      ASSERT(output_str.find("-z <= 0") != std::string::npos,
             "Sample 1 (DD): Missing facet -z <= 0");
      ASSERT(output_str.find("x <= 1") != std::string::npos,
             "Sample 1 (DD): Missing facet x <= 1");
      ASSERT(output_str.find("y <= 1") != std::string::npos,
             "Sample 1 (DD): Missing facet y <= 1");
      ASSERT(output_str.find("z <= 1") != std::string::npos,
             "Sample 1 (DD): Missing facet z <= 1");
   }

   /// Sample 3: Vertex enumeration with Adjacency Decomposition
   /// Input: samples/panda_format/sample_3 (unit cube inequalities)
   /// Expected: 8 vertices
   void sample3_vertexEnumeration_AD()
   {
      SILENCE_CERR();

      char* argv[] = {
         (char*)"panda",
         (char*)"../samples/panda_format/sample_3",
         (char*)"-m", (char*)"ad",
         (char*)"-t", (char*)"1"
      };
      int argc = 6;

      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

      int result = panda::method::vertexEnumeration(argc, argv);

      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, "Sample 3 (AD): Vertex enumeration failed");

      std::string output_str = output.str();
      ASSERT(output_str.find("Vertices / Rays:") != std::string::npos,
             "Sample 3 (AD): Output missing 'Vertices / Rays:' header");

      // Count vertices - should have 8 lines with vertices
      int vertex_count = 0;
      std::istringstream iss(output_str);
      std::string line;
      bool in_vertices = false;
      while (std::getline(iss, line))
      {
         if (line.find("Vertices / Rays:") != std::string::npos)
         {
            in_vertices = true;
            continue;
         }
         if (in_vertices && !line.empty() && line.find_first_not_of(" \t") != std::string::npos)
         {
            vertex_count++;
         }
      }
      ASSERT(vertex_count == 8, "Sample 3 (AD): Expected 8 vertices");
   }

   /// Sample 3: Vertex enumeration with Double Description
   /// Input: samples/panda_format/sample_3 (unit cube inequalities)
   /// Expected: 8 vertices
   void sample3_vertexEnumeration_DD()
   {
      SILENCE_CERR();

      char* argv[] = {
         (char*)"panda",
         (char*)"../samples/panda_format/sample_3",
         (char*)"-m", (char*)"dd",
         (char*)"-t", (char*)"1"
      };
      int argc = 6;

      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

      int result = panda::method::vertexEnumeration(argc, argv);

      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, "Sample 3 (DD): Vertex enumeration failed");

      std::string output_str = output.str();
      // DD method outputs "Vertices:" instead of "Vertices / Rays:"
      ASSERT(output_str.find("Vertices:") != std::string::npos,
             "Sample 3 (DD): Output missing 'Vertices:' header");

      // Count vertices - should have 8 lines with vertices
      int vertex_count = 0;
      std::istringstream iss(output_str);
      std::string line;
      bool in_vertices = false;
      while (std::getline(iss, line))
      {
         if (line.find("Vertices:") != std::string::npos)
         {
            in_vertices = true;
            continue;
         }
         if (in_vertices && !line.empty() && line.find_first_not_of(" \t") != std::string::npos)
         {
            vertex_count++;
         }
      }
      ASSERT(vertex_count == 8, "Sample 3 (DD): Expected 8 vertices");
   }

   /// Sample 4: Vertex enumeration with Adjacency Decomposition
   /// Input: samples/panda_format/sample_4 (same as sample 3)
   /// Expected: 8 vertices
   void sample4_vertexEnumeration_AD()
   {
      SILENCE_CERR();

      char* argv[] = {
         (char*)"panda",
         (char*)"../samples/panda_format/sample_4",
         (char*)"-m", (char*)"ad",
         (char*)"-t", (char*)"1"
      };
      int argc = 6;

      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

      int result = panda::method::vertexEnumeration(argc, argv);

      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, "Sample 4 (AD): Vertex enumeration failed");

      std::string output_str = output.str();
      ASSERT(output_str.find("Vertices / Rays:") != std::string::npos,
             "Sample 4 (AD): Output missing 'Vertices / Rays:' header");

      // Count vertices - should have 8 lines with vertices
      int vertex_count = 0;
      std::istringstream iss(output_str);
      std::string line;
      bool in_vertices = false;
      while (std::getline(iss, line))
      {
         if (line.find("Vertices / Rays:") != std::string::npos)
         {
            in_vertices = true;
            continue;
         }
         if (in_vertices && !line.empty() && line.find_first_not_of(" \t") != std::string::npos)
         {
            vertex_count++;
         }
      }
      ASSERT(vertex_count == 8, "Sample 4 (AD): Expected 8 vertices");
   }

   /// Sample 5: Facet enumeration with Adjacency Decomposition
   /// Input: samples/panda_format/sample_5 (polyhedron with vertices and rays)
   /// Expected: 4 facet inequalities
   void sample5_facetEnumeration_AD()
   {
      SILENCE_CERR();

      char* argv[] = {
         (char*)"panda",
         (char*)"../samples/panda_format/sample_5",
         (char*)"-m", (char*)"ad",
         (char*)"-t", (char*)"1"
      };
      int argc = 6;

      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

      int result = panda::method::facetEnumeration(argc, argv);

      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, "Sample 5 (AD): Facet enumeration failed");

      std::string output_str = output.str();
      ASSERT(output_str.find("Inequalities:") != std::string::npos,
             "Sample 5 (AD): Output missing 'Inequalities:' header");

      // Verify we have the expected facets
      int facet_count = 0;
      std::istringstream iss(output_str);
      std::string line;
      bool in_inequalities = false;
      while (std::getline(iss, line))
      {
         if (line.find("Inequalities:") != std::string::npos)
         {
            in_inequalities = true;
            continue;
         }
         if (in_inequalities && !line.empty() && line.find_first_not_of(" \t") != std::string::npos)
         {
            facet_count++;
         }
      }
      ASSERT(facet_count == 4, "Sample 5 (AD): Expected 4 facets");
   }

   /// PORTA format sample 1: Facet enumeration with Adjacency Decomposition
   /// Input: samples/porta_format/sample_1 (unit cube vertices in PORTA format)
   /// Expected: 6 facet inequalities
   void porta_sample1_facetEnumeration_AD()
   {
      SILENCE_CERR();

      char* argv[] = {
         (char*)"panda",
         (char*)"../samples/porta_format/sample_1",
         (char*)"-m", (char*)"ad",
         (char*)"-t", (char*)"1"
      };
      int argc = 6;

      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

      int result = panda::method::facetEnumeration(argc, argv);

      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, "PORTA Sample 1 (AD): Facet enumeration failed");

      std::string output_str = output.str();
      ASSERT(output_str.find("Inequalities:") != std::string::npos,
             "PORTA Sample 1 (AD): Output missing 'Inequalities:' header");

      // Verify we have the expected 6 facets (same as PANDA format sample 1)
      int facet_count = 0;
      std::istringstream iss(output_str);
      std::string line;
      bool in_inequalities = false;
      while (std::getline(iss, line))
      {
         if (line.find("Inequalities:") != std::string::npos)
         {
            in_inequalities = true;
            continue;
         }
         if (in_inequalities && !line.empty() && line.find_first_not_of(" \t") != std::string::npos)
         {
            facet_count++;
         }
      }
      ASSERT(facet_count == 6, "PORTA Sample 1 (AD): Expected 6 facets");
   }

   /// Sample 1: Facet enumeration with AD and recursion depth 1
   void sample1_facetEnumeration_AD_r1()
   {
      SILENCE_CERR();

      char* argv[] = {
         (char*)"panda",
         (char*)"../samples/panda_format/sample_1",
         (char*)"-m", (char*)"ad",
         (char*)"-t", (char*)"1",
         (char*)"-r", (char*)"1"
      };
      int argc = 8;

      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

      int result = panda::method::facetEnumeration(argc, argv);

      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, "Sample 1 (AD, r=1): Facet enumeration failed");

      std::string output_str = output.str();
      ASSERT(output_str.find("-x <= 0") != std::string::npos,
             "Sample 1 (AD, r=1): Missing facet -x <= 0");
      ASSERT(output_str.find("-y <= 0") != std::string::npos,
             "Sample 1 (AD, r=1): Missing facet -y <= 0");
      ASSERT(output_str.find("-z <= 0") != std::string::npos,
             "Sample 1 (AD, r=1): Missing facet -z <= 0");
      ASSERT(output_str.find("x <= 1") != std::string::npos,
             "Sample 1 (AD, r=1): Missing facet x <= 1");
      ASSERT(output_str.find("y <= 1") != std::string::npos,
             "Sample 1 (AD, r=1): Missing facet y <= 1");
      ASSERT(output_str.find("z <= 1") != std::string::npos,
             "Sample 1 (AD, r=1): Missing facet z <= 1");
   }

   /// Sample 1: Facet enumeration with AD and recursion depth 2
   void sample1_facetEnumeration_AD_r2()
   {
      SILENCE_CERR();

      char* argv[] = {
         (char*)"panda",
         (char*)"../samples/panda_format/sample_1",
         (char*)"-m", (char*)"ad",
         (char*)"-t", (char*)"1",
         (char*)"-r", (char*)"2"
      };
      int argc = 8;

      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

      int result = panda::method::facetEnumeration(argc, argv);

      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, "Sample 1 (AD, r=2): Facet enumeration failed");

      std::string output_str = output.str();
      ASSERT(output_str.find("-x <= 0") != std::string::npos,
             "Sample 1 (AD, r=2): Missing facet -x <= 0");
      ASSERT(output_str.find("-y <= 0") != std::string::npos,
             "Sample 1 (AD, r=2): Missing facet -y <= 0");
      ASSERT(output_str.find("-z <= 0") != std::string::npos,
             "Sample 1 (AD, r=2): Missing facet -z <= 0");
      ASSERT(output_str.find("x <= 1") != std::string::npos,
             "Sample 1 (AD, r=2): Missing facet x <= 1");
      ASSERT(output_str.find("y <= 1") != std::string::npos,
             "Sample 1 (AD, r=2): Missing facet y <= 1");
      ASSERT(output_str.find("z <= 1") != std::string::npos,
             "Sample 1 (AD, r=2): Missing facet z <= 1");
   }

   /// Sample 1: Facet enumeration with AD, recursion depth 1, min-vertices=5
   /// The cube has 4 vertices per face, so min-vertices=5 should prevent recursion
   /// and produce the same result as without recursion.
   void sample1_facetEnumeration_AD_r1_minv5()
   {
      SILENCE_CERR();

      char* argv[] = {
         (char*)"panda",
         (char*)"../samples/panda_format/sample_1",
         (char*)"-m", (char*)"ad",
         (char*)"-t", (char*)"1",
         (char*)"-r", (char*)"1",
         (char*)"--recursion-min-vertices=5"
      };
      int argc = 9;

      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

      int result = panda::method::facetEnumeration(argc, argv);

      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, "Sample 1 (AD, r=1, minv=5): Facet enumeration failed");

      std::string output_str = output.str();
      ASSERT(output_str.find("-x <= 0") != std::string::npos,
             "Sample 1 (AD, r=1, minv=5): Missing facet -x <= 0");
      ASSERT(output_str.find("-y <= 0") != std::string::npos,
             "Sample 1 (AD, r=1, minv=5): Missing facet -y <= 0");
      ASSERT(output_str.find("-z <= 0") != std::string::npos,
             "Sample 1 (AD, r=1, minv=5): Missing facet -z <= 0");
      ASSERT(output_str.find("x <= 1") != std::string::npos,
             "Sample 1 (AD, r=1, minv=5): Missing facet x <= 1");
      ASSERT(output_str.find("y <= 1") != std::string::npos,
             "Sample 1 (AD, r=1, minv=5): Missing facet y <= 1");
      ASSERT(output_str.find("z <= 1") != std::string::npos,
             "Sample 1 (AD, r=1, minv=5): Missing facet z <= 1");
   }

   /// Sample 3: Vertex enumeration with AD and recursion depth 1
   void sample3_vertexEnumeration_AD_r1()
   {
      SILENCE_CERR();

      char* argv[] = {
         (char*)"panda",
         (char*)"../samples/panda_format/sample_3",
         (char*)"-m", (char*)"ad",
         (char*)"-t", (char*)"1",
         (char*)"-r", (char*)"1"
      };
      int argc = 8;

      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

      int result = panda::method::vertexEnumeration(argc, argv);

      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, "Sample 3 (AD, r=1): Vertex enumeration failed");

      std::string output_str = output.str();
      ASSERT(output_str.find("Vertices / Rays:") != std::string::npos,
             "Sample 3 (AD, r=1): Output missing 'Vertices / Rays:' header");

      int vertex_count = 0;
      std::istringstream iss(output_str);
      std::string line;
      bool in_vertices = false;
      while (std::getline(iss, line))
      {
         if (line.find("Vertices / Rays:") != std::string::npos)
         {
            in_vertices = true;
            continue;
         }
         if (in_vertices && !line.empty() && line.find_first_not_of(" \t") != std::string::npos)
         {
            vertex_count++;
         }
      }
      ASSERT(vertex_count == 8, "Sample 3 (AD, r=1): Expected 8 vertices");
   }

   /// Sample 5: Facet enumeration with AD and recursion depth 1
   void sample5_facetEnumeration_AD_r1()
   {
      SILENCE_CERR();

      char* argv[] = {
         (char*)"panda",
         (char*)"../samples/panda_format/sample_5",
         (char*)"-m", (char*)"ad",
         (char*)"-t", (char*)"1",
         (char*)"-r", (char*)"1"
      };
      int argc = 8;

      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

      int result = panda::method::facetEnumeration(argc, argv);

      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, "Sample 5 (AD, r=1): Facet enumeration failed");

      std::string output_str = output.str();
      ASSERT(output_str.find("Inequalities:") != std::string::npos,
             "Sample 5 (AD, r=1): Output missing 'Inequalities:' header");

      int facet_count = 0;
      std::istringstream iss(output_str);
      std::string line;
      bool in_inequalities = false;
      while (std::getline(iss, line))
      {
         if (line.find("Inequalities:") != std::string::npos)
         {
            in_inequalities = true;
            continue;
         }
         if (in_inequalities && !line.empty() && line.find_first_not_of(" \t") != std::string::npos)
         {
            facet_count++;
         }
      }
      ASSERT(facet_count == 4, "Sample 5 (AD, r=1): Expected 4 facets");
   }

   /// Sample 5: Facet enumeration with AD, recursion depth 1, min-vertices=3
   void sample5_facetEnumeration_AD_r1_minv3()
   {
      SILENCE_CERR();

      char* argv[] = {
         (char*)"panda",
         (char*)"../samples/panda_format/sample_5",
         (char*)"-m", (char*)"ad",
         (char*)"-t", (char*)"1",
         (char*)"-r", (char*)"1",
         (char*)"--recursion-min-vertices=3"
      };
      int argc = 9;

      std::ostringstream output;
      std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

      int result = panda::method::facetEnumeration(argc, argv);

      std::cout.rdbuf(old_cout);

      ASSERT(result == 0, "Sample 5 (AD, r=1, minv=3): Facet enumeration failed");

      std::string output_str = output.str();
      ASSERT(output_str.find("Inequalities:") != std::string::npos,
             "Sample 5 (AD, r=1, minv=3): Output missing 'Inequalities:' header");

      int facet_count = 0;
      std::istringstream iss(output_str);
      std::string line;
      bool in_inequalities = false;
      while (std::getline(iss, line))
      {
         if (line.find("Inequalities:") != std::string::npos)
         {
            in_inequalities = true;
            continue;
         }
         if (in_inequalities && !line.empty() && line.find_first_not_of(" \t") != std::string::npos)
         {
            facet_count++;
         }
      }
      ASSERT(facet_count == 4, "Sample 5 (AD, r=1, minv=3): Expected 4 facets");
   }
}

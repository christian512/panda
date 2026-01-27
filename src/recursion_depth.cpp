
//-------------------------------------------------------------------------------//
// Author: Christian Staufenbiel                                                 //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#include "recursion_depth.h"

#include <cassert>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace panda;

namespace
{
   /// Tries to read a number from char*.
   int interpretParameter(char*, const std::string&);
}

int panda::recursion::depth(int argc, char** argv)
{
   assert( argc > 0 && argv != nullptr );
   for ( int i = 1; i < argc; ++i )
   {
      if ( std::strcmp(argv[i], "-r") == 0 )
      {
         if ( i + 1 == argc )
         {
            throw std::invalid_argument("Command line option \"-r <n>\" needs an integral parameter.");
         }
         return interpretParameter(argv[i + 1], "recursion-depth");
      }
      else if ( std::strncmp(argv[i], "-r=", 3) == 0 )
      {
         return interpretParameter(argv[i] + 3, "recursion-depth");
      }
      else if ( std::strncmp(argv[i], "--recursion-depth=", 18) == 0 )
      {
         return interpretParameter(argv[i] + 18, "recursion-depth");
      }
      else if ( std::strncmp(argv[i], "-r", 2) == 0 || std::strncmp(argv[i], "--r", 3) == 0 )
      {
         throw std::invalid_argument("Illegal parameter. Did you mean \"-r <n>\" or \"--recursion-depth=<n>\"?");
      }
   }
   return 0; // Default: no recursion
}

int panda::recursion::minimumVertices(int argc, char** argv)
{
   assert( argc > 0 && argv != nullptr );
   for ( int i = 1; i < argc; ++i )
   {
      if ( std::strcmp(argv[i], "--recursion-min-vertices") == 0 )
      {
         if ( i + 1 == argc )
         {
            throw std::invalid_argument("Command line option \"--recursion-min-vertices <n>\" needs an integral parameter.");
         }
         return interpretParameter(argv[i + 1], "recursion-min-vertices");
      }
      else if ( std::strncmp(argv[i], "--recursion-min-vertices=", 25) == 0 )
      {
         return interpretParameter(argv[i] + 25, "recursion-min-vertices");
      }
   }
   return 0; // Default: no minimum
}

namespace
{
   int interpretParameter(char* string, const std::string& option_name)
   {
      assert( string != nullptr );
      std::istringstream stream(string);
      int n;
      if ( !(stream >> n) )
      {
         std::string message = "Command line option for " + option_name;
         message += " needs an integral parameter.";
         throw std::invalid_argument(message);
      }
      std::string rest;
      stream >> rest;
      if ( !rest.empty() )
      {
         std::string message = "Command line option for " + option_name;
         message += " needs an integral parameter.";
         throw std::invalid_argument(message);
      }
      if ( n < 0 )
      {
         std::string message = "Command line option for " + option_name;
         message += " needs a non-negative integral parameter.";
         throw std::invalid_argument(message);
      }
      return n;
   }
}

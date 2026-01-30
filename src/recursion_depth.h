
//-------------------------------------------------------------------------------//
// Author: Christian Staufenbiel                                                 //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#pragma once

namespace panda
{
   namespace recursion
   {
      /// Returns the recursion depth from command line arguments.
      /// If not specified, returns 0 (no recursion, use FME directly).
      int depth(int, char**);

      /// Returns the minimum number of vertices required for recursion.
      /// If not specified, returns 0 (no minimum).
      int minimumVertices(int, char**);

      /// Returns whether sampling mode is enabled.
      /// In sampling mode, the inner AD does not enqueue newly found facets.
      bool sampling(int, char**);
   }
}

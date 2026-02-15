
//-------------------------------------------------------------------------------//
// Author: Stefan Lörwald, Universität Heidelberg                                //
// License: CC BY-NC 4.0 http://creativecommons.org/licenses/by-nc/4.0/legalcode //
//-------------------------------------------------------------------------------//

#define COMPILE_TEMPLATE_LIST
#include "list.h"
#undef COMPILE_TEMPLATE_LIST

#include <cstddef>
#include <iostream>
#include <sstream>

#include "algorithm_inequality_operations.h"
#include "algorithm_row_operations.h"

using namespace panda;

#define PRINT_DONE_COUNTER /// if enabled, the beginning of processing a row will be announced.

#ifdef PRINT_DONE_COUNTER
   #include "cpp_feature_check_thread_local.h"
   #if HAS_FEATURE_THREAD_LOCAL != 0
      namespace
      {
         thread_local std::size_t index;
      }
   #else
      #include <map>
      #include <thread>
      namespace
      {
         std::map<std::thread::id, std::size_t> indices;
      }
   #endif
#endif

template <typename Integer, typename TagType>
void panda::List<Integer, TagType>::put(const Matrix<Integer>& matrix) const
{
   for ( const auto& row : matrix )
   {
      put(row);
   }
   std::lock_guard<std::mutex> lock(mutex);
   --workers;
   #ifdef PRINT_DONE_COUNTER
   #if HAS_FEATURE_THREAD_LOCAL == 0
   auto index = indices[std::this_thread::get_id()];
   #endif
   if ( index > 0 )
   {
      std::stringstream stream;
      stream << "Done processing #" << index << '\n';
      std::cerr << stream.str();
   }
   #endif
}

template <typename Integer, typename TagType>
void panda::List<Integer, TagType>::put(const Row<Integer>& row) const
{
   // If vertex_group is available, compute canonical support outside the lock
   std::vector<std::size_t> canonical;
   if ( vertex_group )
   {
      std::vector<std::size_t> support;
      for ( std::size_t i = 0; i < vertices.size(); ++i )
      {
         if ( algorithm::distance(row, vertices[i]) == 0 )
         {
            support.push_back(i);
         }
      }
      canonical = vertex_group->canonicalSupport(support);
   }

   std::lock_guard<std::mutex> lock(mutex);

   // Canonical support dedup: skip if this canonical form was already seen
   if ( vertex_group )
   {
      auto result = seen_supports.insert(std::move(canonical));
      if ( !result.second )
      {
         return;
      }
   }

   Iterator it;
   bool added;
   std::tie(it, added) = rows.insert(row);
   if ( added )
   {
      if ( std::is_same<TagType, tag::facet>::value )
      {
         algorithm::prettyPrintln(std::cout, row, names, "<=");
      }
      else
      {
         std::cout << row << '\n';
      }
      std::cout.flush();
      iterators.push_back(it);
      condition.notify_one();
   }
}

template <typename Integer, typename TagType>
Row<Integer> panda::List<Integer, TagType>::get() const
{
   if ( empty() ) // abort
   {
      const auto it = rows.insert(Row<Integer>{}).first;
      std::unique_lock<std::mutex> lock(mutex);
      iterators.push_back(it);
      condition.notify_all();
   }
   std::unique_lock<std::mutex> lock(mutex);
   condition.wait(lock, [&](){ return !iterators.empty(); });
   ++workers;
   const auto row = *iterators.front();
   if ( !row.empty() )
   {
      iterators.erase(iterators.begin());
   }
   #ifdef PRINT_DONE_COUNTER
   if ( !row.empty() )
   {
      ++counter;
      #if HAS_FEATURE_THREAD_LOCAL == 0
      indices[std::this_thread::get_id()] = counter;
      #else
      index = counter;
      #endif
      std::stringstream stream;
      stream << "Processing #" << counter << " of at least " << rows.size();
      stream << " class" << ((rows.size() == 1) ? "" : "es") << '\n';
      std::cerr << stream.str();
   }
   #endif
   return row;
}

template <typename Integer, typename TagType>
panda::List<Integer, TagType>::List(const Names& names_, const std::optional<VertexGroup>& vertex_group_, const Matrix<Integer>& vertices_)
:
   names(names_),
   vertex_group(vertex_group_),
   vertices(vertices_),
   mutex(),
   workers(1),
   condition(),
   rows(),
   iterators(),
   counter(0),
   seen_supports()
{
}

template <typename Integer, typename TagType>
bool panda::List<Integer, TagType>::empty() const
{
   const std::lock_guard<std::mutex> lock(mutex);
   return workers == 0 && iterators.empty();
}


#include <stdio.h>
#include <set>
#define _OPENMP

#ifdef _OPENMP
#include <omp.h>
#endif

class parallel_set{
 private:
   std::set<int> flags;
 #ifdef _OPENMP
   omp_lock_t lock;
 #endif
 public:
   parallel_set() : flags(){
 #ifdef _OPENMP
     omp_init_lock(&lock);
 #endif
   }
   ~parallel_set(){
 #ifdef _OPENMP
     omp_destroy_lock(&lock);
 #endif
   }
   
   bool insert(int c){
   #ifdef _OPENMP
     omp_set_lock(&lock);
   #endif
     bool found = flags.find(c) != flags.end();
     if(!found) flags.insert(c);
   #ifdef _OPENMP
     omp_unset_lock(&lock);
   #endif
     return found;
   }
 };
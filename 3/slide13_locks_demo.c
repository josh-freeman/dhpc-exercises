#include <stdio.h>
#include <omp.h>
#include <unistd.h>

// Global variables for demonstration
int shared_counter = 0;
omp_lock_t simple_lock;
omp_nest_lock_t nested_lock;

// Example 1: Without any synchronization (race condition)
void test_no_sync()
{
    printf("\n=== Test 1: No Synchronization (Race Condition) ===\n");
    shared_counter = 0;

#pragma omp parallel num_threads(4)
    {
        for (int i = 0; i < 1000; i++)
        {
            shared_counter++; // UNSAFE!
        }
    }

    printf("Expected: 4000, Got: %d\n", shared_counter);
    printf("Result: %s\n", shared_counter == 4000 ? "CORRECT (got lucky!)" : "INCORRECT (race condition)");
}

// Example 2: Using critical section
void test_critical()
{
    printf("\n=== Test 2: Using #pragma omp critical ===\n");
    shared_counter = 0;

#pragma omp parallel num_threads(4)
    {
        for (int i = 0; i < 1000; i++)
        {
#pragma omp critical
            {
                shared_counter++; // SAFE
            }
        }
    }

    printf("Expected: 4000, Got: %d\n", shared_counter);
    printf("Result: %s\n", shared_counter == 4000 ? "CORRECT" : "INCORRECT");
}

// Example 3: Using simple lock
void test_simple_lock()
{
    printf("\n=== Test 3: Using Simple Lock ===\n");
    shared_counter = 0;
    omp_init_lock(&simple_lock);

#pragma omp parallel num_threads(4)
    {
        for (int i = 0; i < 1000; i++)
        {
            omp_set_lock(&simple_lock);
            shared_counter++; // SAFE
            omp_unset_lock(&simple_lock);
        }
    }

    printf("Expected: 4000, Got: %d\n", shared_counter);
    printf("Result: %s\n", shared_counter == 4000 ? "CORRECT" : "INCORRECT");

    omp_destroy_lock(&simple_lock);
}

// Example 4: Using omp_test_lock (non-blocking)
void test_lock_trylock()
{
    printf("\n=== Test 4: Using omp_test_lock (non-blocking) ===\n");
    omp_init_lock(&simple_lock);

#pragma omp parallel num_threads(4)
    {
        int attempts = 0;
        int successes = 0;

        for (int i = 0; i < 100; i++)
        {
            attempts++;
            if (omp_test_lock(&simple_lock))
            {
                // Got the lock!
                successes++;
                usleep(100); // Simulate some work
                omp_unset_lock(&simple_lock);
            }
            else
            {
                // Lock was busy, didn't block
            }
        }

#pragma omp critical
        {
            printf("Thread %d: %d/%d lock acquisitions succeeded\n",
                   omp_get_thread_num(), successes, attempts);
        }
    }

    omp_destroy_lock(&simple_lock);
}

// Recursive function to demonstrate nested locks
int recursive_increment(int depth, omp_nest_lock_t *lock)
{
    if (depth <= 0)
        return 0;

    omp_set_nest_lock(lock); // Can acquire multiple times!
    int result = 1 + recursive_increment(depth - 1, lock);
    omp_unset_nest_lock(lock);

    return result;
}

// Example 5: Nested locks (re-entrant)
void test_nested_lock()
{
    printf("\n=== Test 5: Using Nested Lock (Re-entrant) ===\n");
    omp_init_nest_lock(&nested_lock);

#pragma omp parallel num_threads(4)
    {
        int depth = 5;
        int result = recursive_increment(depth, &nested_lock);

#pragma omp critical
        {
            printf("Thread %d: Recursively acquired lock %d times\n",
                   omp_get_thread_num(), result);
        }
    }

    omp_destroy_nest_lock(&nested_lock);

    printf("\nNote: Simple locks would DEADLOCK in this scenario!\n");
}

// Example 6: Demonstrating the difference - simple lock would deadlock
void test_simple_lock_deadlock_demo()
{
    printf("\n=== Test 6: Why Simple Locks Can't Be Nested ===\n");
    printf("If we tried to acquire a simple lock twice in the same thread:\n");
    printf("  omp_set_lock(&lock);\n");
    printf("  omp_set_lock(&lock);  // <-- DEADLOCK! Same thread blocks itself\n");
    printf("  omp_unset_lock(&lock);\n");
    printf("  omp_unset_lock(&lock);\n");
    printf("\nWith nested locks, the same thread can acquire multiple times!\n");
}

int main()
{
    printf("=== OpenMP Locks Demo (Slide 13) ===\n");
    printf("Demonstrating different synchronization mechanisms\n");

    test_no_sync();
    test_critical();
    test_simple_lock();
    test_lock_trylock();
    test_nested_lock();
    test_simple_lock_deadlock_demo();

    printf("\n=== Summary ===\n");
    printf("1. No sync: Fast but UNSAFE (race conditions)\n");
    printf("2. Critical: Easy to use, safe, but less flexible\n");
    printf("3. Simple locks: More control, must manage manually\n");
    printf("4. omp_test_lock: Non-blocking, useful for try-lock patterns\n");
    printf("5. Nested locks: Allow re-entrance by same thread\n");
    printf("\nKey takeaway: Locks give fine-grained control but require careful management!\n");

    return 0;
}

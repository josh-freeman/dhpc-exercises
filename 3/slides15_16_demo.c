#include <stdio.h>
#include <omp.h>
#include <unistd.h>

void simulate_work(int milliseconds, const char* task_name) {
    printf("  [%s] Starting work...\n", task_name);
    usleep(milliseconds * 1000);
    printf("  [%s] Finished work!\n", task_name);
}

// Demonstration of SHALLOW synchronization (taskwait)
void demo_taskwait() {
    printf("\n=== SLIDE 15: Shallow Synchronization (taskwait) ===\n");
    printf("Task A will spawn B and C. Task C will spawn C.1 and C.2\n");
    printf("Taskwait in A only waits for B and C (not C.1, C.2)\n\n");

    #pragma omp parallel
    #pragma omp single
    {
        printf("Task A: Starting\n");

        // Spawn child tasks B and C
        #pragma omp task
        {
            printf("Task B: Started\n");
            simulate_work(100, "Task B");
        }

        #pragma omp task
        {
            printf("Task C: Started\n");
            simulate_work(50, "Task C (itself)");

            // Task C spawns grandchildren C.1 and C.2
            #pragma omp task
            {
                printf("  Task C.1: Started (grandchild of A)\n");
                simulate_work(200, "Task C.1");
            }

            #pragma omp task
            {
                printf("  Task C.2: Started (grandchild of A)\n");
                simulate_work(200, "Task C.2");
            }

            printf("Task C: Done spawning C.1 and C.2 (NOT waiting for them)\n");
            // Note: Task C does NOT have a taskwait here, so it completes
            // immediately after spawning C.1 and C.2
        }

        // SHALLOW WAIT: Only waits for B and C to complete
        // Does NOT wait for C.1 and C.2!
        #pragma omp taskwait
        printf("\n>>> Task A: taskwait completed!\n");
        printf(">>> B and C are done, but C.1 and C.2 might still be running!\n\n");

        // Sleep to show C.1 and C.2 still running
        sleep(1);
        printf("Task A: Exiting\n");
    }

    printf("\n>>> All tasks eventually complete at end of parallel region\n");
}

// Demonstration of DEEP synchronization (taskgroup)
void demo_taskgroup() {
    printf("\n\n=== SLIDE 16: Deep Synchronization (taskgroup) ===\n");
    printf("Task A will spawn B and C. Task C will spawn C.1 and C.2\n");
    printf("Taskgroup in A waits for B, C, C.1, AND C.2 (all descendants)\n\n");

    #pragma omp parallel
    #pragma omp single
    {
        printf("Task A: Starting\n");

        // Everything inside taskgroup: wait for ALL descendants
        #pragma omp taskgroup
        {
            // Spawn child tasks B and C
            #pragma omp task
            {
                printf("Task B: Started\n");
                simulate_work(100, "Task B");
            }

            #pragma omp task
            {
                printf("Task C: Started\n");
                simulate_work(50, "Task C (itself)");

                // Task C spawns grandchildren C.1 and C.2
                #pragma omp task
                {
                    printf("  Task C.1: Started (grandchild of A)\n");
                    simulate_work(200, "Task C.1");
                }

                #pragma omp task
                {
                    printf("  Task C.2: Started (grandchild of A)\n");
                    simulate_work(200, "Task C.2");
                }

                printf("Task C: Done spawning C.1 and C.2\n");
            }
        } // End of taskgroup - DEEP WAIT happens here!

        printf("\n>>> Task A: taskgroup completed!\n");
        printf(">>> B, C, C.1, AND C.2 are ALL guaranteed to be done!\n\n");
        printf("Task A: Exiting\n");
    }
}

// Side-by-side comparison
void comparison() {
    printf("\n\n=== KEY DIFFERENCES ===\n");
    printf("\n┌─────────────────────┬──────────────────────┬──────────────────────┐\n");
    printf("│                     │   taskwait (Slide 15)│  taskgroup (Slide 16)│\n");
    printf("├─────────────────────┼──────────────────────┼──────────────────────┤\n");
    printf("│ What it waits for   │ Direct children only │ All descendants      │\n");
    printf("│ Waits for B, C      │         YES          │         YES          │\n");
    printf("│ Waits for C.1, C.2  │         NO           │         YES          │\n");
    printf("│ Synchronization     │       Shallow        │        Deep          │\n");
    printf("└─────────────────────┴──────────────────────┴──────────────────────┘\n");

    printf("\nAnalogy:\n");
    printf("  taskwait:   \"Wait for my kids to finish\" (but not grandkids)\n");
    printf("  taskgroup:  \"Wait for entire family tree to finish\"\n");
}

int main() {
    printf("=== Understanding Task Synchronization: Slides 15 vs 16 ===\n");

    demo_taskwait();
    sleep(1);  // Let tasks finish before next demo

    demo_taskgroup();

    comparison();

    return 0;
}

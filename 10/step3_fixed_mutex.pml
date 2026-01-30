/*
 * Step 3: Peterson's Algorithm --- Correct Mutual Exclusion
 *
 * Goal: Model Peterson's mutual exclusion algorithm for 2 processes
 *       and verify with SPIN that mutual exclusion holds.
 *
 * Peterson's Algorithm (for process i, with j = 1 - i):
 *   Lock:
 *     1. flag[i] = 1           (I'm interested)
 *     2. victim = i            (I yield to the other)
 *     3. Wait while (flag[j] && victim == i)
 *   Unlock:
 *     4. flag[i] = 0
 *
 * Verify:  spin -a step3_fixed_mutex.pml && gcc -O3 -o pan pan.c && ./pan
 * Expected: no errors found (Peterson's is correct!)
 */

/* TODO 1: Declare shared variables:
 *   - bool flag[2]     (interest flags, one per process)
 *   - byte victim       (who yields)
 *   - byte mutex        (counts processes in critical section)
 */


/* TODO 2: Declare proctype P(bit i) implementing Peterson's algorithm:
 *   a) Compute j = 1 - i
 *   b) Lock:
 *      - Set flag[i] = 1
 *      - Set victim = i
 *      - Block until !(flag[j] && victim == i)
 *   c) Critical section:
 *      - Increment mutex
 *      - printf("P(%d) in critical section\n", i)
 *      - Decrement mutex
 *   d) Unlock:
 *      - Set flag[i] = 0
 */


bit flag[2];
byte mutex;
byte victim;

proctype P(int i)
{
    int j = 1-i;

    flag[i] = 1;
    victim = i;

    !(flag[j] && victim == i)


    mutex++;
    printf("MSC: P(%d) has entered crit. section.\n", i);
    mutex--;
    flag[i] = 0;
}

/* TODO 3: Declare proctype monitor() that:
 *   - asserts mutex != 2
 *   (this checks that both processes are never in the
 *    critical section simultaneously)
 */
proctype monitor()
{
    assert(mutex != 2);
}

/* TODO 4: Write an init block that:
 *   - runs P(0), P(1), and monitor()
 */
init
{
    run P(0);
    run P(1);
    run monitor();
}

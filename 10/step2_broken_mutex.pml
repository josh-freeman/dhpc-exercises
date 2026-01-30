/*
 * Step 2: Broken Mutual Exclusion
 *
 * Goal: Model a naive (broken) mutex using a single flag,
 *       then use SPIN verification to find the bug.
 *
 * The broken protocol:
 *   1. Wait until flag != 1    (blocks until flag is 0)
 *   2. Set flag = 1            (claim the lock)
 *   3. Enter critical section
 *   4. Set flag = 0            (release the lock)
 *
 * The bug: steps 1 and 2 are NOT atomic, so both processes
 * can pass the check before either sets the flag.
 *
 * Simulate:  spin step2_broken_mutex.pml
 * Verify:    spin -a step2_broken_mutex.pml && gcc -O3 -o pan pan.c && ./pan
 * Replay:    spin -t -p step2_broken_mutex.pml
 */

/* TODO 1: Declare shared variables:
 *   - bit flag          (signals lock is held)
 *   - byte mutex        (counts how many processes are in the critical section)
 */
bit flag[2];
byte mutex;

/* TODO 2: Declare proctype P(int i) implementing the broken protocol:
 *   a) Block until flag != 1
 *      (remember: an expression as a statement blocks until it is non-zero)
 *   b) Set flag = 1
 *   c) Increment mutex (entering critical section)
 *   d) printf("P(%d) entered critical section\n", i)
 *   e) Decrement mutex (leaving critical section)
 *   f) Set flag = 0
 */
proctype P(int i)
{
    flag[i] != 1;
    flag[i] = 1;

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

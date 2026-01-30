/*
 * Step 1: Hello World in Promela
 *
 * Goal: Learn the basics of Promela processes.
 *   - Declare a proctype
 *   - Use init to create processes with run
 *   - Use _pid to identify processes
 *
 * Run with: spin step1_hello.pml
 */

/* TODO 1: Declare a proctype called "Hello" that:
 *   - prints "Hello process, my pid is: <pid>\n"
 *   (use printf and the built-in _pid variable)
 */
active proctype Hello(){
    printf("Hello process, my pid is: %d\n", _pid);
}

/* TODO 2: Write an init block that:
 *   a) prints "init process, my pid is: <pid>\n"
 *   b) declares an int variable lastpid
 *   c) uses "run Hello()" to create two Hello processes
 *      (store the return value of the second run in lastpid)
 *   d) prints "last pid was: <lastpid>\n"
 */
init {
    int lastpid;
    printf("init process, my pid is: %d\n", _pid);
    lastpid = run Hello();
    printf("last pid was %d\n", lastpid);
}
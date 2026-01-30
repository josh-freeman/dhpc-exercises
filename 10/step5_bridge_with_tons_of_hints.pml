/*
 * Step 5: The Bridge Crossing Riddle
 *
 * 4 friends want to cross a bridge at night.
 *   - The bridge holds at most 2 people at a time.
 *   - They need a torch to cross; they have exactly one.
 *   - Crossing times: 5, 10, 20, 25 minutes.
 *   - A pair moves at the speed of the slower person.
 *   - Goal: all 4 cross within 60 minutes.
 *
 * Strategy: model all possible moves non-deterministically,
 * then assert it's IMPOSSIBLE to get everyone across in time.
 * SPIN's counterexample IS the solution.
 *
 * Verify:  spin -a step5_bridge.pml && gcc -O3 -o pan pan.c && ./pan
 * Replay:  spin -t -p step5_bridge.pml
 */

#define N       4
#define LIMIT   60
#define START   0
#define GOAL    1

/* Crossing times for each person */
byte time[N];

/* TODO 1: Declare state variables:
 *   - byte pos[N]      (position of each person: START or GOAL)
 *   - byte torch        (which side the torch is on: START or GOAL)
 *   - byte elapsed      (total elapsed time)
 */


/* Inline helper: returns max of two bytes */
inline max(a, b, result) {
    if
    :: (a >= b) -> result = a
    :: (a <  b) -> result = b
    fi
}

/* TODO 2: Declare an active proctype Solve() that:
 *
 *   a) Initialize: set time[0..3] = 5,10,20,25.
 *      Set torch = START, elapsed = 0, all pos[i] = START.
 *
 *   b) Use a do-loop to repeat crossing steps until done.
 *      Inside the loop:
 *
 *      FORWARD TRIP (torch is on START side):
 *        - Use an if-statement with guards to non-deterministically
 *          pick two people i,j (i < j) who are on the START side.
 *          There are 6 possible pairs from 4 people: (0,1), (0,2),
 *          (0,3), (1,2), (1,3), (2,3).
 *        - For each valid choice:
 *            guard: (pos[i] == START && pos[j] == START && torch == START)
 *            body:  move both to GOAL, move torch to GOAL,
 *                   add max(time[i], time[j]) to elapsed
 *        - Print who crossed and the elapsed time.
 *
 *      RETURN TRIP (torch is on GOAL side):
 *        - Non-deterministically pick one person k on the GOAL side
 *          who hasn't finished (or just anyone on GOAL side).
 *        - guard: (pos[k] == GOAL && torch == GOAL)
 *        - Move k back to START, move torch to START,
 *          add time[k] to elapsed.
 *        - Print who returned and the elapsed time.
 *
 *      CHECK if everyone is at GOAL:
 *        - If pos[0]==GOAL && pos[1]==GOAL && pos[2]==GOAL && pos[3]==GOAL
 *          then break out of the loop.
 *        - Also break if elapsed > LIMIT (prune hopeless branches).
 *
 *   c) After the loop, assert that the goal is IMPOSSIBLE:
 *      assert(!(pos[0]==GOAL && pos[1]==GOAL &&
 *               pos[2]==GOAL && pos[3]==GOAL &&
 *               elapsed <= LIMIT))
 *
 *      SPIN's counterexample to this assertion IS the solution.
 */

/*
 * Step 4: Channel Communication --- Simplified Alternating Bit Protocol
 *
 * Goal: Model message passing between a Sender and Receiver
 *       using Promela channels with an alternating sequence bit.
 *
 * Protocol:
 *   - Sender sends MSG with a sequence bit (0 or 1)
 *   - Receiver receives the MSG and sends back an ACK with the same bit
 *   - Sender waits for the ACK, then flips the bit
 *   - Repeat for a fixed number of messages
 *
 * Run with: spin step4_channels.pml
 */

/* TODO 1: Declare an mtype with values MSG and ACK */
mtype = {MSG, ACK};

/* TODO 2: Declare two channels:
 *   - toReceiver: capacity 1, carries {mtype, byte}
 *   - toSender:   capacity 1, carries {mtype, byte}
 */
chan toReceiver = [1] of {mtype, byte};
chan toSender = [1] of {mtype, byte};

#define NUM_MSGS 4

/* TODO 3: Declare proctype Sender() that:
 *   a) Declares a local variable: bit seqbit = 0
 *   b) Declares a local byte counter = 0
 *   c) Loops NUM_MSGS times:
 *      - Sends MSG with seqbit on toReceiver
 *      - printf("Sender: sent MSG seq=%d\n", seqbit)
 *      - Receives ACK with matching seqbit from toSender
 *        (use the seqbit value as a constant in the receive
 *         to do message matching)
 *      - printf("Sender: got ACK seq=%d\n", seqbit)
 *      - Flips seqbit: seqbit = 1 - seqbit
 *      - Increments counter
 *   d) After the loop, assert(counter == NUM_MSGS)
 */
proctype Sender(){
    bit seqbit = 0;
    byte counter = 0;
    do
    :: counter < NUM_MSGS ->
        toReceiver ! MSG, seqbit;
        printf("Sender: sent MSG seq=%d\n", seqbit);
        toSender ? ACK, seqbit;
        printf("Sender: got ACK seq=%d\n", seqbit);
        seqbit = 1 - seqbit;
        counter ++;
    :: counter >= NUM_MSGS -> break;
    od;
}

/* TODO 4: Declare proctype Receiver() that:
 *   a) Declares a local variable: byte recvbit
 *   b) Declares a local byte counter = 0
 *   c) Loops NUM_MSGS times:
 *      - Receives a MSG from toReceiver, storing the
 *        sequence bit in recvbit
 *      - printf("Receiver: got MSG seq=%d\n", recvbit)
 *      - Sends ACK with recvbit back on toSender
 *      - printf("Receiver: sent ACK seq=%d\n", recvbit)
 *      - Increments counter
 *   d) After the loop, assert(counter == NUM_MSGS)
 */
proctype Receiver(){
    byte recvbit;
    byte counter = 0;
    do 
    :: counter < NUM_MSGS ->
        toReceiver ? MSG, recvbit;
        printf("Receiver: got MSG seq=%d\n", recvbit);
        toSender ! ACK, recvbit;
        counter ++;
    :: counter >= NUM_MSGS -> break;
    od;
}

/* TODO 5: Write an init block that runs Sender() and Receiver() */
init {
    run Sender();
    run Receiver();
}
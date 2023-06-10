class PrimeCountStaticBlockThread extends Thread {
    int lo;
    int hi;
    int counter;
    long timeDiff;

    // Constructor, gets number range of block
    PrimeCountStaticBlockThread(int l, int h) {
        lo = l;
        hi = h;
        counter = 0;
    }

    // run()
    public void run() {
        long startTime = System.currentTimeMillis();
        for (int i = lo; i < hi; i++) { // loop in block range
            if (isPrime(i)) // check prime
                counter++;
        }
        long endTime = System.currentTimeMillis();
        timeDiff = endTime - startTime;
    }

    private static boolean isPrime(int x) {
        int i;
        if (x <= 1)
            return false;
        for (i = 2; i < x; i++) {
            if (x % i == 0)
                return false;
        }
        return true;
    }
}

public class pc_static_block {
    private static int NUM_END = 200000;
    private static int NUM_THREADS = 8;
    private static long[] thread_exec_times;

    public static void main(String[] args) {
        if (args.length == 1) { // if 1 argument
            NUM_THREADS = Integer.parseInt(args[0]); // set as NUM_THREADS
        }
        if (args.length == 2) {
            NUM_THREADS = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
        }
        System.out.println("Counting prime numbers between 1..." + (NUM_END - 1) + " with " + NUM_THREADS
                + " threads using static load balancing (block decomposition)");

        thread_exec_times = new long[NUM_THREADS];
        int counter = 0;
        long startTime = System.currentTimeMillis();
        counter = primeCount();
        long endTime = System.currentTimeMillis();
        long timeDiff = endTime - startTime;
        for (int i = 0; i < NUM_THREADS; i++) {
            System.out.println("Thread #" + i + " Execution Time:" + thread_exec_times[i] + "ms");
        }
        System.out.println("Program Execution Time: " + timeDiff + "ms");
        System.out.println("1..." + (NUM_END - 1) + " prime# counter=" + counter);
    }

    private static int primeCount() {
        int counter = 0;
        int i;
        PrimeCountStaticBlockThread[] primeCountSerialThreads = new PrimeCountStaticBlockThread[NUM_THREADS]; // thread array

        for (i = 0; i < NUM_THREADS; i++) {
            primeCountSerialThreads[i] = new PrimeCountStaticBlockThread(i * NUM_END / NUM_THREADS,
                    (i + 1) * NUM_END / NUM_THREADS); // make thread
            primeCountSerialThreads[i].start(); // run thread
        }

        try {
            for (i = 0; i < NUM_THREADS; i++) {
                primeCountSerialThreads[i].join(); // wait for thread end
                counter += primeCountSerialThreads[i].counter;
                thread_exec_times[i] = primeCountSerialThreads[i].timeDiff;
            }
        } catch (InterruptedException e) {
        }

        return counter;
    }
}

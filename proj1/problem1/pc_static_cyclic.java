class PrimeCountStaticCyclicThread extends Thread {
    int lo;
    int hi;
    int thread_idx;
    int num_threads;
    int task_size;
    int counter;
    long timeDiff;

    // Constructor, gets thread index, number of threads, and task size
    PrimeCountStaticCyclicThread(int l, int h, int i, int n, int s) {
        lo = l;
        hi = h;
        thread_idx = i;
        num_threads = n;
        task_size = s;
        counter = 0;
    }

    // run()
    public void run() {
        long startTime = System.currentTimeMillis();
        for (int i = lo + thread_idx * task_size; i < hi; i += num_threads * task_size) { // cyclic loop
            for (int j = i; j < i + task_size; j++) { // inner loop for task size
                if (isPrime(j))
                    counter++;
            }
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

public class pc_static_cyclic {
    private static int NUM_END = 200000;
    private static int NUM_THREADS = 8;
    private static int TASK_SIZE = 10;
    private static long[] thread_exec_times;

    public static void main(String[] args) {
        if (args.length == 1) { // if 1 argument
            NUM_THREADS = Integer.parseInt(args[0]); // set as NUM_THREADS
        }
        if (args.length == 2) {
            NUM_THREADS = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
        }
        if (args.length == 3) { // if 3 arguments
            NUM_THREADS = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
            TASK_SIZE = Integer.parseInt(args[2]); // set last as TASK_SIZE
        }
        System.out.println("Counting prime numbers between 1..." + (NUM_END - 1) + " with " + NUM_THREADS
                + " threads using static load balancing (cyclic decomposition - task size = " + TASK_SIZE + ")");

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
        PrimeCountStaticCyclicThread[] primeCountStaticCyclicThreads = new PrimeCountStaticCyclicThread[NUM_THREADS]; // thread array

        for (i = 0; i < NUM_THREADS; i++) {
            primeCountStaticCyclicThreads[i] = new PrimeCountStaticCyclicThread(1, NUM_END, i, NUM_THREADS, TASK_SIZE); // make thread
            primeCountStaticCyclicThreads[i].start(); // run thread
        }

        try {
            for (i = 0; i < NUM_THREADS; i++) {
                primeCountStaticCyclicThreads[i].join(); // wait for thread end
                counter += primeCountStaticCyclicThreads[i].counter;
                thread_exec_times[i] = primeCountStaticCyclicThreads[i].timeDiff;
            }
        } catch (InterruptedException e) {
        }

        return counter;
    }
}

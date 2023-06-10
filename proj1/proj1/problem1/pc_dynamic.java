// Dynamic number loader for prime number count thread
class DynamicCheckNumberLoader {
    int lo;
    int hi;
    int task_size;
    boolean task_end;
    private int current_num;

    // Constructor
    DynamicCheckNumberLoader(int l, int h, int s) {
        lo = l;
        hi = h;
        task_size = s;
        current_num = l;
        task_end = false;
    }

    // Synchronized getter and update
    synchronized int getCurrentCheckNum() {
        if (task_end) { // if no more number to load
            return 0;
        } else {
            if (current_num >= hi) { // if number reached end
                task_end = true;
                current_num = 0;
            }
            int current = current_num;
            current_num += task_size; // add task size to current number
            return current;
        }
    }
}

class PrimeCountDynamicThread extends Thread {
    int hi;
    int task_size;
    int counter;
    long timeDiff;
    DynamicCheckNumberLoader dynamicCheckNumberLoader;

    // Constructor, gets task size and a dynamic number loader
    PrimeCountDynamicThread(int h, int s, DynamicCheckNumberLoader d) {
        hi = h;
        task_size = s;
        dynamicCheckNumberLoader = d;
        counter = 0;
    }

    // run()
    public void run() {
        long startTime = System.currentTimeMillis();
        int current = 1;
        while (current > 0) {
            current = dynamicCheckNumberLoader.getCurrentCheckNum(); // get number from dynamic loader
            if (current == 0) { // if loader returns 0
                break;
            } else {
                for (int i = 0; i < task_size; i++) {
                    if (current + i >= hi) // if number out of range
                        break;
                    if (isPrime(current + i)) // check prime
                        counter++;
                }
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

public class pc_dynamic {
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
                + " threads using dynamic load balancing (task size = " + TASK_SIZE + ")");

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
        DynamicCheckNumberLoader dynamicCheckNumberLoader = new DynamicCheckNumberLoader(1, NUM_END, TASK_SIZE); // dynamic loader
        PrimeCountDynamicThread[] primeCountDynamicThreads = new PrimeCountDynamicThread[NUM_THREADS]; // thread array

        for (i = 0; i < NUM_THREADS; i++) {
            primeCountDynamicThreads[i] = new PrimeCountDynamicThread(NUM_END, TASK_SIZE, dynamicCheckNumberLoader); // make thread
            primeCountDynamicThreads[i].start(); // run thread
        }

        try {
            for (i = 0; i < NUM_THREADS; i++) {
                primeCountDynamicThreads[i].join(); // wait for thread end
                counter += primeCountDynamicThreads[i].counter;
                thread_exec_times[i] = primeCountDynamicThreads[i].timeDiff;
            }
        } catch (InterruptedException e) {
        }

        return counter;
    }
}

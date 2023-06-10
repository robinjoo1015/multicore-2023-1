import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;

class PrimeCountThread extends Thread {
    int counter;
    BlockingQueue queue;

    PrimeCountThread(String name, BlockingQueue q) {
        super(name);
        queue = q;
    }

    public void run() {
        int current = 1;
        try {
            while (current > 0) {
                current = (int) queue.take();
                if (current > 0)
                    System.out.println(this.getName() + " Consumed: " + current);
                if (isPrime(current))
                    counter++;
            }
        } catch (InterruptedException e) {
        }
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

public class ex1 {
    private static int NUM_END = 30;
    private static int NUM_THREADS = 4;

    public static void main(String[] args) {
        System.out.println("Counting prime numbers between 1..." + (NUM_END - 1));
        int counter = 0;
        counter = primeCount();
        System.out.println("1..." + (NUM_END - 1) + " prime# counter=" + counter);
    }

    private static int primeCount() {
        int counter = 0;
        int i;
        PrimeCountThread[] primeCountThreads = new PrimeCountThread[NUM_THREADS];
        BlockingQueue queue = new ArrayBlockingQueue<Integer>(4);
        Thread producerThread = new Thread(() -> {
            try {
                for (int j = 1; j < NUM_END; j++) {
                    queue.put(j);
                    System.out.println("Produced: " + j);
                }
                for (int j = 0; j < NUM_THREADS; j++) {
                    queue.put(0);
                }
            } catch (InterruptedException e) {
            }
        });
        producerThread.start();

        for (i = 0; i < NUM_THREADS; i++) {
            primeCountThreads[i] = new PrimeCountThread("Thread #" + i, queue);
            primeCountThreads[i].start();
        }

        try {
            for (i = 0; i < NUM_THREADS; i++) {
                primeCountThreads[i].join();
                counter += primeCountThreads[i].counter;
            }
        } catch (InterruptedException e) {
        }

        return counter;
    }
}
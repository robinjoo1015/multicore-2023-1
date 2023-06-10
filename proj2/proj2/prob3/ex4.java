import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

class SleepWaitThread extends Thread {
    CyclicBarrier barrier;

    public SleepWaitThread(String name, CyclicBarrier b) {
        super(name);
        barrier = b;
    }

    public void run() {
        for (int i = 0; i < 5; i++) {
            try {
                int rand = (int) (Math.random() * 5);
                System.out.println(getName() + " sleeping for " + rand + "s");
                sleep(rand * 1000);
            } catch (InterruptedException e) {
            }
            System.out.println("                            " + getName() + " waiting at barrier");
            try {
                barrier.await();
            } catch (InterruptedException | BrokenBarrierException e) {
            }
        }
    }
}

public class ex4 {
    public static void main(String[] args) {
        CyclicBarrier barrier = new CyclicBarrier(4, () -> {
            System.out.println("    ******* All threads finished, continuing *******    ");
        });
        SleepWaitThread swt0 = new SleepWaitThread("Thread #0", barrier);
        SleepWaitThread swt1 = new SleepWaitThread("Thread #1", barrier);
        SleepWaitThread swt2 = new SleepWaitThread("Thread #2", barrier);
        SleepWaitThread swt3 = new SleepWaitThread("Thread #3", barrier);
        swt0.start();
        swt1.start();
        swt2.start();
        swt3.start();
        try {
            swt0.join();
            swt1.join();
            swt2.join();
            swt3.join();
        } catch (InterruptedException e) {
        }
    }
}
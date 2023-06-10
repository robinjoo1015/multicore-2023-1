import java.util.concurrent.atomic.*;

public class AtomicIntegerExample {

    public static void main(String[] args) {
        AtomicInteger atmint = new AtomicInteger(0);
        
        Thread producer = new Thread(new Producer(atmint));
        Thread consumer = new Thread(new Consumer(atmint));
        
        producer.start();
        consumer.start();
        try {
          producer.join();
          consumer.join();
        } catch (InterruptedException s) {}
        System.out.println("result: atmint="+atmint.get());
    }
}

class Producer implements Runnable{
    
    private AtomicInteger atm;
    
    public Producer(AtomicInteger atmval) {
        this.atm = atmval;
    }

    public void run() {
        
        for (int i = 0; i < 10000002; i++) {
		atm.incrementAndGet(); // atm++;
        }
    }
}

class Consumer implements Runnable{
    
    private AtomicInteger atm;
    
    public Consumer(AtomicInteger atmval) {
        this.atm = atmval;
    }

    public void run() {
        for (int i = 0; i < 10000000; i++) {
		atm.decrementAndGet(); // atm--;
        }
    }
}

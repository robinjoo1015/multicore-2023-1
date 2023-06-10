import java.util.concurrent.atomic.AtomicInteger;

class AtomicIntegerThread extends Thread {
    AtomicInteger sharedInt;

    AtomicIntegerThread(String name, AtomicInteger si) {
        super(name);
        sharedInt = si;
    }

    public void run() {
        String name = getName();
        for (int i = 0; i < 10; i++) {
            try {
                sleep((int) (Math.random() * 100));
            } catch (InterruptedException e) {
            }
            switch (name) {
                case "get": {
                    System.out.println("AtomicInteger.get(): " + sharedInt.get());
                    break;
                }
                case "set": {
                    int rand = (int) (Math.random() * 10);
                    sharedInt.set(rand);
                    System.out.println("AtomicInteger.set(" + rand + ")");
                    break;
                }
                case "getAndAdd": {
                    int rand = (int) (Math.random() * 10);
                    System.out.println("AtomicInteger.getAndAdd(" + rand + "): " + sharedInt.getAndAdd(rand));
                    break;
                }
                case "addAndGet": {
                    int rand = (int) (Math.random() * 10);
                    System.out.println("AtomicInteger.addAndGet(" + rand + "): " + sharedInt.addAndGet(rand));
                    break;
                }
            }
        }
    }
}

public class ex3 {
    public static void main(String[] args) {
        AtomicInteger sharedInt = new AtomicInteger(1);

        AtomicIntegerThread ait0 = new AtomicIntegerThread("get", sharedInt);
        AtomicIntegerThread ait1 = new AtomicIntegerThread("set", sharedInt);
        AtomicIntegerThread ait2 = new AtomicIntegerThread("getAndAdd", sharedInt);
        AtomicIntegerThread ait3 = new AtomicIntegerThread("addAndGet", sharedInt);
        ait0.start();
        ait1.start();
        ait2.start();
        ait3.start();
        try {
            ait0.join();
            ait1.join();
            ait2.join();
            ait3.join();
        } catch (InterruptedException e) {
        }
    }
}
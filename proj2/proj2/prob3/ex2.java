import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

class SharedInt {
    private int value = 1;

    public void increment() {
        value += 1;
    }

    public void multiply() {
        value *= 2;
    }

    public int getValue() {
        return value;
    }
}

class ReaderThread extends Thread {
    ReadWriteLock lock;
    SharedInt sharedInt;

    ReaderThread(String name, ReadWriteLock l, SharedInt si) {
        super(name);
        lock = l;
        sharedInt = si;
    }

    public void run() {
        for (int i = 0; i < 5; i++) {
            try {
                sleep((int) (Math.random() * 100));
            } catch (InterruptedException e) {
            }
            lock.readLock().lock();
            try {
                System.out.println(getName() + " acquired readlock: " + sharedInt.getValue());
            } finally {
                System.out.println(getName() + " released readlock");
                lock.readLock().unlock();
            }
        }
    }
}

class WriterThread extends Thread {
    ReadWriteLock lock;
    SharedInt sharedInt;

    WriterThread(String name, ReadWriteLock l, SharedInt si) {
        super(name);
        lock = l;
        sharedInt = si;
    }

    public void run() {
        for (int i = 0; i < 5; i++) {
            try {
                sleep((int) (Math.random() * 100));
            } catch (InterruptedException e) {
            }
            lock.writeLock().lock();
            try {
                if (getName() == "WriterThread #0") {
                    sharedInt.increment();
                } else {
                    sharedInt.multiply();
                    ;
                }
                System.out.println(getName() + " acquired writelock");
            } finally {
                System.out.println(getName() + " released writelock");
                lock.writeLock().unlock();
            }
        }
    }
}

public class ex2 {
    public static void main(String[] args) {
        ReadWriteLock rwLock = new ReentrantReadWriteLock();
        SharedInt sharedInt = new SharedInt();

        WriterThread wt0 = new WriterThread("WriterThread #0", rwLock, sharedInt);
        WriterThread wt1 = new WriterThread("WriterThread #1", rwLock, sharedInt);
        ReaderThread rt0 = new ReaderThread("ReaderThread #0", rwLock, sharedInt);
        ReaderThread rt1 = new ReaderThread("ReaderThread #1", rwLock, sharedInt);
        ReaderThread rt2 = new ReaderThread("ReaderThread #2", rwLock, sharedInt);
        ReaderThread rt3 = new ReaderThread("ReaderThread #3", rwLock, sharedInt);
        wt0.start();
        wt1.start();
        rt0.start();
        rt1.start();
        rt2.start();
        rt3.start();
        try {
            wt0.join();
            wt1.join();
            rt0.join();
            rt1.join();
            rt2.join();
            rt3.join();
        } catch (InterruptedException e) {
        }
    }
}
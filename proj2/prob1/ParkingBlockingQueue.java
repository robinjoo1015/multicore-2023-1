import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

class ParkingGarage {
    private BlockingQueue queue;

    public ParkingGarage(int places) {
        queue = new ArrayBlockingQueue<String>(places);
    }

    public void enter() {
        try {
            queue.put("");
        } catch (InterruptedException e) {
        }
    }

    public void leave() {
        try {
            queue.take();
        } catch (InterruptedException e) {
        }
    }

    public int getPlaces() {
        return queue.remainingCapacity();
    }
}

class Car extends Thread {
    private ParkingGarage parkingGarage;

    public Car(String name, ParkingGarage p) {
        super(name);
        this.parkingGarage = p;
        start();
    }

    private void tryingEnter() {
        System.out.println(getName() + ": trying to enter");
    }

    private void justEntered() {
        System.out.println(getName() + ": just entered");
    }

    private void aboutToLeave() {
        System.out.println(getName() + ":                                     about to leave");
    }

    private void Left() {
        System.out.println(getName() + ":                                     have been left");
    }

    public void run() {
        while (true) {
            try {
                sleep((int) (Math.random() * 10000));
            } catch (InterruptedException e) {
            }
            tryingEnter();
            parkingGarage.enter();
            justEntered();
            try {
                sleep((int) (Math.random() * 20000));
            } catch (InterruptedException e) {
            }
            aboutToLeave();
            parkingGarage.leave();
            Left();
        }
    }
}

public class ParkingBlockingQueue {
    public static void main(String[] args) {
        ParkingGarage parkingGarage = new ParkingGarage(7);
        for (int i = 1; i <= 10; i++) {
            Car c = new Car("Car " + i, parkingGarage);
        }
    }
}

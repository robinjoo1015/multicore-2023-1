class Ex3Thread extends Thread {
    int lo, hi;
    double step;
    double sum;

    Ex3Thread(int l, int h, double s) {
        lo = l;
        hi = h;
        step = s;
    }

    public void run() {
        for (int i = lo; i < hi; i++) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
    }
}

public class ex3 {
    private static final int NUM_STEPS = 100000;
    private static final int NUM_THREAD = 10;

    public static void main(String[] args) {
        System.out.println("main thread start!");

        double step = 1.0 / NUM_STEPS;
        double pi = 0.0;
        Ex3Thread[] ex3Threads = new Ex3Thread [NUM_THREAD];
        for(int i=0;i<NUM_THREAD;i++) {
            ex3Threads[i] = new Ex3Thread(i*NUM_STEPS/NUM_THREAD, (i+1)*NUM_STEPS/NUM_THREAD, step);
            ex3Threads[i].start();
        }
        System.out.println("main thread calls join()!");
        for(int i=0; i < NUM_THREAD; ++i) {
            try { 
              ex3Threads[i].join();
            } catch (InterruptedException e) {}
          }
          System.out.println("Main thread ends!");
          for (int i = 0; i < NUM_THREAD; i++) {
              pi += ex3Threads[i].sum;
          }
          pi *= step;
          System.out.println("Pi=" + pi);

    }
}

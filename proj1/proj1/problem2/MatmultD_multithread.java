import java.util.*;

class MatmulThread extends Thread {
  int[][] ans;
  int[][] a;
  int[][] b;
  int thread_idx;
  int num_threads;
  long timeDiff;

  // Constructor, gets pointer of answer matrix and original matrices a and b,
  // gets thread index and number of threads for cyclic static load balancing
  MatmulThread(int[][] ans, int[][] a, int[][] b, int i, int n) {
    this.ans = ans;
    this.a = a;
    this.b = b;
    thread_idx = i;
    num_threads = n;
  }

  // run()
  public void run() {
    long startTime = System.currentTimeMillis();
    int n = a[0].length;
    int m = a.length;
    int p = b[0].length;
    for (int i = thread_idx; i < m; i += num_threads) { // cyclic loop
      for (int j = 0; j < p; j++) {
        for (int k = 0; k < n; k++) {
          ans[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    long endTime = System.currentTimeMillis();
    timeDiff = endTime - startTime;
  }
}

public class MatmultD_multithread {
  private static Scanner sc = new Scanner(System.in);
  private static int NUM_THREADS = 8;
  private static long[] thread_exec_times;

  public static void main(String[] args) {
    if (args.length == 1) // if 1 argument
      NUM_THREADS = Integer.valueOf(args[0]); // set as NUM_THREADS

    thread_exec_times = new long[NUM_THREADS];

    int a[][] = readMatrix();
    int b[][] = readMatrix();

    long startTime = System.currentTimeMillis();
    int[][] c = multMatrix(a, b);
    long endTime = System.currentTimeMillis();
    long timeDiff = endTime - startTime;
    for (int i = 0; i < NUM_THREADS; i++) { // print thread execution time
      System.out.println("Thread #" + i + " Execution Time:" + thread_exec_times[i] + "ms");
    }
    System.out.println();
    System.out.println("Program Execution Time: " + timeDiff + "ms"); // print program execution time
    System.out.println();
    printMatrix(c, false); // print sum of elements in result matrix
  }

  public static int[][] readMatrix() {
    int rows = sc.nextInt();
    int cols = sc.nextInt();
    int[][] result = new int[rows][cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result[i][j] = sc.nextInt();
      }
    }
    return result;
  }

  public static void printMatrix(int[][] mat, boolean printAll) {
    System.out.println("Matrix[" + mat.length + "][" + mat[0].length + "]");
    int rows = mat.length;
    int columns = mat[0].length;
    int sum = 0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        if (printAll) // if printAll print elements
          System.out.printf("%4d ", mat[i][j]);
        sum += mat[i][j];
      }
      if (printAll)
        System.out.println();
    }
    if (printAll)
      System.out.println();
    System.out.println("Matrix Sum = " + sum + "\n"); // print sum of elements
  }

  public static int[][] multMatrix(int a[][], int b[][]) {// a[m][n], b[n][p]
    if (a.length == 0)
      return new int[0][0];
    if (a[0].length != b.length)
      return null; // invalid dims

    int m = a.length;
    int p = b[0].length;
    int ans[][] = new int[m][p];
    int i;

    MatmulThread[] matmulThreads = new MatmulThread[NUM_THREADS]; // thread array
    for (i = 0; i < NUM_THREADS; i++) {
      matmulThreads[i] = new MatmulThread(ans, a, b, i, NUM_THREADS); // make thread
      matmulThreads[i].start(); // run thread
    }

    try {
      for (i = 0; i < NUM_THREADS; i++) {
        matmulThreads[i].join(); // wait for thread end
        thread_exec_times[i] = matmulThreads[i].timeDiff;
      }
    } catch (InterruptedException e) {
    }

    return ans;
  }
}

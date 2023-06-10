
public class ex2_serial {
  public static void main(String[] args) {
    int[] int_arr = new int [10000]; 
    int i,s;
    for (i=0;i<10000;i++) int_arr[i]=i+1;
    s=sum(int_arr);
    System.out.println("sum=" + s +"\n");
  }

  static int sum(int[] arr) {
    int i;
    int s=0;
    for (i=0;i<arr.length;i++) s+=arr[i];
    return s;
  }
}


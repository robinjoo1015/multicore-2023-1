#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>

#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace std::chrono;

int num_steps = 1000000000; 
double step;

// Make function object for calculating pi
struct pi_functor
{
    const double step;
    
    // Get step on constructor
    pi_functor(double _step) : step(_step) {}

    // Define operator method
    __host__ __device__
        double operator() (const int i) const {
            double x = ((double) i + 0.5) * step;
            return 4.0 / (1.0 + x * x);
        }
};

int main()
{
    double pi, sum = 0.0;

    step = 1.0 / (double) num_steps;
    
    // Start timer
    auto start_time = high_resolution_clock::now();

    // Allocate int vector with size num_steps on device
    thrust::device_vector<int> i(num_steps);
    // Set vector elements to 0, 1, 2, ...
    thrust::sequence(i.begin(), i.end());

    // Calculate sum with transform_reduce
    // First, calculate transformation with pi_functor function object
    // Then, add all vector elements with reduce
    sum = thrust::transform_reduce(i.begin(), i.end(), pi_functor(step), 0.0, thrust::plus<double>());

    pi = step * sum;

    // End timer, calculate execution time
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end_time - start_time);
    double execution_time = duration.count() / 1000000000.0;

    // Print execution time and result
    cout << "Execution Time : " << setprecision(11) <<  execution_time << "sec" << endl;
    cout << "pi=" << setprecision(11) << pi << endl;

    return 0;
}
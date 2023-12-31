#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>

struct saxpy_functor
{
    const int a;

    saxpy_functor(int _a) : a(_a) {}

    __host__ __device__
        int operator()(const int& x, const int& y) const {
            return a * x + y;
        }
};

void saxpy_fast(int A, thrust::device_vector<int>& X, thrust::device_vector<int>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(int A, thrust::device_vector<int>& X, thrust::device_vector<int>& Y)
{
    thrust::device_vector<int> temp(X.size());

    // temp <- A
    thrust::fill(temp.begin(), temp.end(), A);

    // temp <- A * X
    thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<int>());

    // Y <- A * X + Y
    thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<int>());
}

int main(void)
{
    // allocate three device_vectors with 10 elements
    thrust::device_vector<int> X(10);
    thrust::device_vector<int> Y(10);
    thrust::device_vector<int> Y2(10);
    thrust::device_vector<int> Y3(10);
    thrust::device_vector<int> Z(10);

    // initialize X to 0,1,2,3, ....
    thrust::sequence(X.begin(), X.end());

    // compute Y = -X
    thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<int>());

    // fill Z with twos
    thrust::fill(Z.begin(), Z.end(), 2);

    // compute Y = X mod 2
    thrust::transform(X.begin(), X.end(), Z.begin(), Y.begin(), thrust::modulus<int>());

    // replace all the ones in Y with tens
    thrust::replace(Y.begin(), Y.end(), 1, 10);


    // print X
    std::cout << "X = ";
    thrust::copy(X.begin(), X.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "\n";

    // print Y
    std::cout << "Y = ";
    thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "\n";

    std::cout << "Y=2*X+Y using saxpy_slow : ";
    saxpy_slow(2,X,Y);
    thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "\n";

    std::cout << "Y=3*X+Y using saxpy_fast : ";
    saxpy_fast(3,X,Y);
    thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "\n";

    return 0;
}


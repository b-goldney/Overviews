#include <iostream>
#include <cassert>
#include <vector>

using std::cout;
using std::endl;
using std::vector;


int fibonacci_naive(int n) {
    if (n <= 1)
        return n;

    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2);
}

int fibonacci_fast(int n) {
    // write your code here
    if (n == 0) {
        return 0;
    }
    vector<long long> fib_vector(n, 0);
    fib_vector[0] = 0; // assign 0 to first element of fibonacci sequence
    fib_vector[1] = 1; // assign 1 to second element of fibonacci sequence 
    for (int i =2; i <n+1; i++) {
        fib_vector[i] = fib_vector[i-1] + fib_vector[i-2];

        //cout << fib_vector[i-2] << " <<< fib_vector[i-2]" << endl;
        //cout << fib_vector[i-1] << " <<< fib_vector[i-1]" << endl;
        //cout << fib_vector[i] << " <<< fib_vector[i]" << endl;
    }
    //cout << n << " <<< n" << endl;
    //cout << fib_vector[3] << " <<< fib_vector[3]" << endl;
    return fib_vector[n];
}


int main() {
    int n = 0;
    std::cin >> n;

    std::cout << fibonacci_fast(n) << '\n';
    return 0;
}

/* This file demonstrates calculating the greatest common divisor (GCD)
 through a naive algorithm and Euclidian algorithm */

#include <iostream>

using std::cout;
using std::endl;


int gcd_naive(int a, int b) {
  int current_gcd = 1;
  for (int d = 2; d <= a && d <= b; d++) {
    if (a % d == 0 && b % d == 0) {
      if (d > current_gcd) {
        current_gcd = d;
      }
    }
  }
  return current_gcd;
}

// Euclidian algorithm
int gcd_fast(int a, int b) {
    int a_prime = -1;
    int b_prime = -1;

    if (b == 0) {
        b_prime = a;
    }

    while (b != 0) { // Comments below are for gcd_fast(20,15)
        b_prime = b; // 15, 5, 
        a_prime = a % b; // 5, 0,
        a = b_prime; // 15, 0
        b = a_prime; // 5, 0
        //cout << b_prime << " <<< b_prime" << endl;
    }

    return b_prime;
}


int main() {
  int a, b;
  std::cin >> a >> b;
  std::cout << gcd_fast(a, b) << std::endl;
  return 0;
}

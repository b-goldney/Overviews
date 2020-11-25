#include <iostream>

long long lcm_naive(int a, int b) {
  for (long l = 1; l <= (long long) a * b; ++l)
    if (l % a == 0 && l % b == 0)
      return l;

  return (long long) a * b;
}

long long gcd(long a, long b) {
    if (a == 0) {
        return b;
    } else if (b ==0) {
        return a;
    }
    else {
        return gcd(b%a,a);
    }
}
long long lcm_fast(long long a, long long b) {
    return (a*b) / gcd(a,b);
}

int main() {
  int a, b;
  std::cin >> a >> b;
  std::cout << lcm_fast(a, b) << std::endl;
  return 0;
}

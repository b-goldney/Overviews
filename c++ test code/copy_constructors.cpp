#include <iostream>

using std::cout;
using std::endl;

// Purpose: to demonstrate shallow vs deep copies

int main()
{
    int a = 5;
    int b = a;

    cout << &a << " <<< address of a" << endl;
    cout << &b << " <<< address of b" << endl;
    b = 3;

    cout << b << ", " << a << " <<< notice that b is now different from 3" << 
        " because a and b have different addresses." << endl;



return 0;
}


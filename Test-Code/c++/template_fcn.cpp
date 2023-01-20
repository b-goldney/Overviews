// Purpose: to demonstrate how a template class can avoid having to write
// multiple overloaded functions just to handle different argument types

#include <iostream>

using std::cout;
using std::endl;

template<class T>
T max(T a, T b)
{
	return a>b?a:b;
}

int main()
{
	cout << max(3,5) << " <<< max(3,5) called" << endl;
	cout << max(3.0f, 5.0f) << " <<< max(3f, 5f) called" << endl;
}


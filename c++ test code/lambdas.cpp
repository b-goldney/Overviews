#include <iostream>

using std::cout;
using std::endl;

// Purpose: to demonsrate basic functionality of lambda expressions

int main()
{
	// basic lambdas expression
	int a = [](int x, int y)->int{return x+y;}(10,30);
	cout << a << endl;

	// using references
	int b = 10;
	auto f = [&b](){cout << b << endl;};
	f();
	b++;
	f(); // notice that now f prints 11

}

// Purpose: to demonsrate basic functionality of lambda expressions

// Notes from: https://docs.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp?view=msvc-160

#include <iostream>

using std::cout;
using std::endl;


int main()
{
	// basic lambdas expression
	int a = [](int x, int y)->int{return x+y;}(10,30);
    // "An empty capture clause, [ ], indicates that the body of the lambda 
    // expression accesses no variables in the enclosing scope"
    
	cout << a << endl;

	// using references
	int b = 10;
	auto f = [&b](){cout << b << endl;};
	f(); // f prints 10
	b++;
	f(); // notice that now f prints 11

}

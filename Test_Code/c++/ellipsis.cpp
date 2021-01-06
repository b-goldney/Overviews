//Purpose:  demonstrate how we can pass a variable number of arguments to a
//function

// Example adapted from: https://www.cprogramming.com/tutorial/c/lesson17.html

#include <iostream>

using std::cout;
using std::endl;

int sum(int n,...) // "..." is for a variable number of arguments, the n says how many arguments will be given
{
	va_list list; // we need a va_list to hold all the arguments to be summed

    //va_start "Initializes a variable argument list"
	va_start(list, n); // the va_list, named as "list", takes n arguments

	int x;
	int s = 0;
	for(int i =0; i<n;i++) // 
	{
		x = va_arg(list, int);
		s+=x;
	}

    va_end(list);

	return s;
}

int main()
{ 
	cout << sum(3,10,20,30) << endl;
	cout << sum(2,2,3) << endl; //Notice we can put in a different # of arguments
}


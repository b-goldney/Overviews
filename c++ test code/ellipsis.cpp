#include <iostream>

using std::cout;
using std::endl;

int sum(int n,...) // ... is for a variable number of arguments
{
	va_list list; // we need a va_list
	va_start(list, n); // the va_list, named as "list", takes n arguments

	int x;
	int s = 0;
	for(int i =0; i<n;i++) // 
	{
		x = va_arg(list, int);
		s+=x;
	}

	return s;
}

int main()
{ 
	cout << sum(3,10,20,30) << endl;
	cout << sum(1,2,3) << endl; //Notice we can put in a different # of arguments
}


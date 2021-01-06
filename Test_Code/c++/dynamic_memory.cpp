// Purpose: this is a somewhat trivial example, but it's real world
// applications are useful.  In the below example, we simply declare 
// new int[5], meaning the size of the variable is fixed.  However, there 
// are a lot of real world use cases where we won't know the exact size of our
// variable until run time.  Now, we could replace 5 with the desired size of
// the array.

#include <iostream>

using std::cout;
using std::endl;


int main()
{
	int *p =  new int[5];
	
	for(int i = 0; i <5; i++)
	{
		p[i] = i+10;
	}

	for(int i = 0; i<5; i++)
	{
		cout << p[i] << " <<< p[i+10]" << endl;
	}

	delete [] p;

	return 0;
}






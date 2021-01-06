// Purpose: demonstrate how recursion works using a simple integer function
// that calls itself 

#include <iostream>

using std::cout;
using std::endl;


void fun(int n)
{
    cout << "fun(.) called: ";
	if(n>0)
	{
		cout << n << endl;
		fun(n-1);
	};
}

int main()
{
	int x=5;
	fun(x);

	return 0;

}

// Purpose: demonstrate that references simply point to the address of the
// variable, which is comparable to create an alias

#include <iostream>

using std::cout;
using std::endl;

int main()
{
	int x = 10;
	int &y = x;

	cout << &x << " <<< &x" << endl;
	cout << &y << " <<< &y" << endl;
	cout << "Notice both x and y point to the same address. " << endl << 
		"That is because creating a reference is the same as creating an alias" << endl;

	return 0;
}

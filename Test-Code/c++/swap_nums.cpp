// Purpose: to demonstrate how passing by reference lets us access variables outside of the function
// Notice without "&" in the parameters for swap, the values will not change.

#include <iostream>

using std::cout;
using std::endl;


void swap(int &a, int &b)
{
	cout << &a << " " << &b << " <<< address of a and b is the same as x and y because reference is just an alias" << endl;
	int temp;
	temp =a;
	a=b;
	b=temp;
}
int main()
{
	int x = 10, y=20;
	cout << &x << " " << &y << " <<< address of x and y" << endl;

	cout << x << ", " << y << " <<< x, y before the swap" << endl;
	swap(x,y);
	cout << x << ", " << y << " <<< x, y. Notice x and y swapped values" << endl;
	return 0;
}

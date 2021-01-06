//Purpose: to show how a pointer can point to a function and call the function

#include <iostream>

using std::cout;
using std::endl;


void display()
{
	cout << "Hello World" << endl;
};

int max(int x, int y) // Max function)
{
	return x>y? x: y;
};

int min(int x, int y) // Min function
{
	return x<y? x:y;
};

int main()

{
	//Pointer to display() function
	void (*fp)();
	fp = display;
	(*fp)();
	cout << fp << " <<< fp" << endl;

	cout << &fp << " <<< fp" << endl;
	
	cout << *fp << " <<< fp" << endl;

	//declare int pointer
	int (*ptr)(int, int);	
	//Max function
	ptr = max;
	cout << (*ptr)(10,5) << " <<< ptr to max called" << endl;
	//Min function
	ptr  = min;
	cout <<	(*ptr)(10,5) << " <<< ptr to min called" << endl;
	
	return 0;
}



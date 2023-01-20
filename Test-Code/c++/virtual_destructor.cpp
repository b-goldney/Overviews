// Purpose: to demonstrate how pointers to derived classes need to use a virtual 
// destructor to properly deallocate memory. If the Base class' destructor is not 
// declared to be virtual then the derived class destructor will not be called.

#include <iostream>

using std::cout;
using std::endl;


class Base
{
	public:
		virtual ~Base()
		{
			cout << "Destructor of Base" << endl;
		}
};

class Derived: public Base
{
	public:
		~Derived()
		{
			cout << "Destructor of Derived" << endl;
		}
};

void fun()
{
	Base *p = new Derived();
	delete p;
}

int main()
{
	fun();
	return 0;
}

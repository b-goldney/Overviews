//Purpose: to demonstrate the basic functionality of smart_ptr

#include <iostream>
#include <memory>

using std::cout;
using std::endl;
using std::unique_ptr;



class Rectangle
{
private:
	int length_;
	int breadth_;
public:
	Rectangle(int l, int b)
	{
		length_=l;
		breadth_=b;
	}
	int area()
	{
		return length_*breadth_;
	}
};

int main()
{
	unique_ptr<Rectangle> ptr(new Rectangle(10,5)); // this is created in heap because "new"
	cout << ptr->area();	
}

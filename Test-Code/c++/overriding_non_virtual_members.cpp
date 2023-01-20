// Purpose: Demonstrate how overriding base class member names can accidentally
// lead the programmer to believe they've implemented polymorhphism; however,
// the misleading output is simply the result of two methods with the same
// name. 


// Example adapted from Effective C++ by Meyers


#include <iostream>
//#include <conio.h>

using namespace std;

class Base
{
public:
    int b;
    void Display()
    {
        cout<<"Base: Non-virtual display."<<endl;
    };
    virtual void vDisplay()
    {
        cout<<"Base: Virtual display."<<endl;
    };
};

class Derived : public Base
{
public:
    int d;
    void Display()
    {
        cout<<"Derived: Non-virtual display."<<endl;
    };
    virtual void vDisplay()
    {
        cout<<"Derived: Virtual display."<<endl;
    };
};

int main()
{
    Base ba;
    Derived de;

    ba.Display();
    ba.vDisplay();
    de.Display();
    cout << "^^^ we'd expect the line above to be \"Base: Non-virtual display.\" since Display()" 
	    " in the base class is not a virtual function; however, the output is"
	   " \"Derived: Non-virtual display.\" which makes it seem like we are obtaining polymorhphism"
	  " without using the virtual keyword"  << endl;
    de.vDisplay();


    cout << "\n \nEXPLANATION: \n";
   cout << "The method of the same name on the derived class will hide the parent method in this case." 
	   "You would imagine that if this weren't the case, trying to create a method with the same name as a base "
	  "class non-virtual method should throw an error. It is allowed and it's not a problem - and if you call the method "
	  "directly as you have done it will be called fine. But, being non-virtual, C++ method lookup mechanisms that "
	  "allow for polymorphism won't be used. So for example if you created an instance of your derived class but called your "
	  "'Display' method via a pointer to the base class, the base's method will be called, whereas for 'vDisplay' the "
	  "derived method would be called." << endl; 
    
   cout << "\n \nCalling the functions via pointers will demonstrate the advantage of using virtual functions" << endl;
   
   Base *b = &ba;
   b->Display();
   b->vDisplay();
   b = &de;
   b->Display();
   b->vDisplay();
   
   cout << endl << endl;

   return 0;
};

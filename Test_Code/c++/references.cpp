// Purpose: Demonstrate how a reference can change the value of a variable 
// at a specific memory address.

#include <iostream>
using namespace std;

void getSquare(int& number)
{
	number *= number;
	cout << number << " <<< getSquare called" << endl;
}

int main()
{
	cout << "enter the number: ";
	int number = 0;
	cin >> number;

	getSquare(number);
	cout << "square is: " << number <<  endl;
	cout << "Notice if the reference is removed from the function definition then our call to \"number\" is wrong.\n" 
		<< "The number variable is wrong if the reference is removed because the number variable \n"
	        << "within getSquare is deleted after the function is complete.  However, if we are \n"
		<< "working with a reference then the actual value (i.e. value at the memory address) is changed." << endl;
	return 0;

}


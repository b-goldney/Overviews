// Purpose: to show that the address of a pointer, p, is different after using it to iterate
// through an array. 

#include <iostream>

using std::cout;
using std::endl;


int main()
{
	int A[5]{2,4,6,8,10};
	int *p = A;
	cout << p << " <<< addresss of p" << endl;
	for(int i = 0; i <5; i++)
	{
		cout << *p;
        cout << " : " << &*p << " <<< &p \n";
		p++;
	}

	cout << p << " <<< address of p. Notice it's changed." << endl;

	return 0;
}



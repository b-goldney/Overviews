// Purpose: to demonstrate shallow copies when dealing with values vs pointers.
// A shallow copy does not work as intended when dealing with dynamic data.

#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

int main() {
    int a = 4;
    int b = 5;

    vector<int> array = {a,b};
    vector<int> array2 = array;

    cout << "&array[0]: " << &array[0] <<endl;
    cout << "&array2[0]: " << &array2[0] << " <<< different address from above, shallow copy works \n";

    array[0] = 42;

    cout << array2[0] << " array2[0], prints 4 as expected \n \n";
    
    // We can see that copy and setting new data worked as expected; however,
    // this is not the case with dynamic data.
    // Now, let's look at an example using pointers.
    
    //Example adapted from: https://jitpaul.blog/2017/07/12/deepcopy-vs-shallowcopy/
    
    cout << "Duplicate the example above but using pointers: \n";
    int* c = new int;
    int* d = new int;

    *c = 4;
    *d = 5;

    vector<int*> array3 = {c,d};
    vector <int*> array4 = array3;

    *array3[0] = 9;

    cout << "array3[0]: " << array3[0] << endl;
    cout << "array4[0]: " << array4[0] << " <<< Same memory address as above b/c it's a shallow copy \n";

    cout << *array4[0] << " <<< prints 9, even though we changed array3[0] b/c they point to the same address \n";

    // To resolve this issue we will need to create a deeop copy constructor
    return 0;
}


#include <iostream>

using std::cout;
using std::endl;


int main()
{
    const char* staticStr = "Hello World!";

    cout << &staticStr << " <<< &staticStr " << endl;
    cout << *staticStr << " <<< *staticStr " << endl;

return 0;
}


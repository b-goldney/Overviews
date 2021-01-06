// Purpose: to demonstrate some basic functionality with strings

#include <iostream>
#include <string>
#include <iterator> // used for std::size
using std::cout;
using std::endl;
using std::string;
using std::size;

int main()
{
    // Save string as a variable
    const string staticStr = "Hello World!";
    cout << staticStr << " <<< staticStr \n";

    // C-style string
    char cString[] {"test"}; // [] let's the compiler determine the length of the string
    cout << cString << " <<< cString \n";

    const int length{ sizeof(cString) / sizeof(cString[0]) };
    cout << length << " <<< length \n";

    // Notice, length is one larger than the length of the string because C++
    // adds a null terminator to the end of the string
return 0;
}


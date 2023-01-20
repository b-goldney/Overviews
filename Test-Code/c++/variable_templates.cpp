// Purpose: demonstrate how variable templates can sum an arbitrary number 
// of input parameters (e.g. integers or characters)

#include <iostream>

using std::cout;
using std::endl;
using std::string;


template <typename Res, typename ValType>
void Sum(Res& result, ValType& val)
{
    result = result + val;
}

template <typename Res, typename First, typename ... Rest>
void Sum(Res& result, First val1, Rest ... valN)
{
    result = result + val1;
    return Sum(result, valN ...);
}


int main()
{

        double dResult = 0;
        Sum (dResult, 3, 4, 5, 10);
        cout << "dResult = " << dResult << endl;

        string strResult; 
        Sum (strResult, "Hello ", "World ", "these ", "are ", "strings ", "concatenated ");
        cout << "strResult = " << strResult.c_str() << endl;


return 0;
}


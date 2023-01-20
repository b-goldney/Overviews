// Purpose:  program for adding all elements of array

#include <iostream>
using namespace std;

using std::cout;
using std::endl;

void Sum(int A[], int n)
{
    int sum=0; // Temp variable to hold sum

    for(int i=0; i <n; i++)
    {
    sum+=A[i];
    };
    cout << sum << " <<< sum \n";
}

int main()
{
    int A[]={1,2,3,4,5,6},n=6;
	Sum(A, n);
	return 0;
}


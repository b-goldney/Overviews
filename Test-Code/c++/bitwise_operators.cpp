// Purpose: demonstrate how a doubly linked list can be creaetd using only one
// space for memory, as opposed two spaces in a standard doubly linked list.
// Recall, a doubly linked list uses two spaces for memory because it needs to
// hold the address of the next and previous nodes in the list.

// Example adapted from: https://www.geeksforgeeks.org/bitwise-operators-in-c-cpp/#:~:text=The%20%5E%20(bitwise%20XOR)%20in,number%20of%20places%20to%20shift.


#include <iostream>
#include <stdio.h>

using std::cout;
using std::endl;

//Bitwise operators explained:
//The & (bitwise AND) in C or C++ takes two numbers as operands and does AND on every bit of two numbers. The result of AND is 1 only if both bits are 1.
//The | (bitwise OR) in C or C++ takes two numbers as operands and does OR on every bit of two numbers. The result of OR is 1 if any of the two bits is 1.
//The ^ (bitwise XOR) in C or C++ takes two numbers as operands and does XOR on every bit of two numbers. The result of XOR is 1 if the two bits are different.
//The << (left shift) in C or C++ takes two numbers, left shifts the bits of the first operand, the second operand decides the number of places to shift.
// The >> (right shift) in C or C++ takes two numbers, right shifts the bits of the first operand, the second operand decides the number of places to shift.
// The ~ (bitwise NOT) in C or C++ takes one number and inverts all bits of it

// Function to return the only odd occurring element 
int findOdd(int arr[], int n) 
{ 
    int res = 0; 
    for (int i = 0; i < n; i++) 
    {
        cout << "Loop: " << i << endl;
        cout << " res pre assignment: " << res << endl;
        res ^= arr[i];
        cout << " arr[i]: " << arr[i] << endl;
        cout << " res: " << res  << endl;
    }
    return res;
} 


int main() 
{ 
    // a = 5(00000101), b = 9(00001001) 
    unsigned char a = 5, b = 9; 
  
    // The result is 00000001 
    printf("a = %d, b = %d\n", a, b); 
    printf("a&b = %d\n", a & b); 
  
    // The result is 00001101 
    printf("a|b = %d\n", a | b); 
  
    // The result is 00001100 
    printf("a^b = %d\n", a ^ b); 
  
    // The result is 11111010 
    printf("~a = %d\n", a = ~a); 
  
    // The result is 00010010 
    printf("b<<1 = %d\n", b << 1); 
  
    // The result is 00000100 
    printf("b>>1 = %d\n", b >> 1); 


    // Example use case for XOR operator: “Given a set of numbers where all 
    // elements occur even number of times except one number, find the odd occurring number” 
    int arr[] = { 3, 12, 12, 14, 90, 90, 14, 14, 14 }; 
    int n = sizeof(arr) / sizeof(arr[0]);

    cout << "size of n: " << n << endl;
    printf("The odd occurring element is %d ", 
           findOdd(arr, n)); 
    

    return 0; 
} 

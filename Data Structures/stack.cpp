// Purpose: to demonstrate the stack data structure

// Example adapted from: https://www.geeksforgeeks.org/stack-data-structure-introduction-program/

#include <iostream>

using std::cout;
using std::endl;

int const max = 1000; // this will cap the size of our stack

// Create class, Stack, to hold our data
class Stack {
    private:
        int top;
    public:
        int a[max]; // max size of Stack
        Stack() {top = -1;} // constructor, initializing the top variable
        bool push (int x); // modify the stack by adding an element
        int pop(); // this will modify the stack by removing the top value
        int peek(); // this does not modify the stack
        bool isEmpty();
        int size(); // returns size of array
        void print();
};

// push appends values to the stack
bool Stack::push(int x)
{
    if (top >= (max -1)) { // ensure size limit is not exceeded
        cout << "Stack overflow \n"; 
        return false;
    }
    else {
        a[++top] = x; // Assign x to the top value of the stack
        cout << x << " pushed onto stack \n";
        return true;
    }
};

// pop removes the top element from the stack, modifying the stack
int Stack::pop()
{
    if (top < 0) {
        cout << "Stack underflow"; // Can't have element at the -1 position (this is unlike python)
        return 0;
    } 
    else {
        return a[top--]; 
    };
}

int Stack::peek()
{
    cout << " peek is called: ";
    if (top < 0) {
        cout << "Stack is empty \n";
        return 0;
    }
    else {
        cout << a[top] << endl;
        return a[top];
    }
}

bool Stack::isEmpty()
{
    if (top < 0) 
    {
        cout << " stack is empty \n";
        return 0;
    } else {
        cout << " stack is NOT empty \n";
        return 1;
    };
};


void Stack::print() {
    cout << " print called \n";
    for(int i = 0; i <= top; i++) {
        cout << a[i] << ", ";
    }
};

        

int main()
{
    class Stack s;
    s.isEmpty();
    s.push(10);
    s.push(20);
    s.push(30);
    cout << s.pop() << " <<< popped from stack \n";
    s.isEmpty();
    s.peek();
    s.print();

    return 0;
}


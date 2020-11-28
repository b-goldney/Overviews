// Purpose: to demonstrate the stack data structure via linked lists
// The double points can make this hard to follow so the memory addresses
// printed liberally to make it easy to follow how attributes are being
// updated.

// Example adapted from: https://www.geeksforgeeks.org/stack-data-structure-introduction-program/

#include <iostream>
#include <bits/stdc++.h> // this is needef or INT_MIN 

using std::cout;
using std::endl;

int const max = 1000; // this will cap the size of our stack

// Create class, Stack, to hold our data
class StackNode {
    public:
        int data;
        StackNode *next; // this will point to the next element in the stack
};

// Create pointer to point to the next element in the stack
StackNode *newNode(int data)
{
    StackNode *stackNode = new StackNode(); // create on heap
    stackNode->data = data; 
    stackNode->next = NULL;
    return stackNode;
};

// Function to check if stack is empty
int isEmpty(StackNode *root)
{
    //cout << " Called isEmpty(), we can see root always points to the last item. Root is: " << root << endl;
    return !root;
}

// Function to create new element at the end of the list. Root will always be
// the last element in the stack
void push(StackNode **root, int data) // root is a double pointer
{
    cout << " push() called: \n";
    cout << "     &(*root): " << &(*root) << " <<< Same address of the initialized root" << endl;

    StackNode *stackNode = newNode(data); // newNode is pointing to the newly created node
    stackNode->next = *root; // stackNode->next always points to the last created element
    *root = stackNode; // *root is now the last created element (i.e. stackNode)
   
    cout << "     pushed to stack: " << data << endl;
    cout << "     stackNode->next: " << stackNode->next << endl;
    cout << "     stackNode->data: " << stackNode->data << endl;
    cout << "     &(stackNode->data): " << &(stackNode->data) << endl;
    //cout << "     &(stackNode->next): " << &(stackNode->next) << endl;
    cout << "     &(*root): " << &(*root) << " <<< Same address of the initialized root" << endl;
    cout << "     &(**root): " << &(**root) << " <<< Same address as the data \n";
}


// Function to remove last element from the stack
int pop(StackNode **root)
{
    cout << " pop() called: \n";
    if (isEmpty(*root))
    {
        return INT_MIN; // INT_MIN specifies that an integer variable cannot store any value below this limit
    }
    cout << "     &(**root) before popping the element off: " << &(*root) <<
        " <<< same address as initialized root \n";
    StackNode *temp = *root;
    *root = (*root)->next;
    int popped = temp->data;
    free(temp);
    cout << "     &(**root) after popping the element off: " << &(**root) << 
        " <<< same address as the last element in the stack \n";
    return popped;
}

// Function to view, but not remove, the last element in the stack
int peek(StackNode *root)
{
    cout << " peek() called: \n";
    if(isEmpty(root))
    {
        cout << " isEmpty() called inside peek ";
        return INT_MIN;
    }
    cout << root->data;
    return root->data;
}

// Function to print each element in stack
void print(StackNode *n) {
    cout << " print() called: \n";
    while(n != NULL) {
        cout << "     " << n->data << endl;
        n = n->next;
    };
};


int main()
{
    class StackNode *root = NULL;
    cout << &root << " <<< address of root after being initialized \n";
    push(&root, 10);
    push(&root, 20);
    push(&root, 30);

    print(root);
    pop(&root);
    peek(root);

    return 0;
}



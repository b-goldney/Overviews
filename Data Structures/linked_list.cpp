#include <iostream>
#include<stddef.h>

using std::cout;
using std::endl;

// Create Node class which will hold the data and a pointer to the next the
// Node
class Node {
    public:
        int data;
        Node *next;
};

// Create function to print each value in the list until it points to NULL
void print_list(Node *n) {
    while (n != NULL) {
        cout << n->data << " ";
        cout << &n->data << " <<< address of data increases by 1 each iteration\n";
        n = n->next;
    }
}

int main()
{
    // Create pointers to nodes in list
    Node * head = NULL;
    Node * second = NULL;
    Node * third = NULL;

    // Assign each pointer a class of Node on the heap
    head = new Node();
    second = new Node();
    third = new Node();

    // Populate each node with data and a pointer to the next item in the list
    head->data = 1;
    head->next = second;

    second->data=2;
    second->next=third;

    third->data = 3;
    third->next = NULL;

    print_list(head);

return 0;
}


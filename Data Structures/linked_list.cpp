// Purpose: to demonstrate linked lists in c++. We'll create functionality to
// push a node to the front of the list, and to append a node to the end of the
// list.

// Functionality: push, append, print

// Example adapted from: https://www.bitdegree.org/learn/linked-list-c-plus-plus

#include <iostream>

using std::cout;
using std::endl;


// Create Node class which will hold the data and a pointer to the next the
// Node
class Node {
    public:
        int data;
        Node *next;
};

// Create function, print_list, to print each value in the list until it points to NULL
void print_list(Node *n) {
    cout << "print_list() called: \n";
    while (n != NULL) {
        printf("   n->data: %d \n", n->data);
        printf("   &n->data: %s \n", &n->data);
        n = n->next;
    }
    cout << " \n \n ";
}

//  Create function, push, to insert a new node at the front of the list
void push(Node **head_ref, int new_data) {
    printf("   (**head_ref) before pushing new element, matches address of current head:  %p\n", (**head_ref));
    printf(" &(*head_ref): %p  <<< this doesn't change \n", &(*head_ref));
    printf(" *head_ref: %p \n\n" , *head_ref);

    Node *new_node = new Node(); // Create new node
    new_node->data = new_data; // save new_data in new_node
    new_node->next = *head_ref; // point new_node->next to head
    *head_ref = new_node; // point head to new_node

    printf(" &(*head_ref): %p <<< this doesn't change\n", &(*head_ref));
    cout << " *head_ref: " << *head_ref << " <<< *head_ref is now the address of the inserted data"
        << endl << endl;


};

// Create function, append, to append an element to the list
void append(Node **head_ref, int new_data) {
    Node *last = *head_ref; // Create *last, which will be used in the while loop
   
    Node *new_node = new Node(); // Create new node
    new_node->data = new_data;
    new_node->next = NULL; // assign value of NULL since this will be the last node

    cout << "append called, notice how the memory is sequential with respect to the order it was created\n";
    // Check if list is empty, if so, make this Node first
    if (*head_ref == NULL)
    {
        *head_ref = new_node;
        return;
    }

    // If the node is not empty, find the end of the node
    while (last->next != NULL) // when the next node points to NULL then we're at the end of the list
    {
        last = last->next;

        if (last->next == NULL) {
            last->next= new_node;
            return;
        };
    } 
};

int main()
{
    // Create pointers to nodes in list
    Node *head = NULL;
    Node *second = NULL;
    Node *third = NULL;

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

    // push 0 onto the front of the list so the new list is 0->1->2->3
    push(&head, 0);
    
    print_list(head); // print list to confirm it's created accurately
    // Notice, the address of 0 is still 1 ahead of 3, because the 0 was
    // created after 3.  Even though 0 is at the front of the list, its address
    // is after the list elements created before it.
   
    push(&head, 99);

    append(&head, 4);

    print_list(head);


    int *ptr = NULL;
    printf("Value of &ptr: %p\n", ptr);

    return 0;
}


// Purpose: to demonstrate doubly linked lists in c++. We'll create functionality to
// push a node to the front of the list, and to append a node to the end of the
// list. This code is simply an extension of the linked_list.cpp file.

// Functionality: forward print, reverse print, append, push

// Example adapted from: https://www.bitdegree.org/learn/linked-list-c-plus-plus

#include <iostream>

using std::cout;
using std::endl;

// Create Node class which will hold the data and a pointer to the next the
class Node {
    public:
        int data;
        Node *next;
        Node *prev;
};

// Create function, forward_print_list, to print each value in the list until it points to NULL
void forward_print_list(Node *n) {
    cout << " forward_print_list called \n";
    while (n != NULL) {
        cout << n->data << " ";
        cout << &n->data << endl;
        n = n->next;
    }
    cout << " \n \n ";
}

// reverse_print_list will print the list in reverse order
void reverse_print_list(Node *n) {
    cout << " reverse_print_list called \n";
    Node *last = NULL; // this will be used to hold the element's of the last node
    while (n != NULL) {
        last = n;
        n = n->next;
    };
    while (last != NULL) {
        cout << last->data << " <<< " << &last->data << endl;
        last = last->prev;
        }
    cout << " \n \n ";
};

//  Create function, push, to insert a new node at the front of the list
void push(Node **head_ref, int new_data) {
    cout << " push called \n";
    
    Node *new_node = new Node(); // Create new node
    new_node->data = new_data; // save new_data in new_node
    new_node->next = *head_ref; // point new_node->next to head
    new_node->prev = NULL;
    (*head_ref)->prev = new_node; // Assign the "old" head_ref-> prev to the new head_ref
    *head_ref = new_node; // point head_ref to new_node

    };

// Create function, append, to append an element to the list
void append(Node **head_ref, int new_data) {
    Node *last = *head_ref; // Create *last, which will be used in the while loop
   
    Node *new_node = new Node(); // Create new node
    new_node->data = new_data;
    new_node->next = NULL; // assign value of NULL since this will be the last node
    new_node->prev = NULL; // update this in the while loop

    cout << " append called, notice how the memory is sequential with respect to the order it was created\n";
    // Check if list is empty, if so, make this Node first
    if (*head_ref == NULL)
    {
        *head_ref = new_node;
        return;
    }

    // If the node is not empty, find the end of the node
    while (last->next != NULL)
    {
        last = last->next;

        if (last->next == NULL) {
            last->next= new_node;
            new_node->prev = last; // Update node->prev
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
    head->prev = NULL;

    second->data=2;
    second->next=third;
    second->prev=head;

    third->data = 3;
    third->next = NULL;
    third->prev = second;

    forward_print_list(head);

    // push 0 onto the front of the list so the new list is 0->1->2->3
    push(&head, 0);

    forward_print_list(head); // print list to confirm it's created accurately
    // Notice, the address of 0 is still 1 ahead of 3, because the 0 was
    // created after 3.  Even though 0 is at the front of the list, its address
    // is after the list elements created before it.
    
    append(&head, 4);

    forward_print_list(head);
    reverse_print_list(head);


    return 0;
}



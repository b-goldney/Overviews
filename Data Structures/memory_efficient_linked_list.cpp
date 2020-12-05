//Purpose: to demonstrate a memory efficient doubly linked list.
//Recall,  a normal doubly linked list uses two memory spaces to hold 
//the address of the next and previous nodes.
//
// Note, the printf statements have indented text to increase the readability
// on the terminal, after running the program.

// Example adapted from: https://www.geeksforgeeks.org/xor-linked-list-a-memory-efficient-doubly-linked-list-set-2/?ref=lbp

#include <bits/stdc++.h>
#include <cinttypes> 

using namespace std;
 
// Node structure of a memory efficient doubly linked list 
class Node 
{ 
    public:
    int data; 
    Node* npx; // XOR of next and previous node 
}; 
 
// returns XORed value of the node addresses 
Node* XOR (Node *a, Node *b) 
{
    // uintptr_t: Unsigned integer type capable of holding a value converted from a void 
    // pointer and then be converted back to that type with a value that compares 
    // equal to the original pointer.
      cout << "  XOR called:\n";

      Node* temp = reinterpret_cast<Node *>(
      reinterpret_cast<uintptr_t>(a) ^ 
      reinterpret_cast<uintptr_t>(b)); 

      printf("    The value of reinterpret_cast<uintptr_t>(a): %p\n", reinterpret_cast<uintptr_t>(a));
      printf("    The value of reinterpret_cast<uintptr_t>(b): %p\n", reinterpret_cast<uintptr_t>(b));
      printf("    The value of temp: %p \n\n", temp);
      
      return temp;
} 
 
// Insert a node at the beginning of the  XORed linked list and makes the newly 
// inserted node as head void insert(Node **head_ref, int data) 
void insert(Node **head_ref, int data) 
{
    cout << "Insert called: \n";
    cout << " we are inserting: " << data << endl;

    // Allocate memory for new node 
    Node *new_node = new Node(); 
    new_node->data = data; 
    // Since new node is being inserted at the  beginning, npx of new node will always be 
    // XOR of current head and NULL new_node->npx = *head_ref; 
    new_node->npx = *head_ref;

    // Print statements to better understand what's happening
    printf(" &new_node: %p\n", &new_node);
    printf(" &(new_node->data): %p\n", &(new_node->data));
    printf(" &(new_node->npx): %p\n", &(new_node->npx));
    printf(" &(*head_ref): %p  <<< this doesn't change \n", &(*head_ref));
    printf(" *head_ref: %p \n\n" , *head_ref);

    // If linked list is not empty, then npx of  current head node will be XOR of new node 
    // and node next to current head if (*head_ref != NULL) 
    if (*head_ref != NULL)
    { 
        // (head_ref)->npx is XOR of NULL and next. 
        // So if we do XOR of it with NULL, we get next 
        (*head_ref)->npx = XOR(new_node, (*head_ref)->npx); 
    } 
    // Change head 
    *head_ref = new_node;

    printf(" &(new_node->data): %p\n", &(new_node->data));
    printf(" &(*head_ref)->npx): %p <<< this is the address from *head_ref \n", &(*(new_node)->npx));
    printf(" &(*head_ref): %p <<< this doesn't change\n", &(*head_ref));
    cout << " *head_ref: " << *head_ref << " <<< *head_ref is now the address of the inserted data"
        << endl << endl;
} 
 
// prints contents of doubly linked list in forward direction 
void printList (Node *head) 
{ 
    Node *curr = head; 
    Node *prev = NULL; 
    Node *next; 
 
    cout << "printList called: \n"; 
 
    while (curr != NULL) 
    { 
        // print current node 
        cout<<curr->data<<" \n"; 
 
        // get address of next node: curr->npx is next^prev, so curr->npx^prev will be 
        // next^prev^prev which is next 
        next = XOR (prev, curr->npx); 
 
        // update prev and curr for next iteration 
        prev = curr; 
        curr = next; 
    } 
} 
 
// Driver code 
int main () 
{ 
    /* Create following Doubly Linked List 
    head-->40<-->30<-->20<-->10 */
    Node *head = NULL; 
    insert(&head, 10); 
    insert(&head, 20); 
    insert(&head, 30); 
    insert(&head, 40); 
 
    // print the created list 
    printList (head); 
 
    return (0); 
} 

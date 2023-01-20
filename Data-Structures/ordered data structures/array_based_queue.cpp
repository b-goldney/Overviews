// Purpose: demonstrate array-based queue data structure

// Functionality: enqueue, dequeue, isFull, isEmpty, front, rear, 

// Example adapted from: https://www.geeksforgeeks.org/queue-set-1introduction-and-array-implementation/ 

#include <iostream>
#include <bits/stdc++.h> // needed for INT_MIN

using std::cout;
using std::endl;

// A structure to represent a queue 
class Queue { 
public: 
    int front, rear, size; 
    unsigned capacity; 
    int* array; 
}; 
 
// function to create a queue of given capacity. It initializes size of queue as 0 
Queue* createQueue(unsigned capacity) 
{ 
    cout << "createQueue called:\n";
    Queue* queue = new Queue(); // pointer to Queue on heap
    queue->capacity = capacity; // establish max capacity of Queue
    queue->front = queue->size = 0; // set size to 0 at initialization
  
    // This is important, see the enqueue 
    queue->rear = capacity - 1; // adjust for zero-based indexing 
    queue->array = new int[(queue->capacity * sizeof(int))]; // request memory on heap
    
    printf("   queue->capacity:  %d\n", queue->capacity);
    printf("   queue->front:  %d\n", queue->front);
    printf("   queue->rear:  %d\n", queue->rear);
    printf("   queue->array:  %d\n\n", queue->array);
    
    return queue; 
} 

// Queue is full when size becomes equal to the capacity 
int isFull(Queue* queue) 
{ 
    return (queue->size == queue->capacity); 
} 
  
// Queue is empty when size is 0 
int isEmpty(Queue* queue) 
{ 
    return (queue->size == 0); 
} 

// Function to add an item to the queue. It changes rear and size 
// Notice, queue->rear increases as we add elements to the end of the queue
void enqueue(Queue* queue, int item) 
{ 
    if (isFull(queue)) 
        return; 
    
    printf("enqueue called:\n");
    queue->rear = (queue->rear + 1) % queue->capacity; 
    //cout << (queue->rear +1) % queue->capacity << " queue->rear + % queue->capacity\n";
    queue->array[queue->rear] = item; 
    queue->size = queue->size + 1; 
    
    printf("   enqueued to queue: %d\n", item); 
    printf("   queue->front:  %d\n", queue->front);
    printf("   queue->rear:  %d\n\n", queue->rear);

} 

// Function to remove an item from queue. It changes front and size 
// Notice, queue->front increases as the first item in the queue is removed,
// this is opposite of what happens with enqueue 

int dequeue(Queue* queue) 
{ 
    printf("dequeue called:\n");
    
    if (isEmpty(queue)) 
        return INT_MIN; 

    int item = queue->array[queue->front]; 
    queue->front = (queue->front + 1) % queue->capacity; 
    queue->size = queue->size - 1;
    
    printf("   dequeued to queue: %d\n", item); 
    printf("   queue->front:  %d\n", queue->front);
    printf("   queue->rear:  %d\n\n", queue->rear);

    return item; 
} 

// Function to get front of queue 
int front(Queue* queue) 
{ 
    if (isEmpty(queue)) 
        return INT_MIN; 
    return queue->array[queue->front]; 
} 
  
// Function to get rear of queue 
int rear(Queue* queue) 
{ 
    if (isEmpty(queue)) 
        return INT_MIN; 
    return queue->array[queue->rear]; 
} 


int main() 
{ 
    Queue* queue = createQueue(4); 
  
    enqueue(queue, 10); 
    enqueue(queue, 20); 
    enqueue(queue, 30); 
    enqueue(queue, 40); 
  
    dequeue(queue); 
  
    cout << "Front item is " << front(queue) << endl; 
    cout << "Rear item is " << rear(queue) << endl; 
  
    return 0; 
} 

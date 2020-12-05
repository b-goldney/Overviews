#include <iostream>

using std::cout;
using std::endl;


int main() {
   int v = 76;
   int *p1;
   int **p2;
   p1 = &v;
   p2 = &p1;
   printf("Value of v = %d\n", v);
   printf("Address of v = %p\n\n", &v);

   printf("Value of v using single pointer = %d\n", *p1 );
   printf("Value of v using double pointer = %d\n\n", **p2);

   printf("Address of v using single pointer = %p\n", &(*p1));
   printf("Address of v using double pointer = %p\n", &(**p2));
   printf("Access v through double pointer but reference each pointer = %p\n", *(&(*p2)));

   return 0;
}



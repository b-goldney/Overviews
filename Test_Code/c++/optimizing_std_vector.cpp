// Purpose: demonstrate how .push_back is not always a good choice when
// building vectors

// Example adapted from The Cherno's YouTube Series: "Optimizing the usage of std::vector in C++"

#include <iostream>
#include <string>
#include <vector>

using std::cout;
using std::endl;

struct Vertex
{
    float x, y, z;

    Vertex(float x, float y, float z)
        :x(x), y(y), z(z)
    {
    }

    // Copy constructor
    Vertex(const Vertex& vertex)
        : x(vertex.x), y(vertex.y), z(vertex.z)
    {
        cout << "Copy constructor called \n";
    }
};

int main()
{
    // Example 1
    // The below code will call the copy constructor 6 times.
    // The code is called once to copy the object from the main function to the
    // memory allocated by the Vertex object 
    cout << "Example 1: \n";
    std::vector<Vertex> vertices;
    vertices.push_back(Vertex(7,8,9));
    vertices.push_back(Vertex(7,8,9));
    vertices.push_back(Vertex(7,8,9));

    cout << " \n \n \nExample 2: \n";

    // Example 2:
    // By using .reserve() we can save three copies
    std::vector<Vertex> vertices2;
    vertices2.reserve(3);
    vertices2.push_back(Vertex(7,8,9));
    vertices2.push_back(Vertex(7,8,9));
    vertices2.push_back(Vertex(7,8,9));


    cout << " \n \n \nExample 3: \n";
    // Example 3:
    // replace push_back with emplace_back, reducing the number of calls to the
    // copy constructor to 0
    std::vector<Vertex> vertices3;
    vertices3.reserve(3);
    vertices3.emplace_back(7,8,9);
    vertices3.emplace_back(7,8,9);
    vertices3.emplace_back(7,8,9);

    cout << "Example 3 is finished, no calls to the copy constructor \n";

    return 0;
}


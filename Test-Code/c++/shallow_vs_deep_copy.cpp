// Purpose: demonstrate the differences between shallow vs deep copies

// Example adapted from "The Cherno's" YouTube page: "Copying and Copy
// Constructors in C++"


#include <iostream>
#include <cstring>

using std::cout;
using std::endl;

// Used in Example 1
struct Vector2
{
    float x,y;
};

// Used in Example 2
class String 
{
    private:
	char* m_Buffer;
	unsigned int m_Size;
    public:
	String(const char* string)
	{
	    m_Size  = strlen(string);
	    m_Buffer = new char[m_Size + 1]; // +1 to include the null termination character
	    memcpy(m_Buffer, string, m_Size);
	    m_Buffer[m_Size] = 0;
	}

	friend std::ostream& operator<<(std::ostream& stream, const String& string);

	// copy constructor
	String(const String& other)
	    : m_Size(other.m_Size)
	{
	    m_Buffer = new char[m_Size + 1];
	    memcpy(m_Buffer, other.m_Buffer, m_Size + 1);
	}

	~String()
	{
	    delete[] m_Buffer;
	}

	char& operator[](unsigned int index)
	{
	    return m_Buffer[index];
	}
};

std::ostream& operator<<(std::ostream& stream, const String& string)
{
    stream << string.m_Buffer;
    return stream;
}

int main()
{
    cout << "Example 1: \n";
    // Example 1: demonstrate how copying a pointer copies the memory address,
    // resulting in two pointers pointing at the same memory address
    Vector2* a = new Vector2();
    Vector2* b = a;
    b->x = 2;

    cout << a << " <<< a \n";
    cout << b << " <<< b this matches the address above b/c a is a pointer and b copied a\n";
    cout << a->x << " <<< a->x \n";
    cout << b->x << " <<< b->x since a and b both point to the same address then chaning b->x changes a->x\n";


    cout << " \n \n \n Example 2: \n";
    // Example 2: Create a string and then copy that string (i.e. a deep copy
    // not a shallow copy of hte type a = b). In this example, we'll create a
    // string and then deep copy the string
    
    String string = "Tim";
    String second = string;

    second[1] = 'o'; // this won't work without the copy constructor


    cout << string << " <<< string" << endl;
    cout << second << " <<< second: notice we successfully changed the second letter from i to o" << endl;

    std::cin.get();


    return 0;
}

// Before C++20 - you need BOTH operators
struct Point {
    int x, y;
    
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
    
    // You MUST write this too!
    bool operator!=(const Point& other) const {
        return !(*this == other);  // Just the opposite
    }
};

// In C++20 - only need ONE operator
struct Point {
    int x, y;
    
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
    
    // operator!= is automatically generated!
    // No need to write it!
};

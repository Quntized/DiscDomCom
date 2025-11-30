constexpr int square(int x) {
    return x * x;
}

// At compile time:
constexpr int result = square(5);  // Calculated during compilation: result = 25

// At runtime:
int n;
std::cin >> n;
int result2 = square(n);  // Calculated during program execution

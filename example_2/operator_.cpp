
#include <iostream>
#include <cstddef>

// ============================================================================
// Simple Iterator Class (mimics DiscreteDomainIterator)
// ============================================================================

class SimpleIterator {
private:
    size_t m_value;  // Internal position (unsigned!)
    
public:
    using difference_type = std::ptrdiff_t;  // Signed type for differences
    
    // Constructor
    explicit SimpleIterator(size_t value) : m_value(value) {}
    
    // Get the internal value (for demonstration)
    size_t value() const { return m_value; }
    
    // ========================================================================
    // The operator- we're studying
    // ========================================================================
    friend constexpr difference_type operator-(
            SimpleIterator const& xx,
            SimpleIterator const& yy)
    {
        std::cout << "  Computing: xx(" << xx.m_value << ") - yy(" << yy.m_value << ")\n";
        
        // Check if yy is ahead of xx
        if (yy.m_value > xx.m_value) {
            std::cout << "    yy > xx: Need negative result\n";
            std::cout << "    Calculate: yy - xx = " << (yy.m_value - xx.m_value) << "\n";
            difference_type result = -static_cast<difference_type>(yy.m_value - xx.m_value);
            std::cout << "    Negate: -(" << (yy.m_value - xx.m_value) << ") = " << result << "\n";
            return result;
        } else {
            std::cout << "    xx >= yy: Need positive result\n";
            std::cout << "    Calculate: xx - yy = " << (xx.m_value - yy.m_value) << "\n";
            difference_type result = xx.m_value - yy.m_value;
            return result;
        }
    }
    
    // Additional operators for demonstration
    friend bool operator==(const SimpleIterator& a, const SimpleIterator& b) {
        return a.m_value == b.m_value;
    }
    
    friend bool operator!=(const SimpleIterator& a, const SimpleIterator& b) {
        return !(a == b);
    }
};

// ============================================================================
// Demonstration of the Problem with Unsigned Subtraction
// ============================================================================

void demonstrate_unsigned_problem() {
    std::cout << "=== The Problem with Unsigned Subtraction ===\n\n";
    
    unsigned int a = 3;
    unsigned int b = 7;
    
    std::cout << "a = " << a << " (unsigned)\n";
    std::cout << "b = " << b << " (unsigned)\n\n";
    
    std::cout << "Attempt: a - b (expecting -4)\n";
    unsigned int bad_result = a - b;
    std::cout << "Result: " << bad_result << " (WRONG! Underflow!)\n\n";
    
    std::cout << "Correct approach:\n";
    std::cout << "1. Check which is larger\n";
    std::cout << "2. Subtract smaller from larger: b - a = " << (b - a) << "\n";
    std::cout << "3. Apply negative sign: -(" << (b - a) << ") = " << -(int)(b - a) << "\n\n";
}

// ============================================================================
// Main Program
// ============================================================================

int main() {
    std::cout << "=== Iterator Subtraction Operator ===\n\n";
    
    // First, show why we need this logic
    demonstrate_unsigned_problem();
    
    // -------------------------------------------------------------------------
    // Example 1: Iterator ahead
    // -------------------------------------------------------------------------
    std::cout << "--- Example 1: First iterator ahead ---\n";
    SimpleIterator iter1(10);
    SimpleIterator iter2(3);
    
    std::cout << "iter1 points to index: " << iter1.value() << "\n";
    std::cout << "iter2 points to index: " << iter2.value() << "\n\n";
    
    auto diff1 = iter1 - iter2;
    std::cout << "Result: " << diff1 << "\n";
    std::cout << "Meaning: iter1 is " << diff1 << " positions ahead of iter2\n\n";
    
    // -------------------------------------------------------------------------
    // Example 2: Iterator behind
    // -------------------------------------------------------------------------
    std::cout << "--- Example 2: First iterator behind ---\n";
    SimpleIterator iter3(3);
    SimpleIterator iter4(10);
    
    std::cout << "iter3 points to index: " << iter3.value() << "\n";
    std::cout << "iter4 points to index: " << iter4.value() << "\n\n";
    
    auto diff2 = iter3 - iter4;
    std::cout << "Result: " << diff2 << "\n";
    std::cout << "Meaning: iter3 is " << -diff2 << " positions behind iter4\n\n";
    
    // -------------------------------------------------------------------------
    // Example 3: Same position
    // -------------------------------------------------------------------------
    std::cout << "--- Example 3: Same position ---\n";
    SimpleIterator iter5(7);
    SimpleIterator iter6(7);
    
    std::cout << "iter5 points to index: " << iter5.value() << "\n";
    std::cout << "iter6 points to index: " << iter6.value() << "\n\n";
    
    auto diff3 = iter5 - iter6;
    std::cout << "Result: " << diff3 << "\n";
    std::cout << "Meaning: iter5 and iter6 are at the same position\n\n";
    
    // -------------------------------------------------------------------------
    // Example 4: Real-world use case
    // -------------------------------------------------------------------------
    std::cout << "--- Example 4: Measuring loop size ---\n";
    SimpleIterator begin(0);
    SimpleIterator end(100);
    
    std::cout << "Loop from index " << begin.value() 
              << " to index " << end.value() << "\n\n";
    
    auto size = end - begin;
    std::cout << "Number of iterations: " << size << "\n\n";
    
    // -------------------------------------------------------------------------
    // Example 5: The Ternary Operator Breakdown
    // -------------------------------------------------------------------------
    std::cout << "--- Example 5: Understanding the Ternary Operator ---\n\n";
    
    SimpleIterator a(5);
    SimpleIterator b(12);
    
    std::cout << "Computing: a(5) - b(12)\n\n";
    
    std::cout << "The code:\n";
    std::cout << "  return (yy.m_value > xx.m_value)\n";
    std::cout << "         ? (-static_cast<difference_type>(yy.m_value - xx.m_value))\n";
    std::cout << "         : (xx.m_value - yy.m_value);\n\n";
    
    std::cout << "Step-by-step:\n";
    std::cout << "1. Condition: yy(12) > xx(5) ? TRUE\n";
    std::cout << "2. Execute TRUE branch:\n";
    std::cout << "   - Compute: yy - xx = 12 - 5 = 7\n";
    std::cout << "   - Cast to signed: static_cast<ptrdiff_t>(7) = 7\n";
    std::cout << "   - Negate: -(7) = -7\n";
    std::cout << "3. Return: -7\n\n";
    
    auto result = a - b;
    std::cout << "Actual result: " << result << "\n\n";
    
    // -------------------------------------------------------------------------
    // Example 6: Why static_cast is needed
    // -------------------------------------------------------------------------
    std::cout << "--- Example 6: Why static_cast? ---\n\n";
    
    std::cout << "Without static_cast:\n";
    std::cout << "  size_t (unsigned) yy = 12\n";
    std::cout << "  size_t (unsigned) xx = 5\n";
    std::cout << "  Subtraction: yy - xx = 7 (unsigned)\n";
    std::cout << "  Negation: -(unsigned 7) might cause issues\n\n";
    
    std::cout << "With static_cast:\n";
    std::cout << "  size_t (unsigned) yy = 12\n";
    std::cout << "  size_t (unsigned) xx = 5\n";
    std::cout << "  Subtraction: yy - xx = 7 (unsigned)\n";
    std::cout << "  Cast: static_cast<ptrdiff_t>(7) = 7 (signed)\n";
    std::cout << "  Negation: -(signed 7) = -7 (correct!)\n\n";
    
    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    std::cout << "=== Key Concepts ===\n\n";
    
    std::cout << "1. Purpose:\n";
    std::cout << "   Calculate the distance between two iterators\n";
    std::cout << "   Result is signed (can be negative or positive)\n\n";
    
    std::cout << "2. Challenge:\n";
    std::cout << "   Internal m_value is unsigned (size_t)\n";
    std::cout << "   Unsigned subtraction can underflow\n\n";
    
    std::cout << "3. Solution:\n";
    std::cout << "   a) Check which iterator is ahead\n";
    std::cout << "   b) Always subtract smaller from larger\n";
    std::cout << "   c) Manually apply negative sign if needed\n\n";
    
    std::cout << "4. Ternary operator:\n";
    std::cout << "   condition ? value_if_true : value_if_false\n";
    std::cout << "   Compact way to write if-else\n\n";
    
    std::cout << "5. static_cast<difference_type>:\n";
    std::cout << "   Converts unsigned to signed type\n";
    std::cout << "   Ensures safe negation\n\n";
    
    std::cout << "=== Done! ===\n";
    
    return 0;
}


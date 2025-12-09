#include <Kokkos_Core.hpp>
#include <iostream>

// ============================================================================
// DEMONSTRATING THE ERROR AND THE FIX
// ============================================================================

void wrong_approach() {
    std::cout << "=== WRONG APPROACH (This will not compile) ===" << std::endl;
    
    // THIS IS WRONG - Cannot allocate const View directly:
    // Kokkos::View<const double**> const_view("const_view", 10, 20);
    // Error: cannot convert 'const double*' to 'void*' in ZeroMemset
    
    std::cout << "Attempting to allocate const View directly causes compile error" << std::endl;
}

void correct_approach_1() {
    std::cout << "\n=== CORRECT APPROACH 1: Allocate non-const, use as-is ===" << std::endl;
    
    // Allocate a regular (non-const) View
    Kokkos::View<double**> view("my_view", 10, 20);
    
    // Initialize it
    Kokkos::parallel_for("init", 10, KOKKOS_LAMBDA(const int i) {
        for(int j = 0; j < 20; ++j) {
            view(i, j) = i * 20 + j;
        }
    });
    
    std::cout << "Successfully allocated and initialized non-const View" << std::endl;
    std::cout << "View dimensions: " << view.extent(0) << " x " << view.extent(1) << std::endl;
}

void correct_approach_2() {
    std::cout << "\n=== CORRECT APPROACH 2: Allocate non-const, then make const ===" << std::endl;
    
    // Step 1: Allocate a non-const View
    Kokkos::View<double**> writable_view("writable", 10, 20);
    
    // Step 2: Initialize it
    Kokkos::parallel_for("init", 10, KOKKOS_LAMBDA(const int i) {
        for(int j = 0; j < 20; ++j) {
            writable_view(i, j) = i * 100 + j;
        }
    });
    
    // Step 3: Create a const View that references the same data
    Kokkos::View<const double**> read_only_view = writable_view;
    
    std::cout << "Successfully created const View from non-const View" << std::endl;
    std::cout << "Const view dimensions: " << read_only_view.extent(0) 
              << " x " << read_only_view.extent(1) << std::endl;
    std::cout << "Both views share the same data: " 
              << (writable_view.data() == read_only_view.data()) << std::endl;
    
    // You can still modify through writable_view:
    Kokkos::parallel_for("modify", 1, KOKKOS_LAMBDA(const int) {
        writable_view(0, 0) = 999.0;
    });
    
    // But you CANNOT modify through read_only_view:
    // read_only_view(0, 0) = 123.0;  // Would NOT compile!
    
    std::cout << "Modification through writable_view: OK" << std::endl;
    std::cout << "Modification through read_only_view: NOT ALLOWED (compile error)" << std::endl;
}

void correct_approach_3_unmanaged() {
    std::cout << "\n=== CORRECT APPROACH 3: Using Unmanaged Views ===" << std::endl;
    
    // Allocate memory yourself
    const int N = 10, M = 20;
    double* raw_data = new double[N * M];
    
    // Initialize the data
    for(int i = 0; i < N * M; ++i) {
        raw_data[i] = static_cast<double>(i);
    }
    
    // Create an unmanaged non-const View
    Kokkos::View<double**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
        unmanaged_view(raw_data, N, M);
    
    // Create a const unmanaged View (this works because we're not allocating)
    Kokkos::View<const double**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
        const_unmanaged_view(raw_data, N, M);
    
    std::cout << "Successfully created unmanaged Views (both const and non-const)" << std::endl;
    std::cout << "Unmanaged views don't allocate memory, so const is allowed" << std::endl;
    
    // Clean up
    delete[] raw_data;
}

void function_taking_const_view(Kokkos::View<const double**> const_view) {
    std::cout << "\n=== Function receiving const View ===" << std::endl;
    std::cout << "Function can read from const View" << std::endl;
    std::cout << "View dimensions: " << const_view.extent(0) 
              << " x " << const_view.extent(1) << std::endl;
    
    // Can read the data
    auto host_mirror = Kokkos::create_mirror_view(const_view);
    Kokkos::deep_copy(host_mirror, const_view);
    std::cout << "First element: " << host_mirror(0, 0) << std::endl;
    
    // But CANNOT modify:
    // const_view(0, 0) = 123.0;  // Would NOT compile!
}

void demonstrate_use_case() {
    std::cout << "\n=== PRACTICAL USE CASE: Protecting Data ===" << std::endl;
    
    // Create and initialize a View
    Kokkos::View<double**> data("data", 5, 10);
    Kokkos::parallel_for("init", 5, KOKKOS_LAMBDA(const int i) {
        for(int j = 0; j < 10; ++j) {
            data(i, j) = i * 10.0 + j;
        }
    });
    
    // Pass to a function as const to prevent modification
    Kokkos::View<const double**> const_data = data;
    function_taking_const_view(const_data);
    
    std::cout << "Original data is protected from modification in the function" << std::endl;
}

// ============================================================================
// YOUR FIX FOR LINE 175
// ============================================================================

void your_code_fix() {
    std::cout << "\n=== FIX FOR YOUR CODE (line 175) ===" << std::endl;
    
    // BEFORE (line 175 - WRONG):
    // Kokkos::View<const double**> view("your_label", n, m);
    
    // AFTER (CORRECT):
    const size_t n = 10, m = 20;
    
    // Option 1: Just remove const if you don't need it
    Kokkos::View<double**> view1("your_label", n, m);
    std::cout << "Option 1: Removed const - View<double**>" << std::endl;
    
    // Option 2: If you really need const, do this:
    Kokkos::View<double**> temp_view("temp", n, m);
    Kokkos::View<const double**> view2 = temp_view;
    std::cout << "Option 2: Allocate non-const, then assign to const" << std::endl;
    
    // Option 3: If wrapping existing data:
    double* existing_data = new double[n * m];
    Kokkos::View<const double**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
        view3(existing_data, n, m);
    std::cout << "Option 3: Unmanaged const View wrapping existing data" << std::endl;
    delete[] existing_data;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        std::cout << "==========================================" << std::endl;
        std::cout << "KOKKOS CONST VIEW ERROR - FIX DEMONSTRATION" << std::endl;
        std::cout << "==========================================" << std::endl;
        
        wrong_approach();
        correct_approach_1();
        correct_approach_2();
        correct_approach_3_unmanaged();
        demonstrate_use_case();
        your_code_fix();
        
        std::cout << "\n==========================================" << std::endl;
        std::cout << "KEY TAKEAWAY:" << std::endl;
        std::cout << "- NEVER allocate: View<const T**>" << std::endl;
        std::cout << "- ALWAYS allocate: View<T**>" << std::endl;
        std::cout << "- THEN assign to: View<const T**> if needed" << std::endl;
        std::cout << "==========================================" << std::endl;
    }
    Kokkos::finalize();
    
    return 0;
}

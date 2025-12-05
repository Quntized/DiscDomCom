#include <Kokkos_Core.hpp>
#include <iostream>
#include <type_traits>

// Emulating the function from your snippet
namespace Impl {
    template <class HandleType>
    constexpr auto ptr_from_data_handle(const HandleType& handle) {
        static_assert(std::is_pointer_v<HandleType>, "Error: Not a raw pointer!");
        return handle;
    }
}

int main() {
    // Case 1: Raw Pointer (Valid)
    double value = 5.0;
    double* raw_ptr = &value;

    // This works perfectly. The function confirms it's a pointer and returns it.
    auto result = Impl::ptr_from_data_handle(raw_ptr);
    
    std::cout << "Original: " << raw_ptr << ", Result: " << result << std::endl;

    // Case 2: Integer Pointer (Valid)
    int* int_ptr = nullptr;
    auto result2 = Impl::ptr_from_data_handle(int_ptr); // Works fine
    std::cout << "Original_2nd one: " << int_ptr << ", Result_2nd one: " << result2 << std::endl;

    // Case 3: NOT a Pointer (Compilation Error)
    // int not_a_pointer = 5;
    // Impl::ptr_from_data_handle(not_a_pointer); 
    // ^-- This would trigger the static_assert: "Error: Not a raw pointer!"

    return 0;
}

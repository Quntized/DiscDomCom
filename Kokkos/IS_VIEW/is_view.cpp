#include <Kokkos_Core.hpp>
#include <type_traits>
#include <iostream>
#include <vector>

// A generic function that only accepts Views
template <class T>
void process_data(T data) {
    // This check happens at compile-time!
    if constexpr (Kokkos::is_view_v<T>) {
        std::cout << "Processing a Kokkos View!\n";
        std::cout<<"It is = "<<typeid(T).name()<<"\n";
    } else {
        std::cout << "Error: This is not a View. It is a " << typeid(T).name() << "\n";
    }
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    Kokkos::View<double*> my_view("A", 100);
    std::vector<double> my_std_vec(100);

    process_data(my_view);      // Prints: Processing a Kokkos View!
    process_data(my_std_vec);   // Prints: Error: This is not a View...

    return 0;
}

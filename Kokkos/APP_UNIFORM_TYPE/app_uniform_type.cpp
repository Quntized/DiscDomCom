#include <Kokkos_Core.hpp>
#include <iostream>
#include <typeinfo>

// ----------------------------------------------------------------------
// 1. THE HEAVY WORKER (Non-Template or Restricted Template)
// ----------------------------------------------------------------------
// Imagine this function contains 5,000 lines of complex physics code.
// We DO NOT want to compile this 50 times for 50 different array sizes.
// We want to compile it ONCE.
//
// We define it to take a dynamic 'double*' View.
void heavy_physics_solver(Kokkos::View<double*> data) {
    std::cout << "  [Worker] Running heavy physics on " << data.extent(0) 
              << " elements. (Type: " << typeid(data).name() << ")\n";
    
    // Do work...
    Kokkos::parallel_for("Solver", data.extent(0), KOKKOS_LAMBDA(int i) {
        data(i) *= 2.0;
    });
}

// ----------------------------------------------------------------------
// 2. THE PUBLIC API (The Dispatcher)
// ----------------------------------------------------------------------
// This lightweight template accepts ANY View type from the user.
// It converts the input to the 'uniform_runtime_type' and calls the worker.
template <class UserViewType>
void run_simulation(UserViewType user_view) {
    std::cout << "[API] Received user view.\n";

    // STEP A: Get the uniform type alias
    // This calculates "Kokkos::View<double*>" regardless of what user_view is.
    using UniformView = typename UserViewType::uniform_runtime_type;

    // STEP B: Convert the view
    // Kokkos allows assigning static views (double[10]) to dynamic views (double*).
    UniformView uniform_view = user_view;

    // STEP C: Call the single worker instance
    heavy_physics_solver(uniform_view);
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Uniform Type Usage ---\n\n";

    // Case 1: Static View (Size 10)
    Kokkos::View<double[10]> static_10("static_10");
    static_10(0) = 1.0;
    
    // Case 2: Static View (Size 50) - DIFFERENT C++ TYPE
    Kokkos::View<double[50]> static_50("static_50");
    static_50(0) = 1.0;

    // Case 3: Dynamic View (Size 100) - YET ANOTHER TYPE
    Kokkos::View<double*> dynamic_100("dynamic_100", 100);
    dynamic_100(0) = 1.0;

    // 3. EXECUTION
    // Even though we pass 3 different C++ types, they all call 
    // the EXACT SAME 'heavy_physics_solver' function binary.
    
    run_simulation(static_10);
    run_simulation(static_50);
    run_simulation(dynamic_100);

    return 0;
}

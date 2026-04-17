#include <iostream>
#include <array>
#include <cstddef>

// 1. Define the missing type alias
using DiscreteVectorElement = std::size_t;

// 2. Create a generic N-dimensional vector struct
// This acts as the 'discrete_vector_type' expected by your function
template<std::size_t N>
struct Vec {
    std::array<DiscreteVectorElement, N> coords;

    // Variadic constructor so `discrete_vector_type(is...)` works
    template<typename... Args>
    Vec(Args... args) : coords{static_cast<DiscreteVectorElement>(args)...} {}
};

// 3. Create a dummy "Support" domain class
template<std::size_t Dimensions>
struct MyDomain {
    using discrete_vector_type = Vec<Dimensions>;

    // Your function calls domain(...), so we need an operator()
    // We'll just have it pass the vector straight through to the functor.
    discrete_vector_type operator()(discrete_vector_type v) const {
        return v; 
    }
};


template <class Support, std::size_t N, class Functor, class... Is>
void host_for_each_serial(
        Support const& domain,
        std::array<DiscreteVectorElement, N> const& size,
        Functor const& f,
        Is const&... is) noexcept
{
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        f(domain(typename Support::discrete_vector_type(is...)));
    } else {
        for (DiscreteVectorElement ii = 0; ii < size[I]; ++ii) {
            host_for_each_serial(domain, size, f, is...);
        }
    }
}
// ---------------------------

int main() {
    // Let's test this with a 2-Dimensional space
    constexpr std::size_t N = 2; 

    MyDomain<N> domain;
    
    // Set the bounds of our grid: 2 on the x-axis, 3 on the y-axis
    std::array<DiscreteVectorElement, N> grid_size = {2, 3}; 

    // 4. Create a Functor (a lambda) to print the points
    auto print_functor = [](const Vec<N>& point) {
        std::cout << "Visiting point: (" << point.coords[0] << ", " << point.coords[1] << ")\n";
    };

    std::cout << "Iterating over a 2x3 space:\n";
    
    // Kick off the recursive template
    host_for_each_serial(domain, grid_size, print_functor);

    return 0;
}

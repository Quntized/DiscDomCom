#include <Kokkos_Core.hpp>
#include <iostream>
#include <type_traits>

int main(int argc, char* argv[]) {
    // 1. Initialize Kokkos
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos View API Demo ---" << std::endl;

    // =========================================================
    // PART A: The "Lazy Constructor" (Inherited from base_t)
    // =========================================================
    // We define a 1D View of doubles.
    // We do NOT allocate it manually. We just give it a name and size.
    // The "Lazy Constructor" inside Kokkos::View handles the allocation logic.
    using DeviceViewType = Kokkos::View<double*>;
    DeviceViewType device_data("my_gpu_array", 10);

    // =========================================================
    // PART B: The "Public ID Card" (Introspection)
    // =========================================================
    // We use the typedefs exported by the View class to learn about it.
    
    // 1. Ask: "Where do you live?"
// 1. Ask: "Where do you live?"
    using MySpace = DeviceViewType::memory_space;

    // --- FIX START ---
    #ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same_v<MySpace, Kokkos::CudaSpace>) {
        std::cout << "Data is on: NVIDIA GPU" << std::endl;
    } else 
    #endif
    // --- FIX END ---
    if (std::is_same_v<MySpace, Kokkos::HostSpace>) {
        std::cout << "Data is on: CPU RAM" << std::endl;
    } else {
        std::cout << "Data is on: Default Device" << std::endl;
    }
    // 2. Ask: "What is your CPU backup type?"
    // We use 'host_mirror_space' which is exported by the View traits.
    using HostMirrorType = DeviceViewType::HostMirror; 
    
    // Create the mirror (CPU version) using that type
    HostMirrorType host_data = Kokkos::create_mirror_view(device_data);

    // =========================================================
    // PART C: Using the Views (Accessors)
    // =========================================================
    
    // 1. Write on CPU (Host)
    for (int i = 0; i < 10; ++i) {
        host_data(i) = i * 1.5; // Simple math
    }

    // 2. Copy CPU -> GPU (Deep Copy)
    Kokkos::deep_copy(device_data, host_data);

    // 3. Compute on GPU (Device)
    // We use the 'device_data' view inside the kernel.
    Kokkos::parallel_for("SquareKernel", 10, KOKKOS_LAMBDA(const int i) {
        // We modify the data directly in device memory
        device_data(i) = device_data(i) * device_data(i);
    });

    // 4. Copy GPU -> CPU (Deep Copy)
    Kokkos::deep_copy(host_data, device_data);

    // =========================================================
    // PART D: Verification
    // =========================================================
    std::cout << "\nResults:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Index " << i << ": " << host_data(i) << std::endl;
    }

    return 0;
}

#include <Kokkos_Core.hpp>
#include <iostream>
#include <cstdio>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Mirror View Demo ---\n";

    // 1. DEFINE DEVICE VIEW (The "GPU" Array)
    // We use DefaultExecutionSpace. If you had CUDA enabled, this would be on the GPU.
    // On your laptop, this is just in RAM, but the logic remains identical.
    using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
    using ViewType = Kokkos::View<double*, DeviceSpace>;
    
    int N = 10;
    ViewType device_view("device_data", N);

    // 2. CREATE HOST MIRROR (The "CPU" Backup)
    // This creates a View that lives in HostSpace (CPU RAM) but has the
    // same shape and layout as the device_view.
    // KEY FUNCTION: create_mirror_view
    ViewType::HostMirror host_view = Kokkos::create_mirror_view(device_view);

    // 3. FILL DATA ON CPU (Host)
    // We can write to host_view normally because it is in CPU RAM.
    for (int i = 0; i < N; ++i) {
        host_view(i) = (double)i;
    }

    std::cout << "1. Filled data on CPU.\n";

    // 4. COPY CPU -> GPU (Deep Copy)
    // We push the data from the Host Mirror to the Device View.
    Kokkos::deep_copy(device_view, host_view);
    
    std::cout << "2. Copied data to Device (GPU).\n";

    // 5. COMPUTE ON DEVICE (GPU)
    // We assume device_view is on the GPU, so we launch a kernel to modify it.
    Kokkos::parallel_for("SquareKernel", N, KOKKOS_LAMBDA(const int i) {
        // Square the value in place
        device_view(i) = device_view(i) * device_view(i);
        
        // Note: We cannot print easily from here if on GPU, so we wait.
    });

    // Wait for GPU to finish
    Kokkos::fence();
    std::cout << "3. Modified data on Device (Squared it).\n";

    // 6. COPY GPU -> CPU (Deep Copy)
    // Pull the results back to the Host Mirror so we can print them.
    Kokkos::deep_copy(host_view, device_view);

    std::cout << "4. Copied results back to CPU.\n";

    // 7. VERIFY RESULTS
    std::cout << "Results:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "  Index " << i << ": " << host_view(i) << "\n";
    }

    return 0;
}

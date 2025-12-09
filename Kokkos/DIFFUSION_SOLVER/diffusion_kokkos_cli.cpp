#include <iostream>
#include "diffusion_kokkos_py.cpp"

int main(int argc, char* argv[]) {
  // Delegate to cli_main implemented in the module source
  return cli_main(argc, argv);
}

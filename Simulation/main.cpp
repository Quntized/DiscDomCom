/**
 * @file main.cpp
 * @brief Main Simulation Loop for Fusion Transport
 */

#include <iostream>
#include <ddc/ddc.hpp>
#include "plasma_model.hpp"
#include "io_hdf5.hpp" // Hypothetical IO helper

int main(int argc, char** argv) {

    ddc::ScopeGuard ddc_scope(argc, argv);
    Kokkos::ScopeGuard kokkos_scope(argc, argv);


    CoordR r_min(0.0), r_max(2.0);
    CoordVpar v_min(-6.0), v_max(6.0);



    IDimR mesh_r = ddc::init_discretization<IDimR>(r_min, r_max, 128);
    IDimTheta mesh_theta = ddc::init_discretization<IDimTheta>(0.0, 2*M_PI, 64);



    Domain5D global_domain(mesh_r, mesh_theta, /*...*/);


    DistributionFunction<Kokkos::CudaSpace> f_distribution(global_domain);
    

    ddc::parallel_for_each(
        Kokkos::DefaultExecutionSpace(),
        global_domain,
        KOKKOS_LAMBDA(Index5D idx) {
            f_distribution(idx) = init_maxwellian(idx);
        }
    );


    double dt = 0.01;
    for(int step = 0; step < 1000; ++step) {

        advect_spatial(f_distribution, dt/2);
        advect_velocity(f_distribution, dt); // Acceleration
        advect_spatial(f_distribution, dt/2);
        

        apply_collisions(f_distribution, dt);


        if(step % 100 == 0) {
            std::cout << "Step " << step << ": Writing HDF5 output..." << std::endl;
            save_hdf5(f_distribution, "plasma_state_" + std::to_string(step) + ".h5");
        }
    }

    return 0;
}

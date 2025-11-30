/**
 * @file transport_solver.cpp
 * @brief Implementation of the Vlasov Operator using Semi-Lagrangian method
 */

#include "plasma_model.hpp"
#include <Kokkos_Core.hpp>


void advect_vpar(
    DistributionFunction<Kokkos::CudaSpace>& f_dist, 
    double dt, 
    const ElectricField& E_field
) {

    auto domain = f_dist.domain();


    ddc::parallel_for_each(
        Kokkos::DefaultExecutionSpace(),
        domain,
        KOKKOS_LAMBDA(Index5D idx) {

            double v = ddc::coordinate(idx.uid<IDimVpar>());
            double r = ddc::coordinate(idx.uid<IDimR>());
            

            double acceleration = (q_charge / m_mass) * E_field(r);
            double v_star = v - acceleration * dt;
            



        }
    );
}


ddc::Chunk<double, Domain3D> compute_density(
    const DistributionFunction<Kokkos::CudaSpace>& f_dist
) {


    return ddc::integrate<IDimVpar, IDimMu>(f_dist);
}


/**
 * @file plasma_model.hpp
 * @brief Definitions of the 5D Phase Space for Tokamak Transport
 * Uses DDC (Distributed Domain Computation) library.
 */

#pragma once
#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>


struct RDim { static constexpr const char* name = "R"; };
struct ThetaDim { static constexpr const char* name = "Theta"; };
struct PhiDim { static constexpr const char* name = "Phi"; };
struct VparDim { static constexpr const char* name = "Vpar"; };
struct MuDim { static constexpr const char* name = "Mu"; };


using CoordR = ddc::Coordinate<RDim>;
using CoordTheta = ddc::Coordinate<ThetaDim>;



struct BSplinesR : ddc::NonUniformBSplines<RDim, 3> {}; 
struct IDimR : ddc::KnotsAsInterpolationPoints<BSplinesR> {};


struct IDimTheta : ddc::UniformPointSampling<ThetaDim> {};


using Index5D = ddc::DiscreteElement<IDimR, IDimTheta, IDimPhi, IDimVpar, IDimMu>;
using Domain5D = ddc::DiscreteDomain<IDimR, IDimTheta, IDimPhi, IDimVpar, IDimMu>;



template <typename MemorySpace>
using DistributionFunction = ddc::Chunk<double, Domain5D, ddc::KokkosAllocator<double, MemorySpace>>;


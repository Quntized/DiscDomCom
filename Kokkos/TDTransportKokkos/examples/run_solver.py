import tdtransport
import numpy as np

nx,ny,nz = 20,20,1
G = 2
Na = 16
solver = tdtransport.TransportSolver(nx,ny,nz,G,Na)

ncells = nx*ny*nz
# create simple homogeneous cross-sections
sigma_t = np.ones(G*ncells).tolist()
qext = (0.1 * np.ones(G*ncells)).tolist()

solver.set_total_xs(sigma_t)
solver.set_external_source(qext)

solver.run_time_steps(5, 0.01, inner_iters=10)

phi0 = solver.get_scalar_flux(0)
print('phi0[0:8]=', phi0[:8])

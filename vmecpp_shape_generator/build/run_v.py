# validate_my_shape.py
import vmecpp
import numpy as np
import matplotlib.pyplot as plt

# 1. Run your Physics Solver
output = vmecpp.run(input=vmecpp.VmecInput.from_file("input.test_shape"))

# 2. Extract the boundary it solved for
# (Reconstruct R, Z from Fourier coefficients)
ns = output.wout.ns
theta = np.linspace(0, 2*np.pi, 100)
R_boundary = np.zeros_like(theta)
Z_boundary = np.zeros_like(theta)

# Sum Fourier series (m=0 to mmax) for the Last Closed Flux Surface (-1)
for m in range(output.wout.mpol):
    R_boundary += output.wout.rmnc[-1, m] * np.cos(m * theta)
    Z_boundary += output.wout.zmns[-1, m] * np.sin(m * theta)

# 3. Plot it
plt.plot(R_boundary, Z_boundary, label="VMEC++ Solution")
plt.axis('equal')
plt.title("Did the Solver respect my Kokkos Shape?")
plt.show()

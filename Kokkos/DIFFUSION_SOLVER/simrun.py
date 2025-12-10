import numpy as np
import matplotlib.pyplot as plt
import diffusion_kokkos
import time

# --- Parameters ---
D       = 0.01   # Diffusivity
L       = 1.0    # Domain length
Nx      = 101    # Grid points
t_final = 1.0    # Run for 1.0 second

# --- Stability Calculation ---
dx = L / (Nx - 1)
# We choose dt dynamically to be safe (alpha = 0.4)
dt_stable = 0.4 * (dx * dx) / D 

print(f"--- Setup ---")
print(f"dx: {dx:.6f}")
print(f"Calculated stable dt: {dt_stable:.6f}")

# --- Run Simulation ---
print("Running Kokkos simulation...")
start_time = time.time()

# We pass our calculated safe dt
result = diffusion_kokkos.simulate(
    diffusivity=D, 
    length=L, 
    Nx=Nx, 
    dt=dt_stable, 
    t_final=t_final, 
    bc="dirichlet"
)

end_time = time.time()
print(f"Simulation finished in {end_time - start_time:.4f} seconds.")

# --- Plotting (Save to file) ---
x = np.linspace(0, L, Nx)

plt.figure(figsize=(10, 6))
plt.plot(x, result, label=f't={t_final}s')
plt.title(f"1D Diffusion (Kokkos + Python)\nN={Nx}, dt={dt_stable:.2e}")
plt.xlabel("Position")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# FIX: Save the file instead of trying to show a window
filename = "diffusion_result.png"
plt.savefig(filename)
print(f"Plot saved to '{filename}'")

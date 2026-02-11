import vmecpp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# TARGET FILE
FILENAME = "/home/sajid/DDC/Kokkos_Vmecpp_combination/build/input.w7x_robust"

def plot_robust(filename):
    path = Path(filename)
    if not path.exists():
        print(f"ERROR: {filename} does not exist. Did you run the C++ code?")
        return

    # 1. Run VMEC
    print(f"--- Running VMEC++ on {filename} ---")
    indata = vmecpp.VmecInput.from_file(path)
    output = vmecpp.run(indata, verbose=False) # verbose=False keeps terminal clean
    
    wout = output.wout
    if wout.ier_flag != 0:
        print("WARNING: Simulation did not fully converge. Plotting anyway...")
    else:
        print("SUCCESS: Simulation Converged.")

    # 2. Extract Data Safely
    # VMEC++ output arrays are flattened. We use XM and XN maps to decode them.
    xm = np.array(wout.xm)
    xn = np.array(wout.xn)
    
    # Get boundary coefficients (Index -1 is the Last Closed Flux Surface)
    rmnc = np.array(wout.rmnc)[-1, :]
    zmns = np.array(wout.zmns)[-1, :]

    print(f"Plotting Surface using {len(xm)} Fourier Modes...")

    # 3. Reconstruct Surface
    theta = np.linspace(0, 2*np.pi, 500)
    phi = 0.0  # Cross-section at toroidal angle 0
    
    R = np.zeros_like(theta)
    Z = np.zeros_like(theta)

    for i in range(len(xm)):
        m = xm[i]
        n = xn[i]
        
        # Fundamental Harmonic Equation
        # angle = (m * theta) - (n * phi)
        angle = m * theta - n * phi
        
        R += rmnc[i] * np.cos(angle)
        Z += zmns[i] * np.sin(angle)

    # 4. Plot
    plt.figure(figsize=(8, 8))
    plt.plot(R, Z, linewidth=2.5, color='blue', label='Plasma Boundary')
    
    # Add context elements
    plt.axvline(x=5.5, color='red', linestyle='--', alpha=0.5, label='Machine Center (5.5m)')
    plt.scatter([5.5], [0.0], color='red', marker='+')
    
    # Force physical aspect ratio
    plt.axis('equal')
    plt.xlabel("Major Radius R [m]", fontsize=12)
    plt.ylabel("Height Z [m]", fontsize=12)
    plt.title(f"Reconstructed Plasma Shape\nConfig: {filename}", fontsize=14)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    save_name = "robust_plot.png"
    plt.savefig(save_name, dpi=150)
    print(f"Plot saved to: {save_name}")

if __name__ == "__main__":
    plot_robust(FILENAME)

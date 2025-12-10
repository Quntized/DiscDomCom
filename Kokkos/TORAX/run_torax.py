import torax_kokkos as tk
import matplotlib.pyplot as plt
import time
import os

def main():
    # 1. Initialize the C++ Backend
    tk.init()
    print("Kokkos Backend Initialized!")

    try:
        # 2. Setup Simulation
        n_points = 100
        dt = 1e-4
        radius = 2.0
        
        # Instantiate the C++ Solver
        solver = tk.TransportSolver(n_points, dt, radius)

        # 3. Set Initial Physics State
        # (Positional arguments for compatibility with your current build)
        solver.set_initial_condition(10.0, 0.1)
        solver.set_transport_model(1.5)

        # 4. The Time Loop
        print("Starting Simulation...")
        start_time = time.time()
        
        for step in range(10000):
            solver.step()
            
            if step % 2000 == 0:
                print(f"Step {step} complete")

        end_time = time.time()
        print(f"Simulation finished in {end_time - start_time:.4f} seconds")

        # 5. Get Results
        temp_profile = solver.get_temperature()

        # 6. Plotting (Save to file instead of showing)
        plt.figure()
        plt.plot(temp_profile, linewidth=2, color='blue')
        plt.title("Core Temperature Profile (Solved by Kokkos)")
        plt.xlabel("Radial Grid Index")
        plt.ylabel("T_e [keV]")
        plt.grid(True)
        
        output_file = "simulation_result.png"
        plt.savefig(output_file)
        print(f"Plot saved to: {os.path.abspath(output_file)}")
        
        # --- CRITICAL FIX: DESTROY SOLVER BEFORE FINALIZE ---
        del solver 
        # Forces the C++ destructor to run while Kokkos is still alive
        
    finally:
        # 7. Cleanup
        tk.finalize()
        print("Kokkos Finalized Cleanly.")

if __name__ == "__main__":
    main()

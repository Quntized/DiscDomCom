import vmecpp
from pathlib import Path

# 1. Point to your generated Input File
# Replace 'input.w7x' with the file your C++ code generated (e.g., 'input.test_shape')
input_file = Path("input.test_shape")

# 2. Load the Input Configuration
# This reads the text file and converts it into a C++ object
vmec_input = vmecpp.VmecInput.from_file(input_file)

# 3. Run the Solver
# This calls the C++ backend to solve the equilibrium
print(f"Running VMEC++ on {input_file}...")
vmec_output = vmecpp.run(input=vmec_input)

# 4. Access Results (No need to parse 'wout' files!)
# You can access physics data directly from the returned object
nfp = vmec_output.wout.nfp
iota_edge = vmec_output.wout.iotas[-1]  # Rotational transform at the edge
energy = vmec_output.wout.wb            # Stored magnetic energy

print("\n--- Simulation Complete ---")
print(f"Field Periods (NFP): {nfp}")
print(f"Edge Iota:           {iota_edge:.4f}")
print(f"Stored Energy:       {energy:.4e} Joules")

# Note: 'vmec_output.wout.rmnc' contains the Fourier Coefficients of the final shape

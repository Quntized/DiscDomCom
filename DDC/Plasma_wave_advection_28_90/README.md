Here is the README.md file formatted with proper LaTeX math rendering. You can copy and paste this directly into your GitHub repository or documentation.
Mathematical Documentation: Spectral Advection Solver
The mathematics behind this code relies on the Spectral Method for solving Partial Differential Equations (PDEs).
The core concept is efficient computation: calculus operations (like spatial derivatives) are computationally expensive and error-prone in "Real Space," but they transform into simple algebraic operations (multiplication) in "Fourier Space."
Below is the step-by-step breakdown of how the physics translates into mathematics, and finally into the specific C++ implementation.
1. The Physics Equation (Advection)
We are solving the Linear Advection Equation. This fundamental equation describes a wave shape u(x,t) moving with a constant speed c without changing its form.
∂t∂u​+c∂x∂u​=0
Where:

* ∂t∂u​: Describes how the wave changes over time.

* ∂x∂u​: Describes the slope (spatial gradient) of the wave.

* c: The speed of the wave propagation.

2. The Mathematical Trick (Fourier Transform)
Solving the advection equation directly on a grid (using Finite Differences) requires calculating differences between neighbors (ui+1​−ui​). This approach often introduces numerical errors known as numerical diffusion, which causes the wave to smear out over time.
Instead, we move to Fourier Space. We define the Fourier Transform u^(k,t) such that:
u(x)FFT​u^(k)
The most powerful property of the Fourier Transform is how it handles derivatives. A spatial derivative in real space becomes a multiplication by the imaginary unit i and the wavenumber k:
∂x∂​u(x)⟷ik⋅u^(k)
Key Insight: "Taking a derivative in real space is the same as multiplying by ik in Fourier space."
3. Solving the Equation
Let's apply the Fourier Transform to our original physics equation to solve it analytically in k-space.
A. Start with the PDE:
∂t∂u​+c∂x∂u​=0
B. Transform to Fourier Space:
dtdu^​+c(ik)u^=0
C. Rearrange:
dtdu^​=−i(kc)u^
D. Solve the ODE: This is now a simple Ordinary Differential Equation (ODE) of the form y′=−ay. The solution is an exponential function:
u^(k,t)=u^(k,0)⋅e−i(kct)
The Solution: To move a wave forward in time by t, we simply take its initial Fourier coefficients and rotate them by a phase angle θ=kct. This is called a Phase Shift.
4. Mapping Math to C++ Code
Here is exactly how those mathematical steps are implemented in the code using Kokkos and KokkosFFT.
Step A: The Initial Condition
Math:
u(x,0)=e−100(x−0.25)2
C++ Code:
C++Kokkos::parallel_for("Init", N, KOKKOS_LAMBDA(const int i) { double x = (double)i / N * L; u_real(i) = std::exp(-100.0 * (x - 0.25) * (x - 0.25)); });
Step B: Move to Fourier Space
Math:
u^(k)=F(u(x))
C++ Code:
C++

```
KokkosFFT::fft(ExecutionSpace(), u_real, u_hat);
```

Step C: The Phase Shift (Time Evolution)
Math:
u^(k,t)=u^(k,0)⋅(cos(θ)−isin(θ))
Where θ=k⋅c⋅t.
C++ Code:
C++

```
Kokkos::parallel_for("Advect", N/2 + 1, KOKKOS_LAMBDA(const int k_idx) {
    double k = 2.0 * M_PI * k_idx / L;
    
    // Calculate the rotation angle theta
    double theta = k * c * t_final;

    // Euler's Formula: e^(-i * theta)
    Kokkos::complex<double> phase_shift(std::cos(theta), -std::sin(theta));
    
    // Apply the rotation
    u_hat(k_idx) = u_hat(k_idx) * phase_shift;
});
```

Step D: Return to Real Space
Math:
u(x,t)=F−1(u^(k,t))
C++ Code:
C++
KokkosFFT::irfft(ExecutionSpace(), u_hat, u_real);
 
Give me this in readme.md file

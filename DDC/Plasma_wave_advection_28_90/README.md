Spectral Advection Solver
The mathematics behind this code relies on the Spectral Method for solving Partial Differential Equations (PDEs).
The core concept is efficient computation: calculus operations (like spatial derivatives) are computationally expensive and error-prone in "Real Space," but they transform into simple algebraic operations (multiplication) in "Fourier Space."
Below is the step-by-step breakdown of how the physics translates into mathematics, and finally into the specific C++ implementation.
1. The Physics Equation (Advection)
We are solving the Linear Advection Equation. This fundamental equation describes a wave shape $u(x,t)$ moving with a constant speed $c$ without changing its form.
$$\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0$$
Where:

$\frac{\partial u}{\partial t}$: Describes how the wave changes over time.
$\frac{\partial u}{\partial x}$: Describes the slope (spatial gradient) of the wave.
$c$: The speed of the wave propagation.

2. The Mathematical Trick (Fourier Transform)
Solving the advection equation directly on a grid (using Finite Differences) requires calculating differences between neighbors ($u_{i+1} - u_i$). This approach often introduces numerical errors known as numerical diffusion, which causes the wave to smear out over time.
Instead, we move to Fourier Space. We define the Fourier Transform $\hat{u}(k,t)$ such that:
$$u(x) \xrightarrow{\text{FFT}} \hat{u}(k)$$

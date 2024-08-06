# Calculation of the Musculotendon Equilibrium

This repository provides two methods for calculating musculotendon equilibrium using a simple optimal control problem with Bioptim. These methods are applied to a problem where an object with mass is attached to a muscle and needs to be moved either upward or downward, causing the muscle to contract or stretch.

The equilibrium is calculated using De Groote's and Millard's equations. The model includes a musculotendon system with slight damping.

**States**: \(x\) includes joint position \(q\), joint velocity \(\dot{q}\), and normalized muscle length \(\tilde{l}_M\).

**Controls**: \(u\) includes muscle activation and normalized muscle velocity control \(\tilde{v}_M\).

# Method 1: Calculation of Equilibrium Using Gradient Descent

For this method, using Millard's equations, the differential equation for normalized muscle length representing the musculotendon equilibrium is:

![Equation](https://latex.codecogs.com/svg.latex?f_{M0}%20\left%20(a%20f_{act}(\tilde{l}_M)%20f_V(\tilde{v}_M)%20%2B%20f_{pas}(\tilde{l}_M)%20%2B%20\beta%20\tilde{v}_M%20\right%20)\cos%20a%20-%20f_{M0}%20f_T(\tilde{l}_T)%20%3D%200)

where:

- \(a\): Muscle activation
- \(\tilde{l}_M\): Normalized muscle length
- \(\tilde{l}_T\): Normalized tendon length
- \(\tilde{v}_M\): Normalized muscle velocity
- \(f_{M0}\): Maximal isometric force
- \(f_{act}\): Muscle activation force
- \(f_V\): Muscle velocity force
- \(f_{pas}\): Muscle passive force
- \(\beta\): Coefficient of damping (\(\beta = 0.1\))
- \(f_T\): Tendon force

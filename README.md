# Calculation of the Musculotendon Equilibrium

This repository provides two methods for calculating musculotendon equilibrium using a simple optimal control problem with Bioptim. These methods are applied to a problem where an object with mass is attached to a muscle and needs to be moved either upward or downward, causing the muscle to contract or stretch.

The equilibrium is calculated using De Groote's and Millard's equations. The model includes a musculotendon system with slight damping.

States: x includes joint position q, joint velocity qdot​, and normalized muscle length lm_normalized​.

Controls: u includes muscle activation and normalized muscle velocity control vm_c_normalized​.

# Method 1: Calculation of Equilibrium Using Gradient Descent

For this method, thanks to Millard's equations we have the diffential equations of muscle length normalized which represent the musculotendon equilibrium:

```math
f_M \left( af_{act}(\tilde{l}_M) f_V(\tilde{v}_M) + f_{pas}(\tilde{l}_M) + \beta \tilde{v}_M \right) \cos a - f_M f_T(\tilde{l}_T) = 0
```
with 

```math
\begin{enumerate}
f_M :
\end{enumerate}
```

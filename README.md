# Calculation of the Musculotendon Equilibrium

This repository provides two methods for calculating musculotendon equilibrium using a simple optimal control problem with Bioptim. These methods are applied to a problem where an object with mass is attached to a muscle and needs to be moved either upward or downward, causing the muscle to contract or stretch.

The equilibrium is calculated using De Groote's and Millard's equations. The model includes a musculotendon system with slight damping.

**States**: \(x\) includes joint position \(q\), joint velocity \(\dot{q}\), and normalized muscle length \(\tilde{l}_M\).

**Controls**: \(u\) includes muscle activation and normalized muscle velocity control \(\tilde{v}_M\).

# Method 1: Calculation of Equilibrium Using Gradient Descent

For this method, using Millard's equations, we have the differential equation for normalized muscle length, which represents the musculotendon equilibrium:

$$
f_{M0} \left( a f_{act}(\tilde{l}_M) f_V(\tilde{v}_M) + f_{pas}(\tilde{l}_M) + \beta \tilde{v}_M \right) \cos a - f_{M0} f_T(\tilde{l}_T) = 0
$$

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

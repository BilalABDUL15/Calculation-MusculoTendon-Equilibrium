# Calculation of the Musculotendon Equilibrium

This repository provides two methods for calculating musculotendon equilibrium using a simple optimal control problem with Bioptim. These methods are applied to a problem where an object with mass is attached to a muscle and needs to be moved either upward or downward, causing the muscle to contract or stretch.

The equilibrium is calculated using De Groote's and Millard's equations. The model includes a musculotendon system with slight damping.

**States**: x includes joint position q, joint velocity $dot{q}$ , and normalized muscle length $\tilde{l_M}$.

**Controls**: u includes muscle activation and normalized muscle velocity control  $\tilde{v_M}$.


# Method 1: Calculation of Equilibrium Using Gradient Descent

For this method, using Millard's equations, we have the differential equation for normalized muscle length, which represents the musculotendon equilibrium:

```math
f_{M0} \left( af_{act}(\tilde{l_M}) f_V(\tilde{v_M}) + f_{pas}(\tilde{l_M}) + \beta \tilde{v_M} \right) \cos a - f_{M0} f_T(\tilde{l_T}) = 0
```

with 

- $a$ : activation of the muscle
- $\tilde{l_M}$ : muscle length normalized
- $\tilde{l_T}$ : tendon length normalized
- $\tilde{v_M}$ : muscle velocity normalized
- $f_{M0}$ : maximal isometric force
- $f_{act}$ : muscle activation force
- $f_V$ : muscle velocity force
- $f_{pas}$ : muscle passive force
- $\beta$ : coefficient of damping = 0.1
- $f_T$ : tendon force


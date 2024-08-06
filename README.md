# Calculation of the Musculotendon Equilibrium

This repository provides two methods for calculating musculotendon equilibrium using a simple optimal control problem with Bioptim. These methods are applied to a problem where an object with mass is attached to a muscle and needs to be moved either upward or downward, causing the muscle to contract or stretch.

The equilibrium is calculated using De Groote's and Millard's equations. The model includes a musculotendon system with slight damping.

States: x includes joint position q, joint velocity qdot​, and normalized muscle length lm_normalized​.

Controls: u includes muscle activation and normalized muscle velocity control vm_c_normalized​.

# Method 1: Calculation of Equilibrium Using Gradient Descent

For this method, thanks to Millard's equations we have the diffential equations of muscle length normalized which represent the musculotendon equilibrium:

```math
f_{M0} \left( af_{act}(\tilde{l_M}) f_V(\tilde{v_M}) + f_{pas}(\tilde{l_M}) + \beta \tilde{v_M} \right) \cos a - f_{M0} f_T(\tilde{l_T}) = 0
```
with 

1. $a$ : activation of the muscle
2. $\tilde{l_M}$ : muscle length normalized
3. $\tilde{l_T}$ : tendon length normalized
4. $\tilde{v_M}$ : muscle velocity normalized
5. $f_{M0}$ : maximal isometric force
6. $f_{act}$ : muscle activation force
7. $f_V$ : muscle velocity force
8. $f_{pas}$ : muscle passive force
9. $\beta$ : coefficient of damping = 0.1
10. $f_T$ : tendon force

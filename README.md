# Calculation of the Musculotendon Equilibrium

This repository provides two methods for calculating musculotendon equilibrium using a simple optimal control problem with Bioptim. These methods are applied to a problem where an object with mass is attached to a muscle and needs to be moved either upward or downward, causing the muscle to contract or stretch.

The equilibrium is calculated using De Groote's and Millard's equations. The model includes a musculotendon system with slight damping.

**States**: x includes joint position q, joint velocity $dot{q}$ , and normalized muscle length $\tilde{l_M}$.

**Controls**: u includes muscle activation and normalized muscle velocity control  $\tilde{v_M}$.



We have the differential equation for normalized muscle length, which represents the musculotendon equilibrium:

```math
f_{M0} \left( actf_{act}(\tilde{l_M}) f_V(\tilde{v_M}) + f_{pas}(\tilde{l_M}) + \beta \tilde{v_M} \right) \cos a - f_{M0} f_T(\tilde{l_T}) = 0
```

with 

- $act$ : muscle activation
- $\tilde{l_M}$ : muscle length normalized
- $\tilde{l_T}$ : tendon length normalized
- $\tilde{v_M}$ : muscle velocity normalized
- $f_{M0}$ : maximal isometric force
- $f_{act}$ : muscle activation force
- $f_V$ : muscle velocity force
- $f_{pas}$ : muscle passive force
- $\beta$ : coefficient of damping = 0.1
- $f_T$ : tendon force

Moreover, we also have the equality $l_{MT} = l_M\cos{\alpha} + l_T$ where $\tilde{l_{MT}}$ is the musculotendon length which is not unknown. So the differential equation depends only on the activation, muscle lengh normalized and muscle velocity normalized.

Because $act$ is a control and $\tilde{l_M}$ is a state, the only unknown here is $\tilde{v_M}$. So two methods have been proposed to calculate $\tilde{v_M}$.


# Method 1: Calculation of Equilibrium Using Gradient Descent
To determine $\tilde{v_M}$, we use gradient descent on the equation. This method allows us to obtain the correct value of $\tilde{v_M}$ Once we have this value, we can integrate it to determine the value of the state $\tilde{l_M}$. However, an initial guess is needed for the gradient descent, so we use $\tilde{v_M}_{control}$ as our initial guess. This enables us to calculate the state values at each node of our optimal control problem.

To ensure that $\tilde{v_M}_{control}$  is accurate, we introduce a multinode constraint. This constraint allows us to transmit the value of $\tilde{l_M}$ to the next interval of resolution. 

Without this constraint, the initial guess at the beginning of each calculation interval would be random. By integrating the dynamics over the interval, we obtain the state values at the last node of the interval. 

We then calculate $\tilde{l_M}$ at this node using these states and the current $\tilde{v_M}_{control}$. 

Finally, we impose an equality between the next $\tilde{v_M}_{control}$ and the calculated $\tilde{l_M}$.

This ensures that the value of $\tilde{l_M}$ is correctly transmitted to the next interval.

# Method 2 : Calculation of Equilibrium Linear $f_V$
To determine $\tilde{v_M}$, we will first calculate the tangent of the current $\tilde{v_M}_{control}$ of $f_V$ function. Thanks to this, we will be able to linearize our differential equation.

If we note 
```math 
\tilde{v}_M = \gamma \tilde{v}_M + \theta
```
Then we have 
```math
\frac{ f_T(\left \frac{\frac{l_{MT} - l_M \cos{\alpha}}{l_{TS}}} {\cos{\alpha}} )\right - f_{pas}(\tilde{\ell}_M) - a \theta f_{act}(\tilde{l_M})}{\gamma a f_{act}(\tilde{l_M}) + \beta}
```


We can integrate this equation and obtain the values of our state $\tilde{l_M}$.

# Calculation of the Musculotendon Equilibrium

This repository provides two methods for calculating musculotendon equilibrium using a simple optimal control problem with **Bioptim**. These methods are applied to a problem where an object with mass is attached to a muscle and needs to be moved either upward or downward, causing the muscle to contract or stretch.

<p>
  <img src="https://i2.wp.com/youngmok.com/wp-content/uploads/2013/11/hill_type_muscle_model.png?w=425" alt="Hill Model">
</p>




The equilibrium is calculated using De Groote's and Millard's equations. The model includes a musculotendon system with slight damping.

**States**: x includes joint position q, joint velocity $dot{q}$ , and normalized muscle length $\tilde{\ell_M}$.

**Controls**: u includes muscle activation and normalized muscle velocity control  $\tilde{v_M}$.



We have the differential equation for normalized muscle length, which represents the musculotendon equilibrium:

```math
f_{M0} \left( actf_{act}(\tilde{\ell_M}) f_V(\tilde{v_M}) + f_{pas}(\tilde{\ell_M}) + \beta \tilde{v_M} \right) \cos a - f_{M0} f_T(\tilde{\ell_T}) = 0
```

with 

- $act$ : muscle activation
- $\tilde{\ell_M}$ : muscle length normalized
- $\tilde{\ell_T}$ : tendon length normalized
- $\tilde{v_M}$ : muscle velocity normalized
- $f_{M0}$ : maximal isometric force
- $f_{act}$ : muscle activation force
- $f_V$ : muscle velocity force
- $f_{pas}$ : muscle passive force
- $\beta$ : coefficient of damping = 0.1
- $f_T$ : tendon force

Moreover, we also have the equality $\ell_{MT} = \ell_M\cos{\alpha} + \ell_T$ where $\tilde{\ell_{MT}}$ is the musculotendon length which is not unknown. So the differential equation depends only on the activation, muscle lengh normalized and muscle velocity normalized.

Because $act$ is a control and $\tilde{\ell_M}$ is a state, the only unknown here is $\tilde{v_M}$. So two methods have been proposed to calculate $\tilde{v_M}$.


## Method 1: Calculation of Equilibrium Using Gradient Descent
To determine $\tilde{v_M}$, we use gradient descent on the equation. This method allows us to obtain the correct value of $\tilde{v_M}$ Once we have this value, we can integrate it to determine the value of the state $\tilde{\ell_M}$. However, an initial guess is needed for the gradient descent, so we use $\tilde{v_M}_{control}$ as our initial guess. This enables us to calculate the state values at each node of our optimal control problem.

To ensure that $\tilde{v_M}_{control}$  is accurate, we introduce a multinode constraint. This constraint allows us to transmit the value of $\tilde{\ell_M}$ to the next interval of resolution. 

Without this constraint, the initial guess at the beginning of each calculation interval would be random. By integrating the dynamics over the interval, we obtain the state values at the last node of the interval. 

We then calculate $\tilde{\ell_M}$ at this node using these states and the current $\tilde{v_M}_{control}$. 

Finally, we impose an equality between the next $\tilde{v_M}_{control}$ and the calculated $\tilde{l_M}$.

This ensures that the value of $\tilde{\ell_M}$ is correctly transmitted to the next interval.

## Method 2 : Calculation of Equilibrium Linear $f_V$
To determine $\tilde{v_M}$, we will first calculate the tangent of the current $\tilde{v_M}_{control}$ of $f_V$ function. Thanks to this, we will be able to linearize our differential equation.

If we note 
```math 
f_V(\tilde{v}_M) = \gamma \tilde{v}_M + \theta
```
Then we have 
```math
\frac{  \frac{f_T \left(\frac{\ell_{MT} - \ell_M \cos{\alpha}}{\ell_{TS}} \right)} {\cos{\alpha}} - f_{pas}(\tilde{\ell_M}) - a \theta f_{act}(\tilde{\ell_M})}{\gamma a f_{act}(\tilde{\ell_M}) + \beta}
```


We can then integrate this equation and obtain the value of our state $\tilde{\ell_M}$.

## Model 



<p>
  <img src="https://github.com/BilalABDUL15/Calculation-MusculoTendon-Equilibrium/raw/main/images/cube_1_%20muscle.png" alt="Image du cube musculaire">
</p>


### Source Article:
Millard's article : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3705831/pdf/bio_135_2_021005.pdf

De Groote's article : https://link.springer.com/content/pdf/10.1007/s10439-016-1591-9.pdf

Sartori article : https://www.biorxiv.org/content/10.1101/2024.05.14.594110v1.full.pdf





# Calculation of the Musculotendon Equilibrium

This repository provides two methods for calculating musculotendon equilibrium using a simple optimal control problem with Bioptim. These methods are applied to a problem where an object with mass is attached to a muscle and needs to be moved either upward or downward, causing the muscle to contract or stretch.

The equilibrium is calculated using De Groote's and Millard's equations. The model includes a musculotendon system with slight damping.

States: x includes joint position q, joint velocity qdot​, and normalized muscle length lm_normalized​.

Controls: u includes muscle activation and normalized muscle velocity control vm_c_normalized​.

# Method 1: Calculation of Equilibrium Using Gradient Descent


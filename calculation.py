import casadi
import numpy as np
from shoulder import ModelBiorbd, ControlsTypes, MuscleParameter
import timeit

# Initialization of constant and parameters
musculotendon_length = 0.50
muscle_length0 = 0.31
tendon_length0 = 0.20

maximalIsometricForce = 700
pennation = 0.52  # equal to 30 degrees
activation = 0.1  # activationtendon_length
muscle_velocity0 = 0.03
muscle_velocity_max = 0.08

# ft(lt) parameters
c1 = 0.2
c2 = 0.995
c3 = 0.250
kt = 35


# fact parameters
b11 = 0.815
b21 = 1.055
b31 = 0.162
b41 = 0.063

b12 = 0.433
b22 = 0.717
b32 = -0.030
b42 = 0.200

b13 = 0.100
b23 = 1.000
b33 = 0.354
b43 = 0.000

# fpas parameters
kpe = 4.0
e0 = 0.6


# Fvm parameters
d1 = -0.318
d2 = -8.149
d3 = -0.374
d4 = 0.886


# Force passive definition = fpce
def fpas(muscle_length_normalized):
    return (np.exp(kpe * (muscle_length_normalized - 1) / e0) - 1) / (np.exp(kpe) - 1)


# Force active definition = flce
def fact(muscle_length_normalized):
    return (
        b11 * np.exp((-0.5) * ((muscle_length_normalized - b21) ** 2) / (b31 + b41 * muscle_length_normalized))
        + b12 * np.exp((-0.5) * (muscle_length_normalized - b22) ** 2 / (b32 + b42 * muscle_length_normalized))
        + b13 * np.exp((-0.5) * (muscle_length_normalized - b23) ** 2 / (b33 + b43 * muscle_length_normalized))
    )


# Muscle force velocity equation = fvce
def fvm(muscle_velocity):
    return d1 * np.log((d2 * muscle_velocity + d3) + np.sqrt(((d2 * muscle_velocity + d3) ** 2) + 1)) + d4


# ft(lt)
def ft(tendon_length_normalized):
    return c1 * np.exp(kt * (tendon_length_normalized - c2)) - c3


# Definition tendon Force (Ft) with the 3rd equation of De Groote : Ft = maximalIsometricForce * ft(tendon_length)
def calculation_tendonforce_3rd_equation(tendon_length_normalized):
    return maximalIsometricForce * ft(tendon_length_normalized)


# Definition Ft with the 7th equation of De Groote : Ft = Fm * cos(pennation)
def calculation_tendonforce_7th_equation(muscle_length_normalized, muscle_velocity_normalized):
    return (
        maximalIsometricForce
        * (
            fpas(muscle_length_normalized)
            + activation * fvm(muscle_velocity_normalized) * fact(muscle_length_normalized)
        )
        * np.cos(pennation)
    )


def fvm_inverse(muscle_length_normalized, tendon_length_normalized):
    return (ft(tendon_length_normalized) / np.cos(pennation) - fpas(muscle_length_normalized)) / (
        activation * fact(muscle_length_normalized)
    )


def verification_equal_Ft(tendon_length_normalized, muscle_length_normalized, muscle_velocity_normalized):
    return calculation_tendonforce_3rd_equation(tendon_length_normalized) - calculation_tendonforce_7th_equation(
        muscle_length_normalized, muscle_velocity_normalized
    )


# IPOPT Method
"""

def calcul_longueurs(musculotendon_length):
    # Declare the variables
    muscle_length = casadi.SX.sym("x", 1, 1)
    tendon_length = casadi.SX.sym("y", 1, 1)
    muscle_velocity = casadi.SX.sym("z", 1, 1)
    x = casadi.vertcat(tendon_length, muscle_length, muscle_velocity)
    x0 = casadi.DM(
        [
            musculotendon_length * 1 / 3,
            ((musculotendon_length * 2 / 3) / np.cos(pennation)),
            muscle_velocity0,
        ]
    )

    # Declare the constraints
    g1 = musculotendon_length - tendon_length - muscle_length * np.cos(pennation)
    ft3 = calculation_tendonforce_3rd_equation(tendon_length / tendon_length0)
    ft7 = calculation_tendonforce_7th_equation(muscle_length / muscle_length0, muscle_velocity / muscle_velocity0)
    g2 = ft3 - ft7
    g3 = (
        1
        / d2
        * np.sinh(
            1
            / d1
            * (
                (ft(tendon_length / tendon_length0) / np.cos(pennation) - fpas(muscle_length / muscle_length0))
                / (activation * fact(muscle_length / muscle_length0))
                - d4
            )
        )
        * muscle_velocity_max
        - muscle_velocity
    )
    g = casadi.vertcat(g1, g2, g3)
    # Declare the solver
    solver = casadi.nlpsol("solver", "ipopt", {"x": x, "g": g})
    sol = solver(x0=x0, ubx=musculotendon_length, lbx=0, ubg=0, lbg=0)
    Tendon_force_1 = calculation_tendonforce_3rd_equation(sol["x"][0] / tendon_length0)
    Tendon_force_2 = calculation_tendonforce_7th_equation(sol["x"][1] / muscle_length0, sol["x"][2] / muscle_velocity0)

    print(
        "Tendon length cm:",
        sol["x"][0],
        "Muscle length cm:",
        sol["x"][1],
        "Velocity muscle cm/s:",
        sol["x"][2],
        1
        / d2
        * np.sinh(
            1
            / d1
            * (
                (ft(sol["x"][0] / tendon_length0) / np.cos(pennation) - fpas(sol["x"][1] / muscle_length0))
                / (activation * fact(sol["x"][1] / muscle_length0))
                - d4
            )
        )
        * muscle_velocity_max,
        "Tendon force 3 N:",
        Tendon_force_1,
        "Tendon force 7 N:",
        Tendon_force_2,
        "Musculotendon length",
        sol["x"][0] + sol["x"][1] * np.cos(pennation),
    )
    return {
        "Tendon_force": Tendon_force_1,
        "Tendon_length": sol["x"][0],
        "Muscle_length": sol["x"][1],
        "Muscle_velocity": sol["x"][2],
    }


"""


# Newton method 3 times faster than IPOPT method
def calcul_longueurs(musculotendon_length):
    # Declare the variables
    muscle_length = casadi.SX.sym("x", 1, 1)
    tendon_length = casadi.SX.sym("y", 1, 1)
    muscle_velocity = casadi.SX.sym("z", 1, 1)
    x = casadi.vertcat(tendon_length, muscle_length, muscle_velocity)
    x0 = casadi.DM(
        [
            musculotendon_length * 1 / 3,
            ((musculotendon_length * 2 / 3) / np.cos(pennation)),
            muscle_velocity0,
        ]
    )

    # Declare the constraints
    g1 = musculotendon_length - (tendon_length + muscle_length * np.cos(pennation))
    g2 = verification_equal_Ft(
        tendon_length / tendon_length0, muscle_length / muscle_length0, muscle_velocity / muscle_velocity0
    )
    g3 = (1 / d2) * np.sinh(
        (1 / d1) * (fvm_inverse(muscle_length / muscle_length0, tendon_length / tendon_length0) - d4)
    ) * muscle_velocity_max - muscle_velocity

    g = casadi.vertcat(g1, g2, g3)

    # Declare the solver
    solver = casadi.rootfinder("solver", "newton", {"x": x, "g": g})
    sol = solver(x0=x0)
    Tendon_force_1 = calculation_tendonforce_3rd_equation(sol["x"][0] / tendon_length0)
    Tendon_force_2 = calculation_tendonforce_7th_equation(sol["x"][1] / muscle_length0, sol["x"][2] / muscle_velocity0)

    print(
        "Tendon length m:",
        sol["x"][0],
        "Muscle length m:",
        sol["x"][1],
        "Velocity muscle m/s:",
        (1 / d2)
        * np.sinh((1 / d1) * (fvm_inverse(sol["x"][1] / muscle_length0, sol["x"][0] / tendon_length0) - d4))
        * muscle_velocity_max,
        sol["x"][2],
        "Tendon force 3 N:",
        Tendon_force_1,
        "Tendon force 7 N:",
        Tendon_force_2,
        "Musculotendon length",
        sol["x"][0] + sol["x"][1] * np.cos(pennation),
    )

    return {
        "Tendon_force": Tendon_force_1,
        "Tendon_length": sol["x"][0],
        "Muscle_length": sol["x"][1],
        "Muscle_velocity": sol["x"][2],
    }


def main():
    # calcul_longueurs(musculotendon_length)
    execution_time = timeit.timeit(lambda: calcul_longueurs(musculotendon_length), number=1)
    print("Time execution:", execution_time)


"""

    if results["muscle_length"] * np.cos(pennation) > results["musculotendon_length"] or results["tendon_length"] < 0:
        print("Pas de solution pour ce problème.")
    else:
        print(
            "\nMusculotendon length:",
            results["musculotendon_length"],
            "cm",
            "\nWith this value:",
            "\nFt = ",
            results["tendforce_3rd_equation"],
            "N",
            "\nMuscle_length = ",
            results["muscle_length"],
            "cm",
            "\nTendon_length = ",
            results["tendon_length"],
            "cm",
        )
        print(
            f"\nVerification of the equation of De Groote : Musculotendon length: {results["musculotendon_length"]} ?= Tendon_length: {results["tendon_length"]} + Muscle_length * cos(pennation): {results["muscle_length"] * np.cos(pennation)} "
        )

"""
if __name__ == "__main__":
    main()

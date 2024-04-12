import casadi
import numpy as np
from shoulder import ModelBiorbd, ControlsTypes, MuscleParameter

# Initialization of constant
musculotendon_length = 0.30
muscle_length0 = 0.21
tendon_length0 = 0.09
kt = 35  # constante of ft(tendon_length)
maximalIsometricForce = 900
pennation = 0.52
threshold = 1e-8
activation = 0.5  # activationtendon_lengthats
velocity_muscle0 = 2

L0 = 0.014  # Standard length L0
a_velocity = 0.4 * maximalIsometricForce
b_velocity = 0.85 * L0


# Force passive definition = fpce
def fpas(muscle_length):
    return ((muscle_length / muscle_length0) ** 3) * np.exp(8 * (muscle_length / muscle_length0) - 12.9)


# Force active definition = flce
def fact(muscle_length):
    return 1 - (((muscle_length / muscle_length0) - 1) / 0.5) ** 2


# Force velocity equation = fvce
def fv(vm):
    return (2 * velocity_muscle0 - b_velocity + vm * a_velocity / maximalIsometricForce) / (
        velocity_muscle0 - b_velocity
    )


# Definition tendon Force (Ft) with the 3rd equation of De Groote : Ft = maximalIsometricForce * ft(tendon_length)
def calculation_tendonforce_3rd_equation(tendon_length):
    return maximalIsometricForce * kt * (muscle_length0 - tendon_length) ** 2


# Definition Ft with the 7th equation of De Groote : Ft = Fm * cos(pennation)
def calculation_tendonforce_7th_equation(muscle_length):
    return maximalIsometricForce * (fpas(muscle_length) + activation * fv(30) * fact(muscle_length)) * np.cos(pennation)


def calcul_longueurs(musculotendon_length):

    # initial guess muscle_length
    tendon_length = musculotendon_length / 3.2
    muscle_length = (musculotendon_length - tendon_length) / np.cos(pennation)
    # print(muscle_length * np.cos(pennation), tendon_length)
    muscle_length = muscle_length / muscle_length0
    tendon_length = tendon_length / tendon_length0
    # print(muscle_length * np.cos(pennation), tendon_length)
    # Calculation of Ft with the Two equations
    tendforce_3rd_equation = calculation_tendonforce_3rd_equation(tendon_length)
    tendforce_7th_equation = calculation_tendonforce_7th_equation(muscle_length)
    # cpt = 0

    # First loop to determine the int number of Ft
    while int(tendforce_3rd_equation - tendforce_7th_equation) != 0:

        if tendforce_3rd_equation > tendforce_7th_equation:

            # tendforce_3rd_equation > tendforce_7th_equation and muscle_length < tendon_length
            if muscle_length * np.cos(pennation) <= tendon_length:
                eps = abs(muscle_length * np.cos(pennation) - musculotendon_length) / 2
                muscle_length = (muscle_length + eps) / np.cos(pennation)
                tendon_length = musculotendon_length - muscle_length * np.cos(pennation)

            # tendforce_3rd_equation > tendforce_7th_equation and muscle_length > tendon_length
            else:
                eps = abs(muscle_length * np.cos(pennation) - tendon_length) / 20
                muscle_length += eps / np.cos(pennation)
                tendon_length = musculotendon_length - muscle_length * np.cos(pennation)

        elif tendforce_7th_equation > tendforce_3rd_equation:

            # tendforce_3rd_equation < tendforce_7th_equation and muscle_length > tendon_length
            if muscle_length * np.cos(pennation) >= tendon_length:
                eps = abs(muscle_length * np.cos(pennation) - musculotendon_length) / 20
                muscle_length = muscle_length - (eps / np.cos(pennation))
                tendon_length = musculotendon_length - muscle_length * np.cos(pennation)

            # tendforce_3rd_equation < tendforce_7th_equation and muscle_length < tendon_length
            else:
                eps = abs(muscle_length * np.cos(pennation) - tendon_length) / 20
                tendon_length = eps
                muscle_length = (musculotendon_length - tendon_length) / np.cos(pennation)

        tendforce_3rd_equation = calculation_tendonforce_3rd_equation(tendon_length)
        tendforce_7th_equation = calculation_tendonforce_7th_equation(muscle_length)
        """
        if cpt == 100:
            print([tendforce_3rd_equation, tendforce_7th_equation, muscle_length * np.cos(pennation), tendon_length, eps])
            cpt = 0
        else:
            cpt += 1

    print([tendforce_3rd_equation, tendforce_7th_equation, muscle_length * np.cos(pennation), tendon_length, musculotendon_length])
    print("Changement de boucle")
        """
    # Second loop to determine the number after the comma
    while np.abs(tendforce_3rd_equation - tendforce_7th_equation) > threshold:
        if tendforce_3rd_equation > tendforce_7th_equation:
            if muscle_length * np.cos(pennation) <= tendon_length:
                eps = abs(muscle_length * np.cos(pennation) - musculotendon_length) / 2
                muscle_length = (muscle_length + eps) / np.cos(pennation)
                tendon_length = musculotendon_length - muscle_length * np.cos(pennation)
            else:
                eps = abs(muscle_length * np.cos(pennation) - tendon_length) * threshold
                muscle_length += eps / np.cos(pennation)
                tendon_length = musculotendon_length - muscle_length * np.cos(pennation)
        else:
            if muscle_length * np.cos(pennation) >= tendon_length:
                eps = abs(muscle_length * np.cos(pennation) - musculotendon_length) * threshold
                muscle_length -= eps / np.cos(pennation)
                tendon_length = musculotendon_length - muscle_length * np.cos(pennation)
            else:
                eps = abs(muscle_length * np.cos(pennation) - tendon_length) / 2
                tendon_length = eps
                muscle_length = (musculotendon_length - tendon_length) / np.cos(pennation)
        tendforce_3rd_equation = calculation_tendonforce_3rd_equation(tendon_length)
        tendforce_7th_equation = calculation_tendonforce_7th_equation(muscle_length)
        """
        if cpt == 100:
            print([tendforce_3rd_equation, tendforce_7th_equation, muscle_length * np.cos(pennation), tendon_length, musculotendon_length])
            cpt = 0
        else:
            cpt += 1
        """
    return {
        "tendforce_3rd_equation": tendforce_3rd_equation,
        "tendforce_7th_equation": tendforce_7th_equation,
        "muscle_length": muscle_length,
        "tendon_length": tendon_length,
        "musculotendon_length": musculotendon_length,
    }


def main():
    results = calcul_longueurs(musculotendon_length)
    # print(resutendon_lengthats)
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
            f"\nVerification of the equation of De Groote : Musculotendon length: {results["musculotendon_length"]} ?= Tendon_length: {results["tendon_length"]} + Muscle_length * cos(pennation): {results["muscle_length"] * np.cos(pennation)} ",
            results["musculotendon_length"] == results["muscle_length"] * np.cos(pennation) + results["tendon_length"],
        )


if __name__ == "__main__":
    main()



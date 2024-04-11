import casadi
import numpy as np
from shoulder import ModelBiorbd, ControlsTypes, MuscleParameter


# Initialization of constant
lmt = 0.30
lm0 = 0.21
lt0 = 0.09
kt = 35  # constante of ft(lt)
Fm0 = 900  # maximalIsometricForce
alpha = 0.52  # pennation
threshold = 1e-8
a = 0.5  # activation


# Force passive definition
def fpas(lm):
    return ((lm / lm0) ** 3) * np.exp(8 * (lm / lm0) - 12.9)


# Force active definition
def fact(lm):
    return 1 - (((lm / lm0) - 1) / 0.5) ** 2


# Force velocity equation
def fv(vm):
    return 1


# Definition tendon Force (Ft) with the 3rd equation of De Groote : Ft = Fm0 * ft(lt)
def ft3(lt):
    return Fm0 * kt * (lm0 - lt) ** 2


# Definition Ft with the 7th equation of De Groote : Ft = Fm * cos(pennation)
def ft7(lm):
    return Fm0 * (fpas(lm) + a * fv(1) * fact(lm)) * np.cos(alpha)


def calcul_longueurs(lmt):

    # initial guess lm
    lt = lmt / 3.2
    lm = (lmt - lt) / np.cos(alpha)
    # print(lm * np.cos(alpha), lt)
    lm = lm / lm0
    lt = lt / lt0
    # print(lm * np.cos(alpha), lt)
    # Calculation of Ft with the Two equations
    ft_3 = ft3(lt)
    ft_7 = ft7(lm)
    cpt = 0
    # First loop to determine the int number of Ft
    while int(ft_3 - ft_7) != 0:
        if ft_3 > ft_7:

            # Ft_3 > Ft_7 and lm < lt
            if lm * np.cos(alpha) <= lt:
                eps = abs(lm * np.cos(alpha) - lmt) / 2
                lm = (lm + eps) / np.cos(alpha)
                lt = lmt - lm * np.cos(alpha)

            # Ft_3 > Ft_7 and lm > lt
            else:
                eps = abs(lm * np.cos(alpha) - lt) / 20
                lm += eps / np.cos(alpha)
                lt = lmt - lm * np.cos(alpha)
        elif ft_7 > ft_3:

            # Ft_3 < Ft_7 and lm > lt
            if lm * np.cos(alpha) >= lt:
                eps = abs(lm * np.cos(alpha) - lmt) / 20
                lm = lm - (eps / np.cos(alpha))
                lt = lmt - lm * np.cos(alpha)

            # Ft_3 < Ft_7 and lm < lt
            else:
                eps = abs(lm * np.cos(alpha) - lt) / 20
                lt = eps
                lm = (lmt - lt) / np.cos(alpha)
        ft_3 = ft3(lt)
        ft_7 = ft7(lm)
        """
        if cpt == 100:
            print([ft_3, ft_7, lm * np.cos(alpha), lt, eps])
            cpt = 0
        else:
            cpt += 1

    print([ft_3, ft_7, lm * np.cos(alpha), lt, lmt])
    print("Changement de boucle")
        """
    # Second loop to determine the number after the comma
    while np.abs(ft_3 - ft_7) > threshold:
        if ft_3 > ft_7:
            if lm * np.cos(alpha) <= lt:
                eps = abs(lm * np.cos(alpha) - lmt) / 2
                lm = (lm + eps) / np.cos(alpha)
                lt = lmt - lm * np.cos(alpha)
            else:
                eps = abs(lm * np.cos(alpha) - lt) * threshold
                lm += eps / np.cos(alpha)
                lt = lmt - lm * np.cos(alpha)
        else:
            if lm * np.cos(alpha) >= lt:
                eps = abs(lm * np.cos(alpha) - lmt) * threshold
                lm -= eps / np.cos(alpha)
                lt = lmt - lm * np.cos(alpha)
            else:
                eps = abs(lm * np.cos(alpha) - lt) / 2
                lt = eps
                lm = (lmt - lt) / np.cos(alpha)
        ft_3 = ft3(lt)
        ft_7 = ft7(lm)
        """
        if cpt == 100:
            print([ft_3, ft_7, lm * np.cos(alpha), lt, lmt])
            cpt = 0
        else:
            cpt += 1
        """
    return {"ft_3": ft_3, "ft_7": ft_7, "lm": lm, "lt": lt, "lmt": lmt}


def main():
    resultats = calcul_longueurs(lmt)
    # print(resultats)
    if resultats["lm"] * np.cos(alpha) > resultats["lmt"] or resultats["lt"] < 0:
        print("Pas de solution pour ce problème.")
    else:
        print(
            "Longueur du musculotendon:",
            resultats["lmt"],
            "\nPour cette valeur de musculotendon, on a :",
            "\nFt = ",
            resultats["ft_3"],
            "N",
            "\nLm = ",
            resultats["lm"],
            "cm",
            "\nLt = ",
            resultats["lt"],
            "cm",
        )


if __name__ == "__main__":
    main()
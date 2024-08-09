import casadi
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton


"""

    1) Normalized Data

 - To normalize tendon velocity divide by tendon slack length
 - To normalize muscle velocity divide by muscle velocity max
 - To normalize muscle Force divide by maximal Isometric Force
 - To normalize tendon length divide by tendon slack length
 - To normalize muscle length divide by optimal fiber length

"""


# Initialization of constant and parameters
# Length
# Muscle BICLong Parameters
musculotendon_length = 0.220
optimal_fiber_length = 0.11570  # optimal muscle fiber length
tendon_slack_length = 0.09  # tendon slack length


# Other parameters

maximalIsometricForce = 624.300
pennation = 0.000
activation = 0.3  # Muscle Activation


# Velocity

musculotendon_velocity = 0.007027430619001448

muscle_velocity_max = 5.32 * optimal_fiber_length
tendon_velocity_guess = musculotendon_velocity * 0.5
muscle_velocity_guess = (musculotendon_velocity - tendon_velocity_guess) * np.cos(pennation)


# ft(lt) parameters
c1 = 0.2
c2 = 0.995
c3 = 0.250
kt = 35


# fact / flce parameters
b11 = 0.814483478343008
b21 = 1.055033428970575
b31 = 0.162384573599574
b41 = 0.063303448465465

b12 = 0.433004984392647
b22 = 0.716775413397760
b32 = -0.029947116970696
b42 = 0.200356847296188

b13 = 0.100
b23 = 1.000
b33 = 0.354
b43 = 0.000

# fpas / fpce parametersExecution time of
kpe = 4.0
e0 = 0.6


# Fvm / fvce parameters
d1 = -0.318
d2 = -8.149
d3 = -0.374
d4 = 0.886

# Parameters for damping
damp = 0.1  # By default it values is 0.1


# Force passive definition = fpce
# Warning modification of the equation du to sign issue when muscle_length_normalized is under 1
def fpas(muscle_length_normalized):
    offset = (np.exp(kpe * (0.5 - 1) / e0) - 1) / (np.exp(kpe) - 1)
    return (np.exp(kpe * (muscle_length_normalized - 1) / e0) - 1) / (np.exp(kpe) - 1) - offset


# Force active definition = flce
def fact(muscle_length_normalized):
    return (
        b11 * np.exp((-0.5) * ((muscle_length_normalized - b21) ** 2) / ((b31 + b41 * muscle_length_normalized) ** 2))
        + b12 * np.exp((-0.5) * (muscle_length_normalized - b22) ** 2 / ((b32 + b42 * muscle_length_normalized) ** 2))
        + b13 * np.exp((-0.5) * (muscle_length_normalized - b23) ** 2 / ((b33 + b43 * muscle_length_normalized) ** 2))
    )


# Muscle force velocity equation = fvce
def fvm(muscle_velocity_normalized):
    return (
        d1 * np.log((d2 * muscle_velocity_normalized + d3) + np.sqrt(((d2 * muscle_velocity_normalized + d3) ** 2) + 1))
        + d4
    )


# ft(lt) tendon force calculation with tendon length normalized
def ft(tendon_length_normalized):
    return c1 * np.exp(kt * (tendon_length_normalized - c2)) - c3


# Definition tendon Force (Ft) with the 3rd equation of De Groote : Ft = maximalIsometricForce * ft(tendon_length)
def calculation_tendon_force(tendon_length_normalized):
    return maximalIsometricForce * ft(tendon_length_normalized)


# Definition Ft with the 7th equation of De Groote : Ft = Fm * cos(pennation)
def calculation_muscle_force(muscle_length_normalized, muscle_velocity_normalized):
    Fm = maximalIsometricForce * (
        fpas(muscle_length_normalized) + activation * fvm(muscle_velocity_normalized) * fact(muscle_length_normalized)
    )
    return Fm * np.cos(pennation)


# Calculation muscle velocity force with muscle force equation
def fvm_inverse(muscle_length_normalized, tendon_length_normalized):
    return (ft(tendon_length_normalized) / np.cos(pennation) - fpas(muscle_length_normalized)) / (
        activation * fact(muscle_length_normalized)
    )


# Calculation of tendon force obtained with the 3rd equation minus tendon force obtained with the 4th equation
def verification_equal_Ft(tendon_length_normalized, muscle_length_normalized, muscle_velocity_normalized):
    return calculation_tendon_force(tendon_length_normalized) - calculation_muscle_force(
        muscle_length_normalized, muscle_velocity_normalized
    )


# Muscle force velocity equation = fvce
def fvm_linear(muscle_velocity_normalized, point):
    a, b = tangent_factor_calculation(point)
    return a * muscle_velocity_normalized + b


def fdamp(muscle_velocity_normalized):
    return damp * muscle_velocity_normalized


# Definition Ft with the 7th equation of De Groote : Ft = Fm * cos(pennation)
def calculation_tendonforce_7th_equation_damped_linear(muscle_length_normalized, muscle_velocity_normalized, point=0):
    Fm = maximalIsometricForce * (
        fpas(muscle_length_normalized)
        - fdamp(muscle_velocity_normalized)
        + activation * fvm_linear(muscle_velocity_normalized, point) * fact(muscle_length_normalized)
    )
    return Fm * np.cos(pennation)


def calculation_tendonforce_7th_equation_damped(muscle_length_normalized, muscle_velocity_normalized):
    Fm = maximalIsometricForce * (
        fpas(muscle_length_normalized)
        - fdamp(muscle_velocity_normalized)
        + activation * fvm(muscle_velocity_normalized) * fact(muscle_length_normalized)
    )
    return Fm * np.cos(pennation)


# Calculation muscle velocity force with muscle force equation
def velocity_calculation_damped_linear(muscle_length_normalized, tendon_length_normalized, point=0):
    """
    fvm linearized around the point
    y = a*x + b
    """
    a, b = tangent_factor_calculation(point)
    return (
        ft(tendon_length_normalized) / np.cos(pennation)
        - fpas(muscle_length_normalized)
        - activation * b * fact(muscle_length_normalized)
    ) / (activation * fact(muscle_length_normalized) * a + damp)


# Calculation of tendon force obtained with the 3rd equation minus tendon force obtained with the 4th equation
def verification_equal_Ft_damped(tendon_length_normalized, muscle_length_normalized, muscle_velocity_normalized):
    return calculation_tendon_force(tendon_length_normalized) - calculation_tendonforce_7th_equation_damped_linear(
        muscle_length_normalized, muscle_velocity_normalized
    )


def tangent_factor_calculation(point):
    return [
        (d1 / np.log(10))
        * (
            (d2 * np.sqrt((d2 * point + d3) ** 2 + 1) + d2 + d3)
            / ((d2 * point + d3) * np.sqrt((d2 * point + d3) ** 2 + 1) + (d2 * point + d3) ** 2 + 1)
        ),
        (d1 / np.log(10))
        * (
            (d2 * np.sqrt((d2 * point + d3) ** 2 + 1) + d2 + d3)
            / ((d2 * point + d3) * np.sqrt((d2 * point + d3) ** 2 + 1) + (d2 * point + d3) ** 2 + 1)
        )
        * point
        + fvm(point),
    ]


# RK4 method
def RK4_integral(t0, y0, h, n, method):
    """
    t0 : initial time
    y0 : inital value of y
    h : time step
    n : number of iterations
    """

    t_values = [t0]
    muscle_length = [y0]
    tendon_length = [musculotendon_length - y0 * np.cos(pennation)]

    for _ in range(n):
        t = t_values[-1]
        y = muscle_length[-1]

        k1 = h * f(t, y, method)
        k2 = h * f(t + h / 2, y + k1 / 2, method)
        k3 = h * f(t + h / 2, y + k2 / 2, method)
        k4 = h * f(t + h, y + k3, method)

        muscle_length_new = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if np.isnan(muscle_length_new):
            return [-1]

        tendon_length_new = musculotendon_length - muscle_length_new * np.cos(pennation)
        if len(tendon_length) >= 2:
            if np.abs(tendon_length[-1] - tendon_length_new) < 1e-12:
                return t_values, muscle_length, tendon_length

        t_new = t + h
        t_values.append(t_new)
        muscle_length.append(muscle_length_new)
        tendon_length.append(tendon_length_new)

    return t_values, muscle_length, tendon_length


def f(t, y, method):
    if method == "Newton_damp":
        return func_muscle_velocity_newton(t, y)
    elif method == "Damp_linear":
        return func_muscle_velocity(t, y)
    elif method == "None_damp":
        return func_muscle_velocity_without_damp(t, y)
    elif method == "None_damp_newton":
        return func_muscle_velocity_without_damp_newton(t, y)


# Calculation tendon, muscle lengths and tendon,muscle velocities with IPOPT


def calcul_longueurs():
    # Declare the variables
    muscle_length = casadi.SX.sym("x", 1, 1)
    tendon_length = casadi.SX.sym("y", 1, 1)
    muscle_velocity = casadi.SX.sym("z", 1, 1)
    tendon_velocity = casadi.SX.sym("w", 1, 1)

    x = casadi.vertcat(muscle_length, tendon_length, muscle_velocity)
    x0 = casadi.DM([optimal_fiber_length, tendon_slack_length, muscle_velocity_guess])
    # Declare the constraints
    g1 = musculotendon_length - (tendon_length + muscle_length * np.cos(pennation))
    g2 = verification_equal_Ft(
        tendon_length / tendon_slack_length, muscle_length / optimal_fiber_length, muscle_velocity / muscle_velocity_max
    )

    g3 = muscle_velocity - (
        (
            -d3 / d2
            + (1 / d2)
            * np.sinh(
                (1 / d1) * (fvm_inverse(muscle_length / optimal_fiber_length, tendon_length / tendon_slack_length) - d4)
            )
        )
        * muscle_velocity_max
    )

    g = casadi.vertcat(1000 * g1**2, 1000 * g2**2, g3**2)

    lbx = [
        optimal_fiber_length * 0.5,
        tendon_slack_length,
        0,
    ]
    ubx = [
        optimal_fiber_length * 1.5,
        tendon_slack_length * 1.05,
        1e-4,
    ]

    solver = casadi.nlpsol(
        "solver",
        "ipopt",
        {"x": x, "f": casadi.sum1(g)},
        {
            "ipopt.tol": 1e-8,
            "ipopt.constr_viol_tol": 1e-8,
            "print_time": True,
            "ipopt.acceptable_tol": 1e-8,
            "ipopt.max_iter": 50000,
        },
    )

    sol = solver(x0=x0, ubx=ubx, lbx=lbx, ubg=0, lbg=0)

    Tendon_force_1 = calculation_tendon_force(sol["x"][1] / tendon_slack_length)
    Tendon_force_2 = calculation_muscle_force(sol["x"][0] / optimal_fiber_length, sol["x"][2] / muscle_velocity_max)
    diff = verification_equal_Ft(
        sol["x"][1] / tendon_slack_length, sol["x"][0] / optimal_fiber_length, sol["x"][2] / muscle_velocity_max
    )

    print(
        "Muscle length m:",
        sol["x"][0],
        "Tendon length m:",
        sol["x"][1],
        "\nRapports:",
        sol["x"][0] / optimal_fiber_length,
        sol["x"][1] / tendon_slack_length,
        sol["x"][2] / muscle_velocity_max,
        "\nTendon force 1 N:",
        Tendon_force_1,
        "Tendon force 2 N:",
        Tendon_force_2,
    )
    return float(Tendon_force_1)


"""
Integral method with and without damping
"""
temp = 0
temp2 = 1e-8


def velocity_calculation_damped_newton(muscle_length_normalized, tendon_length_normalized):
    global temp
    return newton(
        lambda muscle_velocity_initial: fvm(muscle_velocity_initial / muscle_velocity_max)
        - (
            ft(tendon_length_normalized) / np.cos(pennation)
            - fpas(muscle_length_normalized)
            - damp * muscle_velocity_initial / muscle_velocity_max
        )
        / (activation * fact(muscle_length_normalized)),
        temp,
        tol=1e-8,
        rtol=1e-8,
    )


def velocity_calculation_without_damp_newton(muscle_length_normalized, tendon_length_normalized):
    global temp2
    try:
        value = newton(
            lambda muscle_velocity_initial: fvm(muscle_velocity_initial / muscle_velocity_max)
            * activation
            * fact(muscle_length_normalized)
            + fpas(muscle_length_normalized)
            - (ft(tendon_length_normalized) / np.cos(pennation)),
            temp2,
            tol=1e-8,
            rtol=1e-8,
        )
        return value
    except:
        return -1


def func_muscle_velocity_newton(t, y):
    return (
        velocity_calculation_damped_newton(
            y / optimal_fiber_length, (musculotendon_length - y * np.cos(pennation)) / tendon_slack_length
        )
        * muscle_velocity_max
    )


def func_muscle_velocity(t, y):
    return (
        velocity_calculation_damped_linear(
            y / optimal_fiber_length, ((musculotendon_length - y * np.cos(pennation)) / tendon_slack_length)
        )
        * muscle_velocity_max
    )


def func_muscle_velocity_without_damp(t, y):
    return (
        -d3 / d2
        + (1 / d2)
        * np.sinh(
            (1 / d1)
            * (
                fvm_inverse(
                    y / optimal_fiber_length, (musculotendon_length - y * np.cos(pennation)) / tendon_slack_length
                )
                - d4
            )
        )
    ) * muscle_velocity_max


def func_muscle_velocity_without_damp_newton(t, y):
    return (
        velocity_calculation_without_damp_newton(
            y / optimal_fiber_length, (musculotendon_length - y * np.cos(pennation)) / tendon_slack_length
        )
        * muscle_velocity_max
    )


def integral_method_with_damp_linear():

    x0 = (musculotendon_length - tendon_slack_length) / np.cos(pennation)
    x = [x0, 0, 0]
    tab = RK4_integral(0, x[0], 1e-3, 10000, "Damp_linear")  # 1400 2000
    x[0] = tab[1][-1]
    x[1] = tab[2][-1]
    x[2] = (
        velocity_calculation_damped_linear(x[0] / optimal_fiber_length, x[1] / tendon_slack_length)
        * muscle_velocity_max,
        x[2],
    )
    print(
        "\nTendon Force:",
        calculation_tendon_force(x[1] / tendon_slack_length),
        "\nMuscle Forcfe:",
        calculation_muscle_force(x[0] / optimal_fiber_length, x[2][-1] / muscle_velocity_max),
        "\nMuscle Length:",
        x[0],
        "\nTendon Length:",
        x[1],
        "\nMusculoTendon Length Calculated:",
        x[1] + x[0] * np.cos(pennation),
        "\nMusculoTendon Length Real:",
        musculotendon_length,
    )
    return tab, calculation_tendon_force(x[1] / tendon_slack_length)


def integral_method_without_damp():

    x0 = (musculotendon_length - tendon_slack_length) / np.cos(pennation)
    x = [x0, 0, 0]
    tab = RK4_integral(0, x[0], 1e-3, 10000, "None_damp")  # 1300 2000
    if len(tab) == 1:
        if tab[0] == -1:
            return -1
            # raise ValueError("Error Calculation RK4 integral not a number. Activation may be too small.")
    x[0] = tab[1][-1]
    x[1] = tab[2][-1]
    x[2] = (
        -d3 / d2
        + (1 / d2) * np.sinh((1 / d1) * (fvm_inverse(x[0] / optimal_fiber_length, x[1] / tendon_slack_length) - d4))
    ) * muscle_velocity_max
    print(
        "\nTendon Force:",
        calculation_tendon_force(x[1] / tendon_slack_length),
        "\nMuscle Forcfe:",
        calculation_muscle_force(x[0] / optimal_fiber_length, x[2] / muscle_velocity_max),
        "\nMuscle Length:",
        x[0],
        "\nTendon Length:",
        x[1],
        "\nMusculoTendon Length Calculated:",
        x[1] + x[0] * np.cos(pennation),
        "\nMusculoTendon Length Real:",
        musculotendon_length,
    )
    return tab, calculation_tendon_force(x[1] / tendon_slack_length)


def integral_method_without_damp_newton():
    global temp2
    x0 = (musculotendon_length - tendon_slack_length) / np.cos(pennation)
    x = [x0, 0, 0]
    temp = x[2]
    tab = RK4_integral(
        0, x[0], 1e-3, 10000, "None_damp_newton"
    )  # 2200 pour une erreur < 1%, 6500 pour que tout converge 2500
    x[0] = tab[1][-1]
    x[1] = tab[2][-1]
    x[2] = newton(
        lambda y: fvm(y / muscle_velocity_max) * activation * fact(x[0] / optimal_fiber_length)
        + fpas(x[0] / optimal_fiber_length)
        - (ft(x[1] / tendon_slack_length) / np.cos(pennation)),
        temp2,
        tol=1e-8,
        rtol=1e-8,
    )
    print(
        "\nTendon Force:",
        calculation_tendon_force(x[1] / tendon_slack_length),
        "\nMuscle Forcfe:",
        calculation_muscle_force(x[0] / optimal_fiber_length, x[2] / muscle_velocity_max),
        "\nMuscle Length:",
        x[0],
        "\nTendon Length:",
        x[1],
        "\nMusculoTendon Length Calculated:",
        x[1] + x[0] * np.cos(pennation),
        "\nMusculoTendon Length Real:",
        musculotendon_length,
    )
    return tab, calculation_tendon_force(x[1] / tendon_slack_length)


def integral_method_with_damp_newton():
    global temp
    x0 = (musculotendon_length - tendon_slack_length) / np.cos(pennation)
    x = [x0, 0, 0]
    temp = x[2]
    tab = RK4_integral(
        0, x[0], 1e-3, 10000, "Newton_damp"
    )  # 2200 pour une erreur < 1%, 6500 pour que tout converge 2500
    x[0] = tab[1][-1]
    x[1] = tab[2][-1]
    x[2] = newton(
        lambda y: fvm(y / muscle_velocity_max)
        - (
            ft(x[1] / tendon_slack_length) / np.cos(pennation)
            - fpas(x[0] / optimal_fiber_length)
            - damp * y / muscle_velocity_max
        )
        / (activation * fact(x[0] / optimal_fiber_length)),
        temp,
        rtol=1e-8,
        tol=1e-8,
    )
    print(
        "\nTendon Force:",
        calculation_tendon_force(x[1] / tendon_slack_length),
        "\nMuscle Forcfe:",
        calculation_muscle_force(x[0] / optimal_fiber_length, x[2] / muscle_velocity_max),
        "\nMuscle Length:",
        x[0],
        "\nTendon Length:",
        x[1],
        "\nMusculoTendon Length Calculated:",
        x[1] + x[0] * np.cos(pennation),
        "\nMusculoTendon Length Real:",
        musculotendon_length,
    )
    return tab, calculation_tendon_force(x[1] / tendon_slack_length)


def main():

    global activation
    global optimal_fiber_length
    global tendon_slack_length
    global maximalIsometricForce
    global pennation
    global musculotendon_length
    global musculotendon_velocity

    Muscle = [
        {
            "Muscle": "DELT3",
            "musculotendonLength": 0.18239173838725709,
            "musculotendonVelocity": -0.009259459075135049,
            "optimalLength": 0.1228,
            "maximalForce": 944.7,
            "tendonSlackLength": 0.0975,
            "pennationAngle": 0.314159265359,
        },
        {
            "Muscle": "DELT1",
            "musculotendonLength": 0.19469108075825378,
            "musculotendonVelocity": 0.0021493615729395107,
            "optimalLength": 0.1752,
            "maximalForce": 556.8,
            "tendonSlackLength": 0.0313,
            "pennationAngle": 0.383972435439,
        },
        {
            "Muscle": "BICshort",
            "musculotendonLength": 0.2953593031089707,
            "musculotendonVelocity": 0.005948975815775617,
            "optimalLength": 0.13210,
            "maximalForce": 435.560,
            "tendonSlackLength": 0.19230,
            "pennationAngle": 0.000,
        },
        {
            "Muscle": "TMIN",
            "musculotendonLength": 0.1273211827764525,
            "musculotendonVelocity": -0.04281209104530441,
            "optimalLength": 0.0453,
            "maximalForce": 605.4,
            "tendonSlackLength": 0.1038,
            "pennationAngle": 0.418879020479,
        },
        {
            "Muscle": "CORB",
            "musculotendonLength": 0.12503849135582248,
            "musculotendonVelocity": -0.003372718839228111,
            "optimalLength": 0.0832,
            "maximalForce": 306.9,
            "tendonSlackLength": 0.0615,
            "pennationAngle": 0.0,
        },
        {
            "Muscle": "TRIlong",
            "musculotendonLength": 0.321200616825438,
            "musculotendonVelocity": -0.0056587952943956115,
            "optimalLength": 0.13400,
            "maximalForce": 798.520,
            "tendonSlackLength": 0.14300,
            "pennationAngle": 0.2094,
        },
        {
            "Muscle": "BIClong",
            "musculotendonLength": 0.3786307131845775,
            "musculotendonVelocity": 0.007027430619001448,
            "optimalLength": 0.11570,
            "maximalForce": 624.300,
            "tendonSlackLength": 0.27230,
            "pennationAngle": 0.000,
        },
    ]

    damp_linear = []
    damp_newton = []
    without_damp = []
    x = []

    lt_damp_linear = []
    lt_damp_newton = []
    lt_without_damp = []

    lm_damp_linear = []
    lm_damp_newton = []
    lm_without_damp = []

    time_damp_linear = []
    time_damp_newton = []
    time_without_damp = []
    muscle_name = []

    tendon_force_without_damp = []
    tendon_force_damp_linear = []
    tendon_force_damp_newton = []
    indice = 3
    legend = False

    fig, axs = plt.subplots(5, 2, figsize=(15, 10), layout="constrained")

    """To plot the length of the muscle and tendon"""
    """    
    for i in range(1, 11):
        activation = i * 1e-2
        muscle_current = Muscle[indice]["Muscle"]
        optimal_fiber_length = Muscle[indice]["optimalLength"]
        tendon_slack_length = Muscle[indice]["tendonSlackLength"]
        musculotendon_length = Muscle[indice]["musculotendonLength"]
        musculotendon_velocity = Muscle[indice]["musculotendonVelocity"]
        maximalIsometricForce = Muscle[indice]["maximalForce"]
        pennation = Muscle[indice]["pennationAngle"]

        calcul_longueurs()
        current_without_damp = integral_method_without_damp()
        current_damp_linear = integral_method_with_damp_linear()
        current_damp_newton = integral_method_with_damp_newton()

        if current_without_damp != -1:
            lm_without_damp = current_without_damp[0][1]
            lt_without_damp = current_without_damp[0][2]
        else:
            lm_without_damp = np.zeros(len(current_damp_linear))
            lt_without_damp = np.zeros(len(current_damp_linear))

        lm_damp_newton = current_damp_newton[0][1]
        lm_damp_linear = current_damp_linear[0][1]

        lt_damp_newton = current_damp_newton[0][2]
        lt_damp_linear = current_damp_linear[0][2]

        for j in range(len(lm_without_damp)):
            lm_without_damp[j] = lm_without_damp[j] / optimal_fiber_length

        for j in range(len(lm_damp_linear)):
            lm_damp_linear[j] = lm_damp_linear[j] / optimal_fiber_length

        for j in range(len(lm_damp_newton)):
            lm_damp_newton[j] = lm_damp_newton[j] / optimal_fiber_length

        for j in range(len(lt_without_damp)):
            lt_without_damp[j] = lt_without_damp[j] / tendon_slack_length

        for j in range(len(lt_damp_linear)):
            lt_damp_linear[j] = lt_damp_linear[j] / tendon_slack_length

        for j in range(len(lt_damp_newton)):
            lt_damp_newton[j] = lt_damp_newton[j] / tendon_slack_length

        axs.flat[i - 1].plot(current_damp_newton[0][0], lm_damp_newton, label="With Damp Newton")
        axs.flat[i - 1].plot(current_damp_linear[0][0], lm_damp_linear, label="With Damp Linear")
        if current_without_damp != -1:
            axs.flat[i - 1].plot(current_without_damp[0][0], lm_without_damp, label="Without Damp")
        axs.flat[i - 1].set_title(f"Activation : {activation}")
        axs.flat[i - 1].set_xlabel("Time in sec")
        axs.flat[i - 1].set_ylabel("Muscle length")
        if current_without_damp != -1 and legend == False or legend == False and i == 10:
            axs.flat[i - 1].legend(loc="upper right")
            legend = True
        axs.flat[i - 1].grid(True)

        axs.flat[i - 1].plot(current_damp_newton[0][0], lt_damp_newton, label="With Damp Newton")
        axs.flat[i - 1].plot(current_damp_linear[0][0], lt_damp_linear, label="With Damp Linear")
        if current_without_damp != -1:
            axs.flat[i - 1].plot(current_without_damp[0][0], lt_without_damp, label="Without Damp")
        axs.flat[i - 1].set_title(f"Activation : {activation}")
        axs.flat[i - 1].set_xlabel("Time in sec")
        axs.flat[i - 1].set_ylabel("Tendon length")
        if current_without_damp != -1 and legend == False or legend == False and i == 10:
            axs.flat[i - 1].legend(loc="lower right")
            legend = True
        axs.flat[i - 1].grid(True)

    plt.tight_layout()
    plt.suptitle(f"Muscle Length normalized for {muscle_current} and activation between 0.01 and 0.1.")
    # plt.suptitle(f"Tendon Length normalized for {muscle_current} and activation between 0.01 and 0.1.")
    plt.show()
    """
    """To plot the time of CV of each muscle for a value of activation"""
    """activation = 1
    for muscle in Muscle:
        muscle_current = muscle["Muscle"]
        optimal_fiber_length = muscle["optimalLength"]
        tendon_slack_length = muscle["tendonSlackLength"]
        musculotendon_length = muscle["musculotendonLength"]
        musculotendon_velocity = muscle["musculotendonVelocity"]
        maximalIsometricForce = muscle["maximalForce"]
        pennation = muscle["pennationAngle"]
        muscle_name += [muscle_current]

        current_benchmark = calcul_longueurs()
        current_without_damp = integral_method_without_damp()
        current_damp_linear = integral_method_with_damp_linear()
        current_damp_newton = integral_method_with_damp_newton()

        tendon_force_damp_linear.append(100 - current_damp_linear[1] * 100 / current_benchmark)
        tendon_force_damp_newton.append(100 - current_damp_newton[1] * 100 / current_benchmark)

        if current_without_damp != -1:
            time_without_damp += [current_without_damp[0][0][-1]]
            tendon_force_without_damp.append(100 - current_without_damp[1] * 100 / current_benchmark)
        else:
            time_without_damp += [-0.1]
            tendon_force_without_damp.append(-0.1)
        time_damp_linear += [current_damp_linear[0][0][-1]]
        time_damp_newton += [current_damp_newton[0][0][-1]]

    plt.subplot(1, 2, 1)
    plt.plot(muscle_name, tendon_force_without_damp, label="Tendon force Without Damp", color="red", marker="p")
    plt.plot(muscle_name, tendon_force_damp_linear, label="Tendon force With Damp Linear", color="green", marker="p")
    plt.plot(muscle_name, tendon_force_damp_newton, label="Tendon force With Damp Newton", color="black", marker="p")
    plt.legend()
    plt.title("Tendon Force")
    plt.grid(True)"""

    """plt.subplot(1, 2, 2)
    plt.plot(muscle_name, time_without_damp, label="Time CV Without Damp", color="red", marker="p")
    plt.plot(muscle_name, time_damp_linear, label="Time CV With Damp Linear", color="green", marker="p")
    plt.plot(muscle_name, time_damp_newton, label="Time CV With Damp Newton", color="black", marker="p")
    plt.legend()
    plt.title("Time")
    plt.grid(True)
    plt.show()"""

    """To plot the time of Convergence for all the value of activation or the Tendon Force Error compared to IPOPT value"""

    for i in range(1, 11):
        time_damp_linear = []
        time_damp_newton = []
        time_without_damp = []

        tendon_force_without_damp = []
        tendon_force_damp_linear = []
        tendon_force_damp_newton = []

        muscle_name = []
        activation = i * 1e-1
        x.append(activation)
        for muscle in Muscle:
            muscle_current = muscle["Muscle"]
            optimal_fiber_length = muscle["optimalLength"]
            tendon_slack_length = muscle["tendonSlackLength"]
            musculotendon_length = muscle["musculotendonLength"]
            musculotendon_velocity = muscle["musculotendonVelocity"]
            maximalIsometricForce = muscle["maximalForce"]
            pennation = muscle["pennationAngle"]
            muscle_name += [muscle_current]

            current_benchmark = calcul_longueurs()
            current_without_damp = integral_method_without_damp_newton()
            current_damp_linear = integral_method_with_damp_linear()
            current_damp_newton = integral_method_with_damp_newton()

            tendon_force_damp_linear.append(100 - current_damp_linear[1] * 100 / current_benchmark)
            tendon_force_damp_newton.append(100 - current_damp_newton[1] * 100 / current_benchmark)

            if current_without_damp != -1:
                time_without_damp += [current_without_damp[0][0][-1]]
                tendon_force_without_damp.append(100 - current_without_damp[1] * 100 / current_benchmark)
                lm_without_damp = []
                lt_without_damp = []
            else:
                time_without_damp += [-0.1]
                tendon_force_without_damp.append(-0.1)
                lm_without_damp = []
                lt_without_damp = []

            time_damp_linear += [current_damp_linear[0][0][-1]]
            time_damp_newton += [current_damp_newton[0][0][-1]]
            lm_damp_linear = []
            lt_damp_newton = []

            lt_damp_linear = []
            lt_damp_newton = []

        axs.flat[i - 1].plot(
            muscle_name, time_damp_linear, label="Millard With Damp Method Linear", color="red", marker="p"
        )
        axs.flat[i - 1].plot(
            muscle_name, time_damp_newton, label="Millard With Damp Method Newton", color="green", marker="p"
        )
        axs.flat[i - 1].plot(muscle_name, time_without_damp, label="De Groote Without Damp", color="black", marker="p")
        axs.flat[i - 1].set_title(f"Activation:{activation}")
        axs.flat[i - 1].set_xlabel("Muscle")
        axs.flat[i - 1].set_ylabel("Time in sec")
        if i == 1:
            axs.flat[i - 1].legend(loc="best")
        axs.flat[i - 1].grid(True)

    plt.tight_layout()
    # plt.suptitle("Tendon force Error (compared to IPOPT Value) with activation between 0.1 and 1.")
    plt.suptitle("Time of CV with activation between 0.1 and 1.")
    plt.show()


if __name__ == "__main__":
    main()

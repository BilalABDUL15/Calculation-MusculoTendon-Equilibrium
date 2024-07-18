from typing import Callable
import casadi
import numpy as np
from shoulder import ModelBiorbd, ControlsTypes, MuscleParameter
import timeit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
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
musculotendon_length = 0.1378300166960712
optimal_fiber_length = 0.1323  # optimal muscle fiber length
tendon_slack_length = 0.0337  # tendon slack length


# Other parameters

maximalIsometricForce = 864.6
pennation = 0.322885911619
activation = 1  # Muscle Activation


# Velocity

musculotendon_velocity = -0.024345955973862205

"""Doc Millard for velocity max : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3705831/pdf/bio_135_2_021005.pdf"""
muscle_velocity_max = 5.32 * optimal_fiber_length
muscle_velocity_guess = musculotendon_velocity * 0.5
tendon_velocity_guess = musculotendon_velocity - muscle_velocity_guess / np.cos(pennation)


"""Supplementary materials Doc de Groote : https://static-content.springer.com/esm/art%3A10.1007%2Fs10439-016-1591-9/MediaObjects/10439_2016_1591_MOESM1_ESM.pdf"""
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
k1 = 1
k2 = 2.2


IPOPT_output = []
nb_iter = 0

iterations = []


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
def calculation_tendonforce_3rd_equation(tendon_length_normalized):
    return maximalIsometricForce * ft(tendon_length_normalized)


# Definition Ft with the 7th equation of De Groote : Ft = Fm * cos(pennation)
def calculation_tendonforce_7th_equation(muscle_length_normalized, muscle_velocity_normalized):
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
    return calculation_tendonforce_3rd_equation(tendon_length_normalized) - calculation_tendonforce_7th_equation(
        muscle_length_normalized, muscle_velocity_normalized
    )


"""
Damped version : Millard and Nabipour Paper 
Nabipour : https://www.biorxiv.org/content/10.1101/2024.05.14.594110v1.full.pdf
Millard : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3705831/pdf/bio_135_2_021005.pdf
"""


# Muscle force velocity equation = fvce
def fvm_damped(muscle_velocity_normalized):
    return k1 + k2 * muscle_velocity_normalized


def fdamp(muscle_velocity_normalized):
    return damp * muscle_velocity_normalized


# Definition Ft with the 7th equation of De Groote : Ft = Fm * cos(pennation)
def calculation_tendonforce_7th_equation_damped(muscle_length_normalized, muscle_velocity_normalized):
    Fm = maximalIsometricForce * (
        fpas(muscle_length_normalized)
        + fdamp(muscle_velocity_normalized)
        + activation * fvm_damped(muscle_velocity_normalized) * fact(muscle_length_normalized)
    )
    return Fm * np.cos(pennation)


# Calculation muscle velocity force with muscle force equation
def velocity_calculation_damped(muscle_length_normalized, tendon_length_normalized):
    """
    fvm linearized around 0
    Source: Nabipour Paper
    https://www.biorxiv.org/content/10.1101/2024.05.14.594110v1.full.pdf
    """
    return (
        ft(tendon_length_normalized) / np.cos(pennation)
        - fpas(muscle_length_normalized)
        - activation * k1 * fact(muscle_length_normalized)
    ) / (activation * fact(muscle_length_normalized) * k2 + damp)


# Calculation of tendon force obtained with the 3rd equation minus tendon force obtained with the 4th equation
def verification_equal_Ft_damped(tendon_length_normalized, muscle_length_normalized, muscle_velocity_normalized):
    return calculation_tendonforce_3rd_equation(tendon_length_normalized) - calculation_tendonforce_7th_equation_damped(
        muscle_length_normalized, muscle_velocity_normalized
    )


# Calculation tendon, muscle lengths and tendon, muscle velocity with Rootfinder
def Rootfinder_method(muscle_velocity_guess):

    # Declare the variables
    muscle_length = casadi.SX.sym("x", 1, 1)
    tendon_length = casadi.SX.sym("y", 1, 1)
    muscle_velocity = casadi.SX.sym("z", 1, 1)

    x = casadi.vertcat(muscle_length, tendon_length, muscle_velocity)
    x0 = casadi.DM(
        [
            ((musculotendon_length * 1 / 3) / np.cos(pennation)),
            musculotendon_length * 2 / 3,
            muscle_velocity_guess,
        ]
    )
    # Declare the constraints
    g1 = musculotendon_length - (tendon_length + muscle_length * np.cos(pennation))
    g2 = verification_equal_Ft(
        tendon_length / tendon_slack_length, muscle_length / optimal_fiber_length, muscle_velocity / muscle_velocity_max
    )

    g3 = (
        muscle_velocity
        - (
            -d3 / d2
            + (1 / d2)
            * np.sinh(
                (1 / d1) * (fvm_inverse(muscle_length / optimal_fiber_length, tendon_length / tendon_slack_length) - d4)
            )
        )
        * muscle_velocity_max
    )

    g = casadi.vertcat(g1**2, g2**2, g3**2)
    solver = casadi.rootfinder(
        "solver",
        "newton",
        {"x": x, "g": g},
        {
            "abstol": 1e-4,
            "print_time": True,
        },
    )
    sol = solver(x0=x0)
    Tendon_force_1 = calculation_tendonforce_3rd_equation(sol["x"][1] / tendon_slack_length)
    Tendon_force_2 = calculation_tendonforce_7th_equation(
        sol["x"][0] / optimal_fiber_length, sol["x"][2] / muscle_velocity_max
    )

    print(
        "Tendon length m:",
        sol["x"][1],
        "Muscle length m:",
        sol["x"][0],
        "Velocity muscle m/s:",
        sol["x"][2],
        (
            -d3 / d2
            + (1 / d2)
            * np.sinh(
                (1 / d1) * (fvm_inverse(sol["x"][0] / optimal_fiber_length, sol["x"][1] / tendon_slack_length) - d4)
            )
        )
        * muscle_velocity_max,
        "\nRapports:",
        sol["x"][2] / muscle_velocity_max,
        sol["x"][1] / tendon_slack_length,
        sol["x"][0] / optimal_fiber_length,
        "Tendon velocity m/s",
        musculotendon_velocity - sol["x"][2] / np.cos(pennation),
        "Tendon force 3 N:",
        Tendon_force_1,
        "Tendon force 7 N:",
        Tendon_force_2,
        "Musculotendon length",
        sol["x"][1] + sol["x"][0] * np.cos(pennation),
        "Musculotendon_velocity",
    )
    return float(sol["x"][2])


#######################################################################


def obj_func(x):

    g1 = musculotendon_length - (x[1] + x[0] * np.cos(pennation))
    g2 = verification_equal_Ft(x[1] / tendon_slack_length, x[0] / optimal_fiber_length, x[2] / muscle_velocity_max)

    g3 = (
        x[2]
        - (
            -d3 / d2
            + (1 / d2) * np.sinh((1 / d1) * (fvm_inverse(x[0] / optimal_fiber_length, x[1] / tendon_slack_length) - d4))
        )
        * muscle_velocity_max
    )
    return casadi.vertcat(g1**2, g2**2, g3**2)


def newton_method(muscle_velocity_guess):

    # Declare the variables
    muscle_length = casadi.SX.sym("x", 1, 1)
    tendon_length = casadi.SX.sym("y", 1, 1)
    muscle_velocity = casadi.SX.sym("z", 1, 1)

    x = casadi.vertcat(muscle_length, tendon_length, muscle_velocity)
    tab_tendon = []
    tab_muscle = []
    tab_velocity = []
    tab_diff = []
    tab_tendon_force_3 = []
    tab_tendon_force_7 = []
    x0 = casadi.DM(
        [
            optimal_fiber_length,
            tendon_slack_length,
            muscle_velocity_guess,
            tendon_velocity_guess,
        ]
    )
    f_x = obj_func(x)

    df_x = casadi.jacobian(f_x, x)

    # Convertir la fonction et sa dérivée en des fonctions CasADi
    f_func = casadi.Function("f_func", [x], [f_x])
    df_func = casadi.Function("df_func", [x], [df_x])

    x_opt = x0
    cpt = 0
    dx = np.inf
    nb_bouncing = 0
    calcul_sens = 1
    previous = x_opt[3]
    current = 0
    previous = 0
    jump = 10 ** (-2)
    # First loop
    while True:
        dx = casadi.inv(df_func(x_opt[0:3])) @ f_func(x_opt[0:3])
        current = float(
            verification_equal_Ft(
                x_opt[1] / tendon_slack_length, x_opt[0] / optimal_fiber_length, x_opt[2] / muscle_velocity_max
            )
        )
        if np.abs(np.abs(x_opt[2]) - np.abs(muscle_velocity_max)) < 1e-8 or np.abs(previous) < np.abs(current):
            calcul_sens = -calcul_sens

        for i in range(3):
            x_opt[i] += dx[i] * jump * calcul_sens
        x_opt[3] = musculotendon_velocity - x_opt[2] / np.cos(pennation)
        cpt += 1
        diff = verification_equal_Ft(
            x_opt[1] / tendon_slack_length, x_opt[0] / optimal_fiber_length, x_opt[2] / muscle_velocity_max
        )
        if cpt % 100 == 0:
            tab_diff.append(float(diff))
        print(
            verification_equal_Ft(
                x_opt[1] / tendon_slack_length, x_opt[0] / optimal_fiber_length, x_opt[2] / muscle_velocity_max
            ),
            "Tendon 7",
            calculation_tendonforce_7th_equation(x_opt[0] / optimal_fiber_length, x_opt[2] / muscle_velocity_max),
            "Tendon 3",
            calculation_tendonforce_3rd_equation(x_opt[1] / tendon_slack_length),
            x_opt,
        )

        previous = current
        if np.abs(diff) < 1e-1 or np.isnan(diff):
            break
    nb_bouncing = -3
    cpt = 0
    jump = 10 ** (nb_bouncing)
    while True:
        dx = casadi.inv(df_func(x_opt[0:3])) @ f_func(x_opt[0:3])
        current = float(
            verification_equal_Ft(
                x_opt[1] / tendon_slack_length, x_opt[0] / optimal_fiber_length, x_opt[2] / muscle_velocity_max
            )
        )
        if np.abs(np.abs(x_opt[2]) - np.abs(muscle_velocity_max)) < 1e-8 or np.abs(previous) < np.abs(current):
            calcul_sens = -calcul_sens
            nb_bouncing -= 1
            jump = 10 ** (nb_bouncing)
        previous = current

        for i in range(3):
            x_opt[i] += dx[i] * jump * calcul_sens
        x_opt[3] = musculotendon_velocity - x_opt[2] / np.cos(pennation)

        cpt += 1
        diff = verification_equal_Ft(
            x_opt[1] / tendon_slack_length, x_opt[0] / optimal_fiber_length, x_opt[2] / muscle_velocity_max
        )
        if cpt % 100 == 0:
            tab_diff.append(float(diff))
        print(
            verification_equal_Ft(
                x_opt[1] / tendon_slack_length, x_opt[0] / optimal_fiber_length, x_opt[2] / muscle_velocity_max
            ),
            "Tendon 7",
            calculation_tendonforce_7th_equation(x_opt[0] / optimal_fiber_length, x_opt[2] / muscle_velocity_max),
            "Tendon 3",
            calculation_tendonforce_3rd_equation(x_opt[1] / tendon_slack_length),
            x_opt,
        )

        if np.abs(diff) < 1e-2 or np.isnan(diff):
            break
    # x_opt[3] = musculotendon_velocity - x_opt[2] / np.cos(pennation)
    print(
        "\nTendon length m:",
        x_opt[1],
        "Muscle length m:",
        x_opt[0],
        "Velocity muscle m/s:",
        x_opt[2],
        (
            -d3 / d2
            + (1 / d2)
            * np.sinh((1 / d1) * (fvm_inverse(x_opt[0] / optimal_fiber_length, x_opt[1] / tendon_slack_length) - d4))
        )
        * muscle_velocity_max,
        "Tendon velocity m/s:",
        x_opt[3],
        "\nRapports:",
        x_opt[2] / muscle_velocity_max,
        x_opt[1] / tendon_slack_length,
        x_opt[0] / optimal_fiber_length,
        "\nTendon force 3 N:",
        calculation_tendonforce_3rd_equation(x_opt[1] / tendon_slack_length),
        "Tendon force 7 N:",
        calculation_tendonforce_7th_equation(x_opt[0] / optimal_fiber_length, x_opt[2] / muscle_velocity_max),
        "\nMusculotendon length",
        x_opt[1] + x_opt[0] * np.cos(pennation),
        "Musculotendon_velocity",
        x_opt[3] + x_opt[2] / np.cos(pennation),
    )
    return [tab_tendon, tab_muscle, tab_velocity, tab_diff, tab_tendon_force_3, tab_tendon_force_7, x_opt[2]]


# Calculation tendon, muscle lengths and tendon,muscle velocities with IPOPT


def calcul_longueurs():
    # Declare the variables
    muscle_length = casadi.SX.sym("x", 1, 1)
    tendon_length = casadi.SX.sym("y", 1, 1)
    muscle_velocity = casadi.SX.sym("z", 1, 1)
    tendon_velocity = casadi.SX.sym("w", 1, 1)

    x = casadi.vertcat(muscle_length, tendon_length, muscle_velocity, tendon_velocity)
    x0 = casadi.DM([optimal_fiber_length, tendon_slack_length, muscle_velocity_guess, tendon_velocity_guess])
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
    g4 = musculotendon_velocity - (tendon_velocity + muscle_velocity / np.cos(pennation))

    g = casadi.vertcat(1000 * g1**2, 1000 * g2**2, g3**2, g4**2)

    lbx = [
        optimal_fiber_length * 0.5,
        tendon_slack_length,
        0,  # -muscle_velocity_max,
        (musculotendon_length - muscle_velocity_max / np.cos(pennation)),
    ]
    ubx = [
        optimal_fiber_length * 1.5,
        tendon_slack_length * 1.05,
        1e-4,  # muscle_velocity_max,
        (musculotendon_length + muscle_velocity_max / np.cos(pennation)),
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

    Tendon_force_1 = calculation_tendonforce_3rd_equation(sol["x"][1] / tendon_slack_length)
    Tendon_force_2 = calculation_tendonforce_7th_equation(
        sol["x"][0] / optimal_fiber_length, sol["x"][2] / muscle_velocity_max
    )
    diff = verification_equal_Ft(
        sol["x"][1] / tendon_slack_length, sol["x"][0] / optimal_fiber_length, sol["x"][2] / muscle_velocity_max
    )

    print(
        "Muscle length m:",
        sol["x"][0],
        "Tendon length m:",
        sol["x"][1],
        "\nVelocity muscle m/s:",
        sol["x"][2],
        "Tendon velocity m/s",
        sol["x"][3],
        "\nRapports:",
        sol["x"][0] / optimal_fiber_length,
        sol["x"][1] / tendon_slack_length,
        sol["x"][2] / muscle_velocity_max,
        sol["x"][3] / tendon_slack_length,
        "\nTendon force 3 N:",
        Tendon_force_1,
        "Tendon force 7 N:",
        Tendon_force_2,
        diff,
        "\nMusculotendon length",
        sol["x"][1] + sol["x"][0] * np.cos(pennation),
        "Musculotendon_velocity",
        sol["x"][3] + sol["x"][2] / np.cos(pennation),
        "\nObj Func value:",
        func_obj_minimize(sol["x"]),
    )
    return float(Tendon_force_1)


def func_obj_minimize(x):
    return (
        1000
        * (
            (
                verification_equal_Ft(
                    x[1] / tendon_slack_length,
                    x[0] / optimal_fiber_length,
                    x[2] / muscle_velocity_max,
                )
            )
            ** 2
            + (musculotendon_length - (x[1] + x[0] * np.cos(pennation))) ** 2
        )
        + (
            x[2]
            - (
                (
                    -d3 / d2
                    + (1 / d2)
                    * np.sinh((1 / d1) * (fvm_inverse(x[0] / optimal_fiber_length, x[1] / tendon_slack_length) - d4))
                )
                * muscle_velocity_max
            )
        )
        ** 2
        + (musculotendon_velocity - (x[3] + (x[2] / np.cos(pennation)))) ** 2
    )


cpt = 0


def callback(x):
    global cpt
    global iterations
    iterations.append(x)
    cpt += 1


def minimize_func():
    x0 = [optimal_fiber_length, tendon_slack_length, muscle_velocity_guess, tendon_velocity_guess]

    lbx = [
        optimal_fiber_length * 0.5,
        tendon_slack_length,
        -1e-4,  # -muscle_velocity_max,
        (musculotendon_length - muscle_velocity_max / np.cos(pennation)),
    ]
    ubx = [
        optimal_fiber_length * 1.5,
        tendon_slack_length * 1.05,
        1e-4,  # muscle_velocity_max,
        (musculotendon_length + muscle_velocity_max / np.cos(pennation)),
    ]

    bounds = ((lbx[0], ubx[0]), (lbx[1], ubx[1]), (lbx[2], ubx[2]), (lbx[3], ubx[3]))
    options_powell = {"maxiter": 1000, "xtol": 1e-8, "ftol": 1e-8}
    options_nelder = {"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8}
    sol = minimize(func_obj_minimize, x0=x0, method="Powell", callback=callback, options=options_powell, bounds=bounds)

    while cpt < 10000:
        sol = minimize(
            func_obj_minimize,
            x0=sol["x"],
            method="Nelder-Mead",
            callback=callback,
            options=options_nelder,
            bounds=bounds,
        )

    print(
        "Muscle length:",
        sol["x"][0],
        "Tendon length:",
        sol["x"][1],
        "Muscle velocity:",
        sol["x"][2],
        "Tendon velocity:",
        sol["x"][3],
        "\nDiffs:",
        verification_equal_Ft(
            sol["x"][1] / tendon_slack_length,
            sol["x"][0] / optimal_fiber_length,
            sol["x"][2] / muscle_velocity_max,
        ),
        musculotendon_velocity,
        (sol["x"][3] + sol["x"][2] / np.cos(pennation)),
        musculotendon_length,
        (sol["x"][1] + sol["x"][0] * np.cos(pennation)),
        "\nTendon Force calculation:",
        calculation_tendonforce_3rd_equation(sol["x"][1] / tendon_slack_length),
        calculation_tendonforce_7th_equation(sol["x"][0] / optimal_fiber_length, sol["x"][2] / muscle_velocity_max),
        "\nRapports:",
        sol["x"][0] / optimal_fiber_length,
        sol["x"][1] / tendon_slack_length,
        sol["x"][2] / muscle_velocity_max,
        sol["x"][3] / tendon_slack_length,
        "\nObj Func value:",
        func_obj_minimize(sol["x"]),
    )

    return float(calculation_tendonforce_3rd_equation(sol["x"][1] / tendon_slack_length))


"""
Integral method with and without damping
"""
temp = 0


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


def func_muscle_velocity_newton(t, y):
    tendon_length = musculotendon_length - y[0] * np.cos(pennation)
    return [
        velocity_calculation_damped_newton(y[0] / optimal_fiber_length, tendon_length / tendon_slack_length)
        * muscle_velocity_max
    ]


def func_muscle_velocity(t, y):
    tendon_length = musculotendon_length - y[0] * np.cos(pennation)
    return [
        velocity_calculation_damped(y[0] / optimal_fiber_length, (tendon_length / tendon_slack_length))
        * muscle_velocity_max
    ]


def func_muscle_velocity_without_damp(t, y):
    tendon_length = musculotendon_length - y[0] * np.cos(pennation)
    return [
        (
            -d3 / d2
            + (1 / d2)
            * np.sinh((1 / d1) * (fvm_inverse(y[0] / optimal_fiber_length, tendon_length / tendon_slack_length) - d4))
        )
        * muscle_velocity_max
    ]


def integral_method_with_damp():

    x0 = [optimal_fiber_length, tendon_slack_length * 1.02, muscle_velocity_max * 0.5]
    tab = []
    lm = []
    vm = []
    lt = []

    x = x0
    cpt = 0
    t_span = [cpt, cpt + 1e-4]
    sol = solve_ivp(
        func_muscle_velocity,
        t_span=t_span,
        y0=x,
        method="RK45",
        atol=1e-8,
        rtol=1e-8,
    )
    for i in range(len(sol["y"][0])):
        x[0] += sol["y"][0][i]
    x[0] = x[0] / len(sol["y"][0])
    x[1] = musculotendon_length - x[0] * np.cos(pennation)
    x[2] = velocity_calculation_damped(x[0] / optimal_fiber_length, (x[1]) / tendon_slack_length) * muscle_velocity_max
    cpt += 1e-4
    while True:
        sol = solve_ivp(
            func_muscle_velocity,
            t_span=[cpt, cpt + 1e-4],
            y0=x,
            method="RK45",
            atol=1e-8,
            rtol=1e-8,
        )
        if sol["status"] == -1:
            print("Error status == -1")
            break
        previous = x[0]
        x[0] = 0
        for i in range(len(sol["y"][0])):
            x[0] += sol["y"][0][i]
        x[0] = x[0] / len(sol["y"][0])
        if previous == x[0]:
            break
        x[1] = musculotendon_length - x[0] * np.cos(pennation)
        x[2] = (
            velocity_calculation_damped(x[0] / optimal_fiber_length, x[1] / tendon_slack_length) * muscle_velocity_max
        )
        lm += [x[0]]
        lt += [x[1]]
        vm += [x[2]]
        cpt += 1e-4
        if (
            np.abs(
                verification_equal_Ft_damped(
                    (x[1]) / tendon_slack_length,
                    x[0] / optimal_fiber_length,
                    x[2] / muscle_velocity_max,
                )
            )
            < 1e-8
        ):
            tab += [
                [
                    calculation_tendonforce_3rd_equation(x[1] / tendon_slack_length),
                    calculation_tendonforce_7th_equation_damped(
                        x[0] / optimal_fiber_length, x[2] / muscle_velocity_max
                    ),
                    x[0],
                    x[1],
                    x[2],
                ]
            ]
        """
        print(
            sol,
            verification_equal_Ft_damped(
                (x[1] - tendon_slack_length) / tendon_slack_length,
                x[0] / optimal_fiber_length,
                x[2] / muscle_velocity_max,
            ),
            calculation_tendonforce_3rd_equation((x[1] - tendon_slack_length) / tendon_slack_length),
            calculation_tendonforce_7th_equation_damped(x[0] / optimal_fiber_length, x[2] / muscle_velocity_max),
            [x[0], x[1] - tendon_slack_length, x[2]],
        )
        """
    j = 0
    for i in range(len(tab)):
        if (
            np.abs(tab[i][4]) < np.abs(tab[j][4])
            and x[2] < 1
            and calculation_tendonforce_3rd_equation((x[1]) / tendon_slack_length) > 0
        ):
            j = i
    if len(tab) != 0:
        print(tab[j])
        return calculation_tendonforce_3rd_equation(x[1] / tendon_slack_length)

    else:
        print("Pas de solution trouvé.")
        return 0
    # plt.plot(lt, label="with")
    # plt.legend()
    # plt.show()
    # print(tab)
    # plt.plot(tab)
    # plt.show()
    return [lt, lm, vm]


def integral_method_without_damp():

    x0 = [optimal_fiber_length, tendon_slack_length * 8.02, 0]  # muscle_velocity_max * 0.5]
    tab = []
    lm = []
    lt = []
    vm = []

    x = x0
    cpt = 0
    t_span = [cpt, cpt + 1e-4]
    sol = solve_ivp(
        func_muscle_velocity_without_damp,
        t_span=t_span,
        y0=x,
        method="RK45",
        atol=1e-8,
        rtol=1e-8,
    )
    for i in range(len(sol["y"][0])):
        x[0] += sol["y"][0][i]
    x[0] = x[0] / len(sol["y"][0])
    x[1] = musculotendon_length - x[0] * np.cos(pennation)
    x[2] = (
        -d3 / d2
        + (1 / d2) * np.sinh((1 / d1) * (fvm_inverse(x[0] / optimal_fiber_length, x[1] / tendon_slack_length) - d4))
    ) * muscle_velocity_max
    cpt += 1e-4
    while True:
        sol = solve_ivp(
            func_muscle_velocity_without_damp,
            t_span=[cpt, cpt + 1e-4],
            y0=x,
            method="RK45",
            atol=1e-8,
            rtol=1e-8,
        )
        if sol["status"] == -1:
            print("Error status == -1")
            print(sol)
            break
        previous = x[0]
        x[0] = 0
        for i in range(len(sol["y"][0])):
            x[0] += sol["y"][0][i]
        x[0] = x[0] / len(sol["y"][0])
        if previous == x[0]:
            break
        x[1] = musculotendon_length - x[0] * np.cos(pennation)
        x[2] = (
            -d3 / d2
            + (1 / d2) * np.sinh((1 / d1) * (fvm_inverse(x[0] / optimal_fiber_length, x[1] / tendon_slack_length) - d4))
        ) * muscle_velocity_max
        vm += [x[2]]
        lt += [x[1]]
        lm += [x[0]]
        cpt += 1e-4
        if (
            np.abs(
                verification_equal_Ft(
                    x[1] / tendon_slack_length,
                    x[0] / optimal_fiber_length,
                    x[2] / muscle_velocity_max,
                )
            )
            < 1e-8
        ):
            tab += [
                [
                    calculation_tendonforce_3rd_equation(x[1] / tendon_slack_length),
                    calculation_tendonforce_7th_equation(x[0] / optimal_fiber_length, x[2] / muscle_velocity_max),
                    x[0],
                    x[1],
                    x[2],
                ]
            ]
        """
        print(
            sol,
            verification_equal_Ft_damped(
                (x[1] - tendon_slack_length) / tendon_slack_length,
                x[0] / optimal_fiber_length,
                x[2] / muscle_velocity_max,
            ),
            calculation_tendonforce_3rd_equation((x[1] - tendon_slack_length) / tendon_slack_length),
            calculation_tendonforce_7th_equation_damped(x[0] / optimal_fiber_length, x[2] / muscle_velocity_max),
            [x[0], x[1] - tendon_slack_length, x[2]],
        )
        """

    j = 0
    k = 0
    for i in range(len(tab)):
        if (
            np.abs(tab[i][4]) < np.abs(tab[j][4])
            and x[2] < 1
            and calculation_tendonforce_3rd_equation(x[1] / tendon_slack_length) > 0
        ):
            j = i
    if len(tab) != 0:
        print(tab[j])
        return calculation_tendonforce_3rd_equation(x[1] / tendon_slack_length)

    else:
        print("Pas de solution trouvé.")
        return 0
    # plt.plot(lt, label="without")
    # plt.legend()
    # plt.show()
    # print(tab)
    # plt.plot(tab)
    # plt.show()
    return [lt, lm, vm]


def integral_method_with_damp_newton():
    global temp
    x0 = [optimal_fiber_length, tendon_slack_length * 1.02, muscle_velocity_max * 0.5]
    tab = []
    lm = []
    lt = []
    vm = []
    x = x0
    cpt = 0
    temp = x[2]
    while True:
        sol = solve_ivp(
            func_muscle_velocity_newton,
            t_span=[cpt, cpt + 1e-3],
            y0=x,
            method="RK45",
            atol=1e-8,
            rtol=1e-8,
        )
        cpt += 1e-3
        previous = x[0]
        x[0] = 0
        for i in range(len(sol["y"][0])):
            x[0] += sol["y"][0][i]
        x[0] = x[0] / len(sol["y"][0])
        if x[0] == previous or cpt > 20:
            break
        x[1] = musculotendon_length - x[0] * np.cos(pennation)
        x[2] = newton(
            lambda y: fvm(y / muscle_velocity_max)
            - (
                ft(x[1] / tendon_slack_length) / np.cos(pennation)
                - fpas(x[0] / optimal_fiber_length)
                - damp * y / muscle_velocity_max
            )
            / (activation * fact(x[0] / optimal_fiber_length)),
            x[2],
            rtol=1e-8,
            tol=1e-8,
        )
        temp = x[2]
        """
        print(
            x,
            verification_equal_Ft_damped(
                x[1] / tendon_slack_length, x[0] / optimal_fiber_length, x[2] / muscle_velocity_max
            ),
        ) 
        """
        lm += [x[0]]
        lt += [x[1]]
        vm += [x[2]]

    print(
        calculation_tendonforce_3rd_equation(x[1] / tendon_slack_length),
        calculation_tendonforce_7th_equation_damped(x[0] / optimal_fiber_length, x[2] / muscle_velocity_max),
        x,
    )
    return calculation_tendonforce_3rd_equation(x[1] / tendon_slack_length)


def main():
    """
    temp_1 = timeit.timeit(lambda: calcul_longueurs(), number=1)
    temp_2 = timeit.timeit(lambda: minimize_func(), number=1)
    temp_4 = timeit.timeit(lambda: integral_method_without_damp(), number=1)
    temp_3 = timeit.timeit(lambda: integral_method_with_damp(), number=1)
    temp_5 = timeit.timeit(lambda: integral_method_with_damp_newton(), number=1)
    print(
        "\nExecution time of IPOPT sec:",
        temp_1,
        "\nExecution time of minimize_func sec:",
        temp_2,
        "\nExecution time of integration without damp sec:",
        temp_4,
        "\nExecution time of integration with damp sec:",
        temp_3,
        "\nExecution time of integration with damp and Newton sec:",
        temp_5,
    )
    # calcul_longueurs()
    # minimize_func()  # With this opti bounds of the velocity control the result
    """

    global activation
    ipopt = []
    scipy = []
    with_newton = []
    avec = []
    without = []
    x = []
    temp_1 = []
    temp_2 = []
    temp_3 = []
    temp_4 = []
    temp_5 = []
    for i in range(1, 101):
        activation = i * 1e-2
        x += [activation]
        # ipopt += [calcul_longueurs() / maximalIsometricForce]
        # scipy += [minimize_func() / maximalIsometricForce]
        # with_newton += [integral_method_with_damp_newton() / maximalIsometricForce]
        # avec += [integral_method_with_damp() / maximalIsometricForce]
        # without += [integral_method_without_damp() / maximalIsometricForce]
        temp_1 += [timeit.timeit(lambda: calcul_longueurs(), number=1)]
        temp_2 += [timeit.timeit(lambda: minimize_func(), number=1)]
        temp_3 += [timeit.timeit(lambda: integral_method_with_damp(), number=1)]
        temp_4 += [timeit.timeit(lambda: integral_method_without_damp(), number=1)]
        temp_5 += [timeit.timeit(lambda: integral_method_with_damp_newton(), number=1)]

    # plt.plot(x, without, label="without damping")
    # plt.plot(x, avec, label="with damping")
    # plt.plot(x, with_newton, label="with damping newton")
    # plt.plot(x, ipopt, label="ipopt")
    # plt.plot(x, scipy, label="scipy")

    plt.plot(temp_1, label="ipopt")
    plt.plot(temp_2, label="scipy")
    plt.plot(temp_3, label="with damp")
    plt.plot(temp_4, label="without damp")
    plt.plot(temp_5, label="with damp newton")

    # plt.title("Tendon Force Normalized with different values of activation")
    plt.title("Time of execution with different values of Activation")
    plt.xlabel("Activation")
    # plt.ylabel("Tendon Force normalized")
    plt.ylabel("Time of execution")
    # calcul_longueurs()
    # minimize_func()
    # integral_method_without_damp()
    # integral_method_with_damp_newton()
    # integral_method_with_damp()
    # Plot l'ensemble

    # plt.plot(with_newton[1], label="lt with_newton")
    # plt.plot(with_newton[0], label="lm with_newton")
    # plt.plot(with_newton[2], label="vm with_newton")

    # plt.plot(avec[0], label="lt with")
    # plt.plot(avec[1], label="lm with")
    # plt.plot(avec[2], label="vm with")

    # plt.plot(without[0], label="lt without")
    # plt.plot(without[1], label="lm without")
    # plt.plot(without[2], label="vm without")

    plt.legend()
    plt.grid(True)
    plt.show()

    """
    tab = []
    for i in range(0, 11):
        global activation
        activation = i * 1e-1
        tab += [float(calcul_longueurs())]
    plt.plot(tab)
    plt.show()
    """
    # diff_length = []
    # obj_func_value = []
    # for i in range(len(iterations)):
    #    diff_length.append(musculotendon_length - iterations[i][1] - iterations[i][0] * np.cos(pennation))
    #    obj_func_value.append(func_obj_minimize(iterations[i]))
    # plt.plot(diff_length, label="length_diff")
    # plt.plot(obj_func_value, label="obj_value")
    # plt.legend()
    # plt.show()

    """
    x = np.linspace(0.5, 1.5, 1000)
    y = np.linspace(-1, 1, 1000)
    w = np.linspace(1, 1.05, 1000)
    plt.plot(x, fact(x), label="fact")
    plt.plot(x, fpas(x), label="fpas")
    plt.plot(w, ft(w), label="ft")
    plt.plot(y, fvm(y), label="fvm")
    plt.legend()
    plt.show()
    """


if __name__ == "__main__":
    main()

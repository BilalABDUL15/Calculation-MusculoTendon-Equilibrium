from definition_func_class_without_equilibrium import *
import platform


from ....bioptim.bioptim import CostType
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np


from bioptim import SolutionMerge


import biorbd_casadi as biorbd
import biorbd as biorbd_eigen

from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedTorque,
    GeneralizedAcceleration,
)

import casadi
from casadi import SX, MX, vertcat, horzcat, norm_fro, Function, jacobian, norm_2

from typing import Callable, Any

from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedTorque,
    GeneralizedAcceleration,
)


from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsList,
    ConfigureProblem,
    DynamicsFcn,
    DynamicsFunctions,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    ObjectiveList,
    BoundsList,
    VariableScalingList,
    OdeSolver,
    OdeSolverBase,
    NonLinearProgram,
    Solver,
    DynamicsEvaluation,
    PhaseDynamics,
    PenaltyController,
    InitialGuessList,
    ControlType,
    PlotType,
)


def custom_dynamics(
    time: MX | SX,
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    algebraic_states: MX | SX,
    numerical_timeseries: MX | SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(x, u, p)

    Parameters
    ----------
    time: MX | SX
        The time of the system
    states: MX | SX
        The state of the system
    controls: MX | SX
        The controls of the system
    parameters: MX | SX
        The parameters acting on the system
    algebraic_states: MX | SX
        The algebraic states of the system
    nlp: NonLinearProgram
        A reference to the phase
    my_additional_factor: int
        An example of an extra parameter sent by the user

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    q = DynamicsFunctions_musculotendon_equilibrium.get(nlp.states["q"], states)
    qdot = DynamicsFunctions_musculotendon_equilibrium.get(nlp.states["qdot"], states)

    mus_activations = nlp.get_var_from_states_or_controls("muscles", states, controls)
    dq = DynamicsFunctions_musculotendon_equilibrium.compute_qdot(nlp, q, qdot)
    tau = DynamicsFunctions_musculotendon_equilibrium.compute_tau_from_muscle(nlp, q, qdot, mus_activations)
    ddq = nlp.model.forward_dynamics(q, qdot, tau)
    return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    my_additional_factor: int
        An example of an extra parameter sent by the user
    """

    # States
    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qddot(ocp, nlp, as_states=False, as_controls=False, as_states_dot=True)

    # Control
    ConfigureProblem.configure_muscles(ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics)


def prepare_ocp(
    biorbd_model_path: str,
    problem_type_custom: bool = True,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    # --- Options --- #
    # BioModel path
    global bio_model
    bio_model = BiorbdModel_musculotendon_equilibrium(biorbd_model_path)

    # Problem parameters
    global N
    global Time
    n_shooting = N

    final_time = Time

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=50)

    # Dynamics
    dynamics = DynamicsList()

    dynamics.add(
        custom_configure,
        dynamic_function=custom_dynamics,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    )

    # Constraints
    constraints = ConstraintList()

    activation_min, activation_max = 0.0, 1.0

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bio_model.bounds_from_ranges("qdot"))

    q0_init = q0_init_value
    q0_final = q0_final_value

    q0_offset = 0.25
    optimalLength = 0.10
    tendonSlackLength = 0.16

    x_bounds["q"].min += q0_offset
    x_bounds["q"].max += q0_offset

    x_bounds["q"][0, 0] = q0_init + q0_offset
    x_bounds["q"][0, -1] = q0_final + q0_offset

    x_bounds["qdot"].min[0, :] = -31.4
    x_bounds["qdot"].max[0, :] = 31.4

    x_bounds["qdot"].min[0, 0] = 0
    x_bounds["qdot"].max[0, 0] = 0

    x_bounds["qdot"].min[0, -1] = 0
    x_bounds["qdot"].max[0, -1] = 0

    x_init = InitialGuessList()
    x_init["q"] = q0_init + q0_offset
    x_init["qdot"] = 0

    u_bounds = BoundsList()
    u_bounds["muscles"] = [activation_min] * bio_model.nb_muscles, [activation_max] * bio_model.nb_muscles

    # u_bounds["muscles"][0, -1] = 0

    """
    without damping, there is a  singularity with the activation, 
    so it needs to be > (0 + eps) where eps is a neighborhood of 0, 
    eps depends on the muscle, it can be very small or not
    """
    # u_bounds["muscles"].min[0, :] = 0.5

    u_init = InitialGuessList()

    global LinearContinuous_value
    if LinearContinuous_value:
        return OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
            constraints=constraints,
            ode_solver=ode_solver,
            use_sx=use_sx,
            control_type=ControlType.LINEAR_CONTINUOUS,
        )
    else:
        return OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            x_scaling=x_scaling,
            u_scaling=u_scaling,
            objective_functions=objective_functions,
            constraints=constraints,
            ode_solver=ode_solver,
            use_sx=use_sx,
            control_type=ControlType.LINEAR_CONTINUOUS,
        )


def problem_resolution(
    path: str,
    n_shooting: int,
    Time_sim: float,
    q0_init: float,
    q0_final: float,
    LinearContinuous: bool = False,
    Collocation: bool = False,
):
    global N
    global Time
    global LinearContinuous_value
    global q0_init_value
    global q0_final_value

    q0_init_value = q0_init
    q0_final_value = q0_final
    LinearContinuous_value = LinearContinuous
    N = n_shooting
    Time = Time_sim

    if Collocation:
        ocp = prepare_ocp(biorbd_model_path=path, ode_solver=OdeSolver.COLLOCATION())
    else:
        ocp = prepare_ocp(biorbd_model_path=path, ode_solver=OdeSolver.RK4())

    # ocp.add_plot_penalty(CostType.ALL)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
    sol.print_cost()

    from bioptim import TimeAlignment

    musculoTendonLength = []
    tendon_length_normalized = []
    muscle_length_normalized = []
    musculoTendonVelocity = []

    model = biorbd_eigen.Model(path)
    m = model

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    controls = sol.stepwise_controls(to_merge=SolutionMerge.NODES)

    all_q = states["q"]
    all_qdot = states["qdot"]
    all_activation = controls["muscles"]
    q_tab = []
    qdot_tab = []
    activation_tab = []
    muscle_velocity_normalized_tab = []
    Fm_normalized = []
    Ft_normalized = []
    fpas = []

    muscle_velocity_max = 5

    """
    Code which calculate the forces manually by recalculating the muscle velocity at each node. At the end, we can
    compare what we've calculated with the output of Bioptim.
    """
    cpt = 0
    for q, qdot in zip(all_q.T, all_qdot.T):
        updated_kinematic_model = m.UpdateKinematicsCustom(q)
        m.updateMuscles(updated_kinematic_model, q)
        for group_idx in range(m.nbMuscleGroups()):
            for muscle_idx in range(m.muscleGroup(group_idx).nbMuscles()):
                musc = m.muscleGroup(group_idx).muscle(muscle_idx)
                global optimalLength
                optimalLength = musc.characteristics().optimalLength()
                tendonSlackLength = musc.characteristics().tendonSlackLength()
                pennationAngle = musc.characteristics().pennationAngle()
                maximalForce = musc.characteristics().forceIsoMax()
        musculoTendonLength += [musc.musculoTendonLength(updated_kinematic_model, q, True)]
        musculoTendonVelocity += [musc.velocity(m, q, qdot)]
        tendon_length_normalized += [tendonSlackLength / tendonSlackLength]
        muscle_length_normalized += [
            ((musculoTendonLength[-1] - tendonSlackLength) / np.cos(pennationAngle)) / optimalLength
        ]
        if cpt % 3 == 0:
            muscle_velocity_normalized_tab += [
                (musculoTendonVelocity[-1] / np.cos(pennationAngle)) / muscle_velocity_max
            ]
        cpt += 1
        q_tab.append(q[0])
        qdot_tab.append(qdot[0])

    for activation in all_activation.T:
        activation_tab.append(activation[0])

    if LinearContinuous:
        end_loop = 101
    else:
        end_loop = 100
    u = 0
    for i in range(200):
        for j in range(1):
            # if LinearContinuous and Collocation:
            #     p = 5 * i
            #     u = 2 * (i - 1)
            # elif Collocation and not LinearContinuous:
            #     p = 5 * i
            #     u = i - 1
            # elif not Collocation and LinearContinuous:
            #     p = i
            #     u = 2 * (i - 1)
            # else:
            #     p = i
            #     u = i - 1
            u = i
            p = 3 * i + j
            Fm_normalized.append(
                (
                    activation_tab[i]
                    * bio_model.fact(muscle_length_normalized[p])
                    * bio_model.fvm(muscle_velocity_normalized_tab[i])
                    + bio_model.fpas(muscle_length_normalized[p])
                )
                * np.cos(pennationAngle)
            )
            Ft_normalized.append(bio_model.ft(tendon_length_normalized[p]))
            fpas.append(bio_model.fpas(muscle_length_normalized[p]))
        print(
            "Fm_normalized pen:",
            Fm_normalized[-1] * np.cos(pennationAngle),
        )

    with open("/home/mickael/Desktop/without_equilibrium.txt", "w") as fichier:
        fichier.write("\nMusculoTendonLength\n")
        for lmt in musculoTendonLength:
            fichier.write(str(lmt))
            fichier.write(" ")

        fichier.write("\nMuscle\n")
        for lm in muscle_length_normalized:
            fichier.write(str(lm))
            fichier.write(" ")

        fichier.write("\nTendon\n")
        for lt in tendon_length_normalized:
            fichier.write(str(lt))
            fichier.write(" ")

        fichier.write("\nQ\n")
        for q in q_tab:
            fichier.write(str(q))
            fichier.write(" ")

        fichier.write("\nQdot\n")
        for qdot in qdot_tab:
            fichier.write(str(qdot))
            fichier.write(" ")

        fichier.write("\nActivation\n")
        for activation in activation_tab:
            fichier.write(str(activation))
            fichier.write(" ")

        fichier.write("\nMuscleVelocity\n")
        for vm in muscle_velocity_normalized_tab:
            fichier.write(str(vm))
            fichier.write(" ")

        fichier.write("\nMuscleForce\n")
        for fm in Fm_normalized:
            fichier.write(str(fm))
            fichier.write(" ")

        fichier.write("\nTendonForce\n")
        for ft in Ft_normalized:
            fichier.write(str(ft))
            fichier.write(" ")

        fichier.write("\nRealTime\n")
        fichier.write(str(sol.real_time_to_optimize))

        fichier.write("\nPassiveForce\n")
        for f in fpas:
            fichier.write(str(f))
            fichier.write(" ")

    # sol.print_cost()
    # --- Show results --- #
    plt.show()
    sol.animate(show_gravity_vector=False)
    return


def main():
    pass


if __name__ == "__main__":
    main()

from definition_func_class import *
import platform

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
    lm = DynamicsFunctions_musculotendon_equilibrium.get(nlp.states["lm"], states)

    mus_activations = nlp.get_var_from_states_or_controls("muscles", states, controls)
    dq = DynamicsFunctions_musculotendon_equilibrium.compute_qdot(nlp, q, qdot)
    tau, dlm = DynamicsFunctions_musculotendon_equilibrium.compute_tau_from_muscle(nlp, q, qdot, lm, mus_activations)
    ddq = nlp.model.forward_dynamics(q, qdot, tau)
    return DynamicsEvaluation(dxdt=vertcat(dq, ddq, dlm), defects=None)


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
    ConfigureProblem.configure_new_variable(
        name="lm",
        name_elements=("l"),
        ocp=ocp,
        nlp=nlp,
        as_states=True,
        as_controls=False,
        axes_idx=ConfigureProblem._apply_phase_mapping(ocp, nlp, "lm"),
    )

    # Control
    ConfigureProblem.configure_muscles(ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics)


def prepare_ocp(
    biorbd_model_path: str,
    problem_type_custom: bool = True,
    ode_solver: OdeSolverBase = OdeSolver.COLLOCATION(),
    use_sx: bool = False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    # --- Options --- #
    # BioModel path
    global bio_model
    bio_model = BiorbdModel_musculotendon_equilibrium(biorbd_model_path)

    # Problem parameters
    n_shooting = 100

    # final_time = 0.01
    final_time = 0.2

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
    x_bounds.add("lm", min_bound=0, max_bound=1)

    q0_init = -0.30
    q0_final = -0.22
    q0_offset = 0.3

    x_bounds["q"].min += q0_offset
    x_bounds["q"].max += q0_offset

    x_bounds["q"][0, 0] = q0_init + q0_offset
    x_bounds["q"][0, -1] = q0_final + q0_offset

    x_bounds["qdot"].min[0, :] = -31.4
    x_bounds["qdot"].max[0, :] = 31.4

    x_bounds["qdot"][0, 0] = 0
    x_bounds["qdot"][0, -1] = 0

    x_bounds["lm"].min[0, :] = 0.134 * 0.5  # optimalLength * 0.5
    x_bounds["lm"].max[0, :] = 0.134 * 1.5  # optimalLength * 1.5

    x_init = InitialGuessList()
    x_init["q"] = q0_init + q0_offset
    # x_init["qdot"] = 0
    x_init["lm"] = 0.134

    u_bounds = BoundsList()
    u_bounds["muscles"] = [activation_min] * bio_model.nb_muscles, [activation_max] * bio_model.nb_muscles
    """
    without damping, there is a  singularity with the activation, 
    so it needs to be > (0 + eps) where eps is a neighborhood of 0, 
    eps depends on the muscle, it can be very small or not
    """
    # u_bounds["muscles"].min[0, :] = 0.2

    u_init = InitialGuessList()
    u_init["muscles"] = 0.2

    x_scaling = VariableScalingList()

    u_scaling = VariableScalingList()

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
    )


def main():
    """import numpy as np
    import bioviz

    biorbd_viz = bioviz.Viz("models/test.bioMod")

    n_frames = 20
    all_q = np.zeros((biorbd_viz.nQ, n_frames))

    biorbd_viz.load_movement(all_q)
    biorbd_viz.exec()
    return"""

    model_path = "models/test.bioMod"
    # model_path = "models/exemple_test.bioMod"

    from bioptim import CostType
    import numpy as np
    import matplotlib.pyplot as plt

    ocp = prepare_ocp(biorbd_model_path=model_path)

    # ocp.add_plot_penalty(CostType.ALL)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
    sol.print_cost()

    musculoTendonLength = []
    tendon_length = []
    muscle_length = []
    musculoTendonVelocity = []
    length_muscle = []

    import matplotlib.pyplot as plt
    import numpy as np

    model = biorbd_eigen.Model("models/test.bioMod")
    # model = biorbd_eigen.Model("models/exemple_test.bioMod")
    m = model

    from bioptim import SolutionMerge

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    all_q = states["q"]
    all_qdot = states["qdot"]
    all_lm = states["lm"]
    all_activation = sol.decision_controls(to_merge=SolutionMerge.NODES)["muscles"]
    q_tab = []
    qdot_tab = []
    activation_tab = []
    muscle_velocity = []
    Fm = []
    Ft = []

    for q, qdot, lm in zip(all_q.T, all_qdot.T, all_lm.T):
        updated_kinematic_model = m.UpdateKinematicsCustom(q)
        m.updateMuscles(updated_kinematic_model, q)
        for group_idx in range(m.nbMuscleGroups()):
            for muscle_idx in range(m.muscleGroup(group_idx).nbMuscles()):
                musc = m.muscleGroup(group_idx).muscle(muscle_idx)
                optimalLength = musc.characteristics().optimalLength()
                tendonSlackLength = musc.characteristics().tendonSlackLength()
                pennationAngle = musc.characteristics().pennationAngle()
                maximalForce = musc.characteristics().forceIsoMax()
                muscle_velocity_max = 5
        musculoTendonLength += [musc.musculoTendonLength(updated_kinematic_model, q, True)]
        musculoTendonVelocity += [musc.velocity(m, q, qdot)]
        muscle_length += [lm[0]]
        tendon_length += [(musculoTendonLength[-1] - muscle_length[-1] * np.cos(pennationAngle))]
        q_tab.append(q[0])
        qdot_tab.append(qdot[0])

    for activation in all_activation.T:
        activation_tab.append(activation[0])

    """ Vm Fm Ft recalculation for Linear method"""

    for i in range(len(activation_tab)):
        if i == 0:
            alpha, beta = bio_model.tangent_factor_calculation(0)
            vm = (
                bio_model.ft(tendon_length[5 * i] / tendonSlackLength) / casadi.cos(pennationAngle)
                - bio_model.fpas(muscle_length[5 * i] / optimalLength)
                - beta * activation_tab[i] * bio_model.fact(muscle_length[5 * i] / optimalLength)
            ) / (alpha * activation_tab[i] * bio_model.fact(muscle_length[5 * i] / optimalLength) + damp)
        else:
            alpha, beta = bio_model.tangent_factor_calculation(vm / muscle_velocity_max)
            vm = (
                bio_model.ft(tendon_length[(5 * i) - 1] / tendonSlackLength) / casadi.cos(pennationAngle)
                - bio_model.fpas(muscle_length[(5 * i) - 1] / optimalLength)
                - beta * activation_tab[i] * bio_model.fact(muscle_length[(5 * i) - 1] / optimalLength)
            ) / (alpha * activation_tab[i] * bio_model.fact(muscle_length[(5 * i) - 1] / optimalLength) + damp)
        muscle_velocity.append(vm)
        Fm.append(
            maximalForce
            * (
                activation_tab[i]
                * bio_model.fact(muscle_length[5 * i] / optimalLength)
                * bio_model.fvm(muscle_velocity[i] / muscle_velocity_max)
                + bio_model.fpas(muscle_length[5 * i] / optimalLength)
            )
        )
        Ft.append(maximalForce * bio_model.ft(tendon_length[5 * i] / tendonSlackLength))
        print(Ft[-1], Fm[-1] * np.cos(pennationAngle))

    """ Vm Fm Ft recalculation for Newton method"""
    from scipy.optimize import newton

    """for i in range(len(activation_tab)):
        if i == 0:
            initial_guess = 0
            vm = newton(
                lambda muscle_velocity_initial: bio_model.fvm(muscle_velocity_initial / muscle_velocity_max)
                - (
                    bio_model.ft(tendon_length[5 * i] / tendonSlackLength) / np.cos(pennationAngle)
                    - bio_model.fpas(muscle_length[5 * i] / optimalLength)
                    - damp * muscle_velocity_initial / 5
                )
                / (activation_tab[i] * bio_model.fact(muscle_length[5 * i] / optimalLength)),
                initial_guess,
                tol=1e-8,
                rtol=1e-8,
            )
        else:
            initial_guess = vm
            vm = newton(
                lambda muscle_velocity_initial: bio_model.fvm(muscle_velocity_initial / muscle_velocity_max)
                - (
                    bio_model.ft(tendon_length[(5 * i) - 1] / tendonSlackLength) / np.cos(pennationAngle)
                    - bio_model.fpas(muscle_length[(5 * i) - 1] / optimalLength)
                    - damp * muscle_velocity_initial / 5
                )
                / (activation_tab[i] * bio_model.fact(muscle_length[(5 * i) - 1] / optimalLength)),
                initial_guess,
                tol=1e-8,
                rtol=1e-8,
            )
        muscle_velocity.append(vm)
        Fm.append(
            maximalForce
            * (
                activation_tab[i]
                * bio_model.fact(muscle_length[5 * i] / optimalLength)
                * bio_model.fvm(muscle_velocity[i] / muscle_velocity_max)
                + bio_model.fpas(muscle_length[5 * i] / optimalLength)
            )
        )
        Ft.append(maximalForce * bio_model.ft(tendon_length[5 * i] / tendonSlackLength))
        print(Ft[-1], Fm[-1] * np.cos(pennationAngle))"""

    model = BiorbdModel_musculotendon_equilibrium("models/test.bioMod")

    with open("with_equilibrium_linear_method.txt", "w") as fichier:
        # with open("with_equilibrium_newton_method.txt", "w") as fichier:
        fichier.write("MusculoTendonLength\n")
        for lmt in musculoTendonLength:
            fichier.write(str(lmt))
            fichier.write(" ")

        fichier.write("\nMuscle\n")
        for lm in muscle_length:
            fichier.write(str(lm))
            fichier.write(" ")

        fichier.write("\nTendon\n")
        for lt in tendon_length:
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
        for vm in muscle_velocity:
            fichier.write(str(vm))
            fichier.write(" ")

        fichier.write("\nMuscleForce\n")
        for fm in Fm:
            fichier.write(str(fm))
            fichier.write(" ")

        fichier.write("\nTendonForce\n")
        for ft in Ft:
            fichier.write(str(ft))
            fichier.write(" ")

        fichier.write("\nRealTime\n")
        fichier.write(str(sol.real_time_to_optimize))

    # plt.plot(ft)
    """
    plt.plot(muscle_length, label="lm")
    plt.plot(musculoTendonLength, label="lmt")
    plt.plot(tendon_length, label="lt")

    plt.legend()
    plt.show()
    """
    # sol.print_cost()
    # --- Show results --- #
    # sol.animate(show_gravity_vector=False)


if __name__ == "__main__":
    main()

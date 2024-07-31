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
from casadi import SX, MX, vertcat, horzcat, norm_fro, Function, jacobian, norm_2, vertsplit, horzsplit

from typing import Callable, Any
from casadi import Function
import numpy as np

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
    BiMapping,
    InitialGuessList,
    MultinodeConstraintList,
    MultinodeObjectiveList,
    MultinodeObjectiveFcn,
    ControlType,
    PlotType,
)

from bioptim.gui.plot import CustomPlot


def custom_multinode_constraint(controllers: list[PenaltyController, ...]) -> MX:
    """
    The constraint of the transition of the muscle velocity. We calculate the muscle velocity and we interpolate the
    control vm_c_normalized to it. Thanks to this, we can use the previous vm_normalized to calculate the new vm.

    Returns
    -------
    The constraint such that: vm_c_normalized_post - vm_normalized = 0
    """

    q_prev = controllers[0].states["q"].cx_start
    qdot_prev = controllers[0].states["qdot"].cx_start
    lm_normalized_prev = controllers[0].states["lm_normalized"].cx_start
    act = controllers[0].controls["muscles"].cx_start
    # vm_c_normalized_prev = controllers[0].algebraic_states["vm_c_normalized"].cx_start
    vm_c_normalized_prev = controllers[0].controls["vm_c_normalized"].cx_start
    # vm_c_normalized_prev = controllers[0].states_dot["vm_c_normalized"].cx_start

    # vm_normalized = bio_model.compute_vm_normalized(q_prev, qdot_prev, lm_normalized_prev, act, vm_c_normalized_prev)

    vm_normalized = vm_c_normalized_prev

    muscle_velocity_max = MX(5)
    optimalLength = MX(0.10)

    # dlm_normalized = controllers[0].controls["vm_c_normalized"].cx * muscle_velocity_max / optimalLength

    t_span = controllers[0].t_span.cx
    x = controllers[0].states.cx
    # u = vertcat(controllers[0].controls["muscles"].cx, dlm_normalized)
    u = controllers[0].controls.cx
    p = controllers[0].parameters.cx
    a = controllers[0].algebraic_states.cx
    d = MX(controllers[0].numerical_timeseries.cx)

    """
    x_all = controllers[0].integrate(t_span=t_span, x0=x, u=u, p=p, a=a, d=d)["xall"]
    for i in range(1, len(horzsplit(x_all))):
        vm_normalized_previous = vm_normalized
        q_current, qdot_current, lm_normalized_current = vertsplit(horzsplit(x_all)[i])
        vm_normalized = bio_model.compute_vm_normalized(
            q_current,
            qdot_current,
            lm_normalized_current,
            act,
            vm_normalized_previous,
        )
    """

    x_all = controllers[0].integrate(t_span=t_span, x0=x, u=u, p=p, a=a, d=d)["xf"]
    q_last, qdot_last, lm_normalized_last = vertsplit(x_all)
    vm_normalized = bio_model.compute_vm_normalized(
        q_last,
        qdot_last,
        lm_normalized_last,
        act,
        vm_c_normalized_prev,
    )

    vm_c_normalized_post = controllers[1].controls["vm_c_normalized"].cx_start
    # vm_c_normalized_post = controllers[1].states_dot["vm_c_normalized"].cx
    # vm_c_normalized_post = controllers[1].algebraic_states["vm_c_normalized"].cx_start
    return vm_c_normalized_post - vm_normalized


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
    lm_normalized = DynamicsFunctions_musculotendon_equilibrium.get(nlp.states["lm_normalized"], states)
    mus_activations = DynamicsFunctions_musculotendon_equilibrium.get(nlp.controls["muscles"], controls)
    vm_c_normalized = DynamicsFunctions_musculotendon_equilibrium.get(nlp.controls["vm_c_normalized"], controls)
    # vm_c_normalized = DynamicsFunctions_musculotendon_equilibrium.get(nlp.states_dot["vm_c_normalized"], states)
    # vm_c_normalized = DynamicsFunctions_musculotendon_equilibrium.get(
    #    nlp.algebraic_states["vm_c_normalized"], algebraic_states
    # )

    # vm_normalized = bio_model.compute_vm_normalized(q, qdot, lm_normalized, mus_activations, vm_c_normalized)
    vm_normalized = vm_c_normalized
    dq = DynamicsFunctions_musculotendon_equilibrium.compute_qdot(nlp, q, qdot)
    tau = DynamicsFunctions_musculotendon_equilibrium.compute_tau_from_muscle(
        nlp, q, qdot, lm_normalized, mus_activations, vm_normalized
    )

    ddq = nlp.model.forward_dynamics(q, qdot, tau)

    global muscle_velocity_max
    muscle_velocity_max = MX(5)
    optimalLength = MX(0.10)

    dlm_normalized = vm_normalized * muscle_velocity_max / optimalLength

    return DynamicsEvaluation(dxdt=vertcat(dq, ddq, dlm_normalized), defects=None)


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
        name="lm_normalized",
        name_elements=("l"),
        ocp=ocp,
        nlp=nlp,
        as_states=True,
        as_controls=False,
        axes_idx=ConfigureProblem._apply_phase_mapping(ocp, nlp, "lm_normalized"),
    )
    ConfigureProblem.configure_new_variable(
        name="vm_c_normalized",
        name_elements=("v"),
        ocp=ocp,
        nlp=nlp,
        as_states=False,
        as_controls=True,
        as_states_dot=False,
        as_algebraic_states=False,
        axes_idx=ConfigureProblem._apply_phase_mapping(ocp, nlp, "vm_c_normalized"),
    )

    # Control
    ConfigureProblem.configure_muscles(ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics)

    # configure_compute_vm(ocp, nlp, dyn_func=nlp.model.compute_vm_normalized)


def configure_compute_vm(ocp, nlp, dyn_func: Callable, **extra_params):
    """
    Configure the contact points

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    dyn_func: Callable[time, states, controls, param, algebraic_states]
        The function to get the values of contact forces from the dynamics
    """

    time_span_sym = vertcat(nlp.time_mx, nlp.dt_mx)

    def get_control(t_span, control):
        if control.shape[1] == 1:
            return control[:, 0]

        dt_norm = (t_span[1] - t_span[0]) / t_span[1]
        return control[:, 0] + (control[:, 1] - control[:, 0]) * dt_norm

    nlp.compute_vm = Function(
        "compute_vm_normalized",
        [
            time_span_sym,
            nlp.states.scaled.mx_reduced,
            nlp.controls.scaled.mx_reduced,
            nlp.parameters.scaled.mx_reduced,
            nlp.algebraic_states.scaled.mx_reduced,
            nlp.numerical_timeseries.mx,
        ],
        [
            dyn_func(
                nlp.get_var_from_states_or_controls("q", nlp.states.scaled.mx_reduced, nlp.controls.scaled.mx_reduced),
                nlp.get_var_from_states_or_controls(
                    "qdot", nlp.states.scaled.mx_reduced, nlp.controls.scaled.mx_reduced
                ),
                nlp.get_var_from_states_or_controls(
                    "lm_normalized", nlp.states.scaled.mx_reduced, nlp.controls.scaled.mx_reduced
                ),
                nlp.get_var_from_states_or_controls(
                    "muscles", nlp.states.scaled.mx_reduced, nlp.controls.scaled.mx_reduced
                ),
                nlp.get_var_from_states_or_controls(
                    "vm_c_normalized", nlp.states.scaled.mx_reduced, nlp.controls.scaled.mx_reduced
                ),
            )
        ],
        ["t_span", "x", "u", "p", "a", "d"],
        ["vm"],
    )

    all_vm = ["v"]
    all_vm_in_phase = ["v"]
    axes_idx = BiMapping(
        to_first=[i for i, c in enumerate(all_vm) if c in all_vm_in_phase],
        to_second=[i for i, c in enumerate(all_vm) if c in all_vm_in_phase],
    )

    nlp.plot["vm"] = CustomPlot(
        lambda t0, phases_dt, node_idx, x, u, p, a, d: np.array(
            nlp.compute_vm(
                np.concatenate([t0, t0 + phases_dt[nlp.phase_idx]]),
                x,
                # get_control(np.concatenate([t0, t0 + phases_dt[nlp.phase_idx]]), u),
                u[:, 0],
                p,
                a,
                d,
            )
        ),
        plot_type=PlotType.INTEGRATED,
        axes_idx=axes_idx,
        legend=all_vm,
        combine_to="vm_c_normalized_controls",
    )


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

    N = 100
    Time = 0.2

    n_shooting = N
    final_time = Time

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)

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
    multinode_constraint = MultinodeConstraintList()

    for node in range(100 - 1):
        multinode_constraint.add(custom_multinode_constraint, nodes_phase=[0, 0], nodes=[node, node + 1])
    multinode_constraint.add(custom_multinode_constraint, nodes_phase=[0, 0], nodes=[99, 98])

    activation_min, activation_max = 0.0, 1.0
    vm_c_normalized_min, vm_c_normalized_max = -1, 1

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bio_model.bounds_from_ranges("qdot"))
    x_bounds.add("lm_normalized", min_bound=0, max_bound=1)

    q0_init = -0.22
    q0_final = -0.31

    q0_offset = 0.25
    optimalLength = 0.10
    tendonSlackLength = 0.16

    x_bounds["q"].min += q0_offset
    x_bounds["q"].max += q0_offset

    gamma = 0.05
    eps = 0.1

    # x_bounds["q"].min[0, 0] = q0_init + q0_offset - gamma
    # x_bounds["q"].max[0, 0] = q0_init + q0_offset + gamma

    # x_bounds["q"].min[0, -1] = q0_final + q0_offset - gamma
    # x_bounds["q"].max[0, -1] = q0_final + q0_offset + gamma

    x_bounds["q"][0, 0] = q0_init + q0_offset
    x_bounds["q"][0, -1] = q0_final + q0_offset

    x_bounds["qdot"].min[0, :] = -31.4
    x_bounds["qdot"].max[0, :] = 31.4

    x_bounds["lm_normalized"].min[0, :] = 0.5
    x_bounds["lm_normalized"].max[0, :] = 1.5

    # x_bounds["qdot"].min[0, -1] = 0 - eps
    # x_bounds["qdot"].max[0, -1] = 0 + eps

    # x_bounds["qdot"].min[0, -1 ] = 0
    # x_bounds["qdot"].max[0, -1] = 0

    x_bounds["qdot"].min[0, 0] = 0
    x_bounds["qdot"].max[0, 0] = 0

    x_bounds["qdot"].min[0, -1] = 0
    x_bounds["qdot"].max[0, -1] = 0

    x_init = InitialGuessList()
    x_init["q"] = q0_init + q0_offset
    x_init["qdot"] = 0
    x_init["lm_normalized"] = (-q0_init - tendonSlackLength) / optimalLength

    u_bounds = BoundsList()
    u_bounds["muscles"] = [activation_min] * bio_model.nb_muscles, [activation_max] * bio_model.nb_muscles
    u_bounds["vm_c_normalized"] = [vm_c_normalized_min] * bio_model.nb_muscles, [
        vm_c_normalized_max
    ] * bio_model.nb_muscles

    u_bounds["vm_c_normalized"].min[0, [0, -1]] = 0
    u_bounds["vm_c_normalized"].max[0, [0, -1]] = 0

    """
    without damping, there is a  singularity with the activation, 
    so it needs to be > (0 + eps) where eps is a neighborhood of 0, 
    eps depends on the muscle, it can be very small or not
    """
    # u_bounds["muscles"].min[0, :] = 0.5

    u_init = InitialGuessList()
    # u_init["muscles"] = 0.5
    u_init["vm_c_normalized"] = 0

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
        multinode_constraints=multinode_constraint,
        ode_solver=ode_solver,
        use_sx=use_sx,
        control_type=ControlType.LINEAR_CONTINUOUS,
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

    from bioptim import CostType, SolutionMerge
    import numpy as np
    import matplotlib.pyplot as plt

    ocp = prepare_ocp(biorbd_model_path=model_path)
    ocp.add_plot(
        "vm_c_normalized_controls",
        lambda t0, phases_dt, node_idx, x, u, p, a, d: ocp.nlp[0].model.compute_vm_normalized_dm(
            t0,
            phases_dt,
            casadi.DM(x)[0, :],
            casadi.DM(x)[1, :],
            casadi.DM(x)[2, :],
            casadi.DM(u)[1, :] if len(u) else casadi.DM(0),
            casadi.DM(u)[0, :] if len(u) else casadi.DM(0),
        ),
        plot_type=PlotType.INTEGRATED,
    )
    # ocp.add_plot(
    #    "My New Extra Plot",
    #    lambda t0, phases_dt, node_idx, x, u, p, a, d: ocp.nlp[0].model.compute_lt_normalized(
    #        vertsplit(x), vertsplit(u)
    #    ),
    #    plot_type=PlotType.INTEGRATED,
    # )
    # ocp.add_plot_penalty(CostType.ALL)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))  # platform.system() == "Linux"))
    sol.graphs()
    sol.print_cost()
    musculoTendonLength = []
    tendon_length = []
    muscle_length = []
    musculoTendonVelocity = []
    length_muscle = []

    model = biorbd_eigen.Model("models/test.bioMod")
    # model = biorbd_eigen.Model("models/exemple_test.bioMod")
    m = model

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    all_q = states["q"]
    all_qdot = states["qdot"]
    all_lm_normalized = states["lm_normalized"]
    all_activation = sol.decision_controls(to_merge=SolutionMerge.NODES)["muscles"]
    q_tab = []
    qdot_tab = []
    activation_tab = []
    muscle_velocity_normalized = []
    Fm = []
    Ft = []

    for q, qdot, lm_normalized in zip(all_q.T, all_qdot.T, all_lm_normalized.T):
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
        muscle_length += [lm_normalized[0] * optimalLength]
        tendon_length += [(musculoTendonLength[-1] - muscle_length[-1] * np.cos(pennationAngle))]
        q_tab.append(q[0])
        qdot_tab.append(qdot[0])

    for activation in all_activation.T:
        activation_tab.append(activation[0])

    linear = False
    Collocation = False

    for i in range(len(activation_tab)):
        if Collocation == True:
            p = 5 * i
        else:
            p = i
        vm_normalized = sol.decision_controls(to_merge=SolutionMerge.NODES)["vm_c_normalized"].T[i][0]
        # muscle_velocity_normalized.append(vm_normalized)
        if linear:
            for j in range(1):
                previous_vm_normalized = vm_normalized
                alpha, beta = bio_model.tangent_factor_calculation(previous_vm_normalized)
                vm_normalized = bio_model.ft(tendon_length[p + j] / tendonSlackLength) / np.cos(pennationAngle)
                -bio_model.fpas(muscle_length[p + j] / optimalLength)
                -beta * activation_tab[i] * bio_model.fact(muscle_length[p + j] / optimalLength) / (
                    alpha * activation_tab[i] * bio_model.fact(muscle_length[p + j] / optimalLength) + damp
                )
                muscle_velocity_normalized.append(vm_normalized)

        else:
            from scipy.optimize import newton

            for j in range(1):
                vm_normalized_init = vm_normalized
                vm_normalized = newton(
                    func=lambda vm_n: maximalForce * bio_model.ft(tendon_length[p] / tendonSlackLength)
                    - maximalForce
                    * (
                        bio_model.fpas(muscle_length[p + j] / optimalLength)
                        + activation_tab[i] * bio_model.fact(muscle_length[p + j] / optimalLength) * bio_model.fvm(vm_n)
                        + bio_model.fdamp(vm_n)
                    )
                    * np.cos(pennationAngle),
                    x0=vm_normalized_init,
                    tol=1e-8,
                    rtol=1e-8,
                )
                muscle_velocity_normalized.append(vm_normalized)
        Fm.append(
            maximalForce
            * (
                activation_tab[i]
                * bio_model.fact(muscle_length[p] / optimalLength)
                * bio_model.fvm(muscle_velocity_normalized[i])
                + bio_model.fpas(muscle_length[p] / optimalLength)
            )
        )
        Ft.append(maximalForce * bio_model.ft(tendon_length[p] / tendonSlackLength))
        print("Ft:", Ft[-1], "Fm:", Fm[-1] * np.cos(pennationAngle), "Act:", activation_tab[i])

    global N
    global Time
    y = np.linspace(0, Time, N)
    # z = np.linspace(0, Time, 5 * N + 1)
    z = np.linspace(0, Time, N + 1)
    plt.subplot(1, 2, 1)
    plt.plot(z, muscle_length, label="lm")
    plt.plot(z, tendon_length, label="lt")
    plt.plot(z, musculoTendonLength, label="lmt")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(y, muscle_velocity_normalized, label="vm")
    plt.legend()
    plt.grid(True)

    plt.show()

    # sol.print_cost()
    # --- Show results --- #
    # sol.animate(show_gravity_vector=False)
    # return
    # with open("/home/mickael/Desktop/with_equilibrium_linear_method.txt", "w") as fichier:
    with open("/home/mickael/Desktop/with_equilibrium_newton_method.txt", "w") as fichier:
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
        for vm in muscle_velocity_normalized:
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


if __name__ == "__main__":
    main()

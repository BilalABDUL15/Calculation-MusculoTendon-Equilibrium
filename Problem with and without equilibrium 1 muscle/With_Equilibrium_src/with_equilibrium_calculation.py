from With_Equilibrium_src.definition_func_class import *
from With_Equilibrium_src.mt_eq_results import *
import platform


from bioptim import CostType, SolutionMerge
import numpy as np
import matplotlib.pyplot as plt


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
    QuadratureRule,
)

from bioptim.gui.plot import CustomPlot


def custom_multinode_constraint(controllers: list[PenaltyController, ...]) -> MX:
    """
    The constraint of the transition of the muscle velocity. We calculate the muscle velocity and we interpolate the
    control vm_c_normalized to it. Thanks to this, we can use the previous vm_normalized to calculate the new vm.

    This constraint will force the vm_c_normalized to be interpolated to the current vm_normalized intern calculated
    with the current states and controls. Indeed, to calculate vm_normalized intern, we need an initial guess for
    vm_normalized, so we use the previous vm_c_normalized.

    Returns
    -------
    The constraint such that: vm_c_normalized_post - vm_normalized = 0
    """

    act = controllers[0].controls["muscles"].cx
    vm_c_normalized_prev = controllers[0].controls["vm_c_normalized"].cx_start

    vm_normalized = vm_c_normalized_prev

    t_span = controllers[0].t_span.cx
    x = controllers[0].states.cx
    u = horzcat(controllers[0].controls.cx, controllers[1].controls.cx_start)
    p = controllers[0].parameters.cx
    a = controllers[0].algebraic_states.cx
    d = MX(controllers[0].numerical_timeseries.cx)

    x_f = controllers[0].integrate(t_span=t_span, x0=x, u=u, p=p, a=a, d=d)["xf"]
    q_last, qdot_last, lm_normalized_last = vertsplit(x_f)
    vm_normalized = bio_model.compute_vm_normalized(
        q_last,
        qdot_last,
        lm_normalized_last,
        act,
        vm_c_normalized_prev,
    )

    vm_c_normalized_post = controllers[1].controls["vm_c_normalized"].cx_start
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
    vm_normalized = vm_c_normalized
    dq = DynamicsFunctions_musculotendon_equilibrium.compute_qdot(nlp, q, qdot)
    tau = DynamicsFunctions_musculotendon_equilibrium.compute_tau_from_muscle(
        nlp, q, qdot, lm_normalized, mus_activations, vm_normalized
    )

    ddq = nlp.model.forward_dynamics(q, qdot, tau)

    """
    The derivative of the state lm_normalized is dlm_normalized which is different from vm_normalized. Indeed,
    one is normalized with respect to the optimal Length, while the other is normalized with respect to the 
    muscle velocity max. So we have to denormalize vm_normalized with muscle velocity max and normalize the result with
    optimal Length.
    """

    global muscle_velocity_max
    global optimalLength
    muscle_velocity_max = MX(5)

    dlm_normalized = vm_normalized * muscle_velocity_max / optimalLength

    return DynamicsEvaluation(dxdt=vertcat(dq, ddq, dlm_normalized), defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Vm_c_normalized is a normalized muscle velocity control. It is different from the muscle velocity intern
    calculated with the states and control of the current node. This control is used on the multinode constraint
    to transmit the muscle velocity interm of the current interval to the next one.

    vm_c_normalized : Muscle Velocity Control normalized != Vm_normalized which the muscle velocity calculated at each
    node with the states and controls.

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

    global Time
    global N
    global LinearContinuous_value
    global q0_init_value
    global q0_final_value

    n_shooting = N
    final_time = Time

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="muscles",
        weight=1,
        integration_rule=QuadratureRule.TRAPEZOIDAL,
    )

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

    for node in range(n_shooting - 1):
        multinode_constraint.add(custom_multinode_constraint, nodes_phase=[0, 0], nodes=[node, node + 1])
    multinode_constraint.add(custom_multinode_constraint, nodes_phase=[0, 0], nodes=[n_shooting - 1, n_shooting - 2])

    activation_min, activation_max = 0.0, 1.0
    vm_c_normalized_min, vm_c_normalized_max = -1, 1

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bio_model.bounds_from_ranges("qdot"))
    x_bounds.add("lm_normalized", min_bound=0, max_bound=1)

    q0_init = q0_init_value
    q0_final = q0_final_value

    global optimalLength
    q0_offset = 0.25
    optimalLength = 0.10
    tendonSlackLength = 0.16

    x_bounds["q"].min += q0_offset
    x_bounds["q"].max += q0_offset

    x_bounds["q"][0, 0] = q0_init + q0_offset
    x_bounds["q"][0, -1] = q0_final + q0_offset

    x_bounds["qdot"].min[0, :] = -31.4
    x_bounds["qdot"].max[0, :] = 31.4

    x_bounds["lm_normalized"].min[0, :] = 0.5
    x_bounds["lm_normalized"].max[0, :] = 1.5

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

    u_bounds["vm_c_normalized"].min[0, 0] = 0
    u_bounds["vm_c_normalized"].max[0, 0] = 0

    u_bounds["vm_c_normalized"].max[0, -1] = 0
    u_bounds["vm_c_normalized"].max[0, -1] = 0

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
            x_scaling=x_scaling,
            u_scaling=u_scaling,
            objective_functions=objective_functions,
            constraints=constraints,
            multinode_constraints=multinode_constraint,
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
            multinode_constraints=multinode_constraint,
            ode_solver=ode_solver,
            use_sx=use_sx,
            control_type=ControlType.CONSTANT,
        )


def problem_resolution(
    n_shooting: int,
    Time_sim: float,
    path: str,
    q0_init: float,
    q0_final: float,
    LinearContinuous: bool = False,
    Collocation: bool = False,
    Method_VM_Calculation_value: Method_VM_Calculation = Method_VM_Calculation.DAMPING_NEWTON,
):
    global Time
    global N
    global LinearContinuous_value
    global q0_init_value
    global q0_final_value

    q0_init_value = q0_init
    q0_final_value = q0_final
    LinearContinuous_value = LinearContinuous
    N = n_shooting
    Time = Time_sim

    Method_VM_Calculation.method_used_vm = Method_VM_Calculation_value

    if Collocation:
        ocp = prepare_ocp(biorbd_model_path=path, ode_solver=OdeSolver.COLLOCATION())
    else:
        ocp = prepare_ocp(biorbd_model_path=path, ode_solver=OdeSolver.RK4())

    """
    Custom plot:
        1 - Plot the muscle velocity calculated to compare its value with the muscle velocity control
        2 - Plot the forces to verifiy if they are equal.
            2a - Plot the Muscle Force
            2b - Plot the Tendon Force
    """
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
    ocp.add_plot(
        "Muscle_forces_Values",
        lambda t0, phases_dt, node_idx, x, u, p, a, d: ocp.nlp[0].model.Muscle_force_dm(
            t0,
            phases_dt,
            casadi.DM(x)[0, :],
            casadi.DM(x)[1, :],
            casadi.DM(x)[2, :],
            casadi.DM(u)[1, :] if len(u) else casadi.DM(0),
            casadi.DM(u)[0, :] if len(u) else casadi.DM(0),
        ),
        plot_type=PlotType.INTEGRATED,
        label="Muscle_Force",
    )
    ocp.add_plot(
        "Muscle_forces_Values",
        lambda t0, phases_dt, node_idx, x, u, p, a, d: ocp.nlp[0].model.Tendon_force_dm(
            t0,
            phases_dt,
            casadi.DM(x)[0, :],
            casadi.DM(x)[1, :],
            casadi.DM(x)[2, :],
            casadi.DM(u)[1, :] if len(u) else casadi.DM(0),
            casadi.DM(u)[0, :] if len(u) else casadi.DM(0),
        ),
        label="Tendon_Force",
        plot_type=PlotType.INTEGRATED,
    )

    # ocp.add_plot_penalty(CostType.ALL)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))  # platform.system() == "Linux"))
    sol.print_cost()

    Result = Musculotendon_equilibrium_results(path)
    Result.show_results(
        sol=sol,
        path=path,
        Time=Time,
        N=N,
        Collocation=Collocation,
        LinearContinuous=LinearContinuous,
    )

    sol.graphs()
    plt.show()
    sol.print_cost()
    # --- Show results --- #
    sol.animate(show_gravity_vector=False)
    return


def main():
    pass


if __name__ == "__main__":
    main()

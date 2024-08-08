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
from casadi import SX, MX, vertcat, horzcat, norm_fro, Function, jacobian, fabs, norm_2

from typing import Callable, Any


from bioptim import CostType, SolutionMerge
import numpy as np
import matplotlib.pyplot as plt


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
    OdeSolver,
    OdeSolverBase,
    NonLinearProgram,
    Solver,
    DynamicsEvaluation,
    PhaseDynamics,
)


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


""" Damp from Millard doc: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3705831/pdf/bio_135_2_021005.pdf"""
damp = 0.1


class Method_VM_Calculation:
    """Which Method we want to use to calculate the vm_normalized"""

    DAMPING_NEWTON = "DAMPING_NEWTON_METHOD_VM"
    DAMPING_LINEAR = "DAMPING_LINEAR_METHOD_VM"
    WTHOUT_DAMPING = "WITHOUT_DAMPING_METHOD_VM"
    method_used_vm = None


class DynamicsFunctions_musculotendon_equilibrium(DynamicsFunctions):
    @staticmethod
    def compute_tau_from_muscle(
        nlp,
        q: MX | SX,
        qdot: MX | SX,
        lm_normalized: MX | SX,
        vm_c_normalized: MX | SX,
        muscle_activations: MX | SX,
        fatigue_states: MX | SX = None,
    ):

        activations = []
        for k in range(len(nlp.controls["muscles"])):
            if fatigue_states is not None:
                activations.append(muscle_activations[k] * (1 - fatigue_states[k]))
            else:
                activations.append(muscle_activations[k])
        return nlp.model.muscle_joint_torque(activations, q, qdot, lm_normalized, vm_c_normalized)


class BiorbdModel_musculotendon_equilibrium(BiorbdModel):
    """Musculotendon Equilibrium"""

    def vm_normalized_calculation_without_damping(
        self,
        lm_normalized: MX,
        tendon_length_normalized: MX,
        pennationAngle: MX,
        MaximalForce: MX,
        activation: MX,
        vm_c_normalized: MX,
        q: MX,
    ):
        """
        Calculation of vm when there is no damping, so the activation is a singularity here (by dividing by 0).
        """

        fv_inv = (self.ft(tendon_length_normalized) / casadi.cos(pennationAngle) - self.fpas(lm_normalized)) / (
            activation * self.fact(lm_normalized)
        )

        vm_normalized = 1 / d2 * (casadi.sinh(1 / d1 * (fv_inv - d4)) - d3)
        return vm_normalized

    def fpas(self, muscle_length_normalized):
        """Force passive definition = fpce
        Warning modification of the equation du to sign issue when muscle_length_normalized is under 1 !!!"""
        offset = (casadi.exp(kpe * (0 - 1) / e0) - 1) / (casadi.exp(kpe) - 1)
        return (casadi.exp(kpe * (muscle_length_normalized - 1) / e0) - 1) / (casadi.exp(kpe) - 1) - offset

    def fact(self, muscle_length_normalized):
        """Force active definition = flce"""
        return (
            b11
            * casadi.exp(
                (-0.5) * ((muscle_length_normalized - b21) ** 2) / ((b31 + b41 * muscle_length_normalized) ** 2)
            )
            + b12
            * casadi.exp((-0.5) * (muscle_length_normalized - b22) ** 2 / ((b32 + b42 * muscle_length_normalized) ** 2))
            + b13
            * casadi.exp((-0.5) * (muscle_length_normalized - b23) ** 2 / ((b33 + b43 * muscle_length_normalized) ** 2))
        )

    def fvm(self, muscle_velocity_normalized):
        """Muscle force velocity equation = fvce"""
        return (
            d1
            * casadi.log(
                (d2 * muscle_velocity_normalized + d3) + casadi.sqrt(((d2 * muscle_velocity_normalized + d3) ** 2) + 1)
            )
            + d4
        )

    def ft(self, tendon_length_normalized):
        """
        ft(lt) tendon force calculation with tendon length normalized
        Offset here because without it we have ft(1) < 0 whereas it should be 0.
        """
        offset = 0.01175075667752834
        return c1 * casadi.exp(kt * (tendon_length_normalized - c2)) - c3 + offset

    def fdamp(self, muscle_velocity_normalized):
        """
        Damping force Definition
        """
        return damp * muscle_velocity_normalized

    def tangent_factor_calculation(self, point):
        """

        Calculation of the coefficient of the Tangent equation of muscle velocity force to linearize fvm around
        a point.

        ---------------------------------------------------------------------------------------
        Paramameters:
        point : muscle velocity normalize at which we want to compute the tangent factor

        return coefficients of the linearize form of fvm => y = a * x + b return a,b
        ---------------------------------------------------------------------------------------
        """
        return [
            (d1 / casadi.log(10))
            * (
                (d2 * casadi.sqrt((d2 * point + d3) ** 2 + 1) + d2 + d3)
                / ((d2 * point + d3) * casadi.sqrt((d2 * point + d3) ** 2 + 1) + (d2 * point + d3) ** 2 + 1)
            ),
            (d1 / casadi.log(10))
            * (
                (d2 * casadi.sqrt((d2 * point + d3) ** 2 + 1) + d2 + d3)
                / ((d2 * point + d3) * casadi.sqrt((d2 * point + d3) ** 2 + 1) + (d2 * point + d3) ** 2 + 1)
            )
            * point
            + self.fvm(point),
        ]

    def compute_vm_normalized(self, q: MX, qdot: MX, lm_normalized: MX, act: MX, vm_c_normalized: MX):
        """
        Calculation of the muscle velocity intern.
        ---------------------------------------------------------------------------------------
        Paramameters:
        q,qdot,lm_normalized,act : state
        act : muscle activation
        vm_c_normalized : muscle velocity normalized control. We use this value as an initial guess for our different
        method of calculation of the muscle velocity intern.

        Return the value of muscle velocity intern (Calculated)
        ---------------------------------------------------------------------------------------
        """

        _, vm_normalized, _, _ = self.Muscles_parameters(q, qdot, lm_normalized, act, vm_c_normalized)

        return vm_normalized

    def compute_vm_normalized_dm(self, t0, phases_dt, q: MX, qdot: MX, lm_normalized: MX, act: MX, vm_c_normalized: MX):
        """
        Function which is used to plot the value of the muscle velocity intern to compare it to the value of the muscle
        velocity normalized control.
        """

        def get_control(dt_norm, control):
            if control.shape[1] == 1:
                return control[:, 0]

            return (control[:, 0] + (control[:, 1] - control[:, 0]) * dt_norm).T

        q_mx = MX.sym("q", self.nb_q, 1)
        qdot_mx = MX.sym("qdot", self.nb_qdot, 1)
        lm_normalized_mx = MX.sym("lm", self.nb_muscles, 1)
        act_mx = MX.sym("act", self.nb_muscles, 1)
        vm_c_normalized_mx = MX.sym("vm_c", self.nb_muscles, 1)

        _, vm_normalized, _, _ = self.Muscles_parameters(q_mx, qdot_mx, lm_normalized_mx, act_mx, vm_c_normalized_mx)

        dt = np.linspace(0, 1, q.shape[1])
        return np.array(
            casadi.Function("vm_n", [q_mx, qdot_mx, lm_normalized_mx, act_mx, vm_c_normalized_mx], [vm_normalized])(
                q, qdot, lm_normalized, get_control(dt, act), get_control(dt, vm_c_normalized)
            )
        )

    def Muscle_force_dm(self, t0, phases_dt, q: MX, qdot: MX, lm_normalized: MX, act: MX, vm_c_normalized: MX):
        """
        Function which is used to plot the value of the muscle force pennated.
        """

        def get_control(dt_norm, control):
            if control.shape[1] == 1:
                return control[:, 0]

            return (control[:, 0] + (control[:, 1] - control[:, 0]) * dt_norm).T

        q_mx = MX.sym("q", self.nb_q, 1)
        qdot_mx = MX.sym("qdot", self.nb_qdot, 1)
        lm_normalized_mx = MX.sym("lm", self.nb_muscles, 1)
        act_mx = MX.sym("act", self.nb_muscles, 1)
        vm_c_normalized_mx = MX.sym("vm_c", self.nb_muscles, 1)

        _, _, Muscular_force, _ = self.Muscles_parameters(q_mx, qdot_mx, lm_normalized_mx, act_mx, vm_c_normalized_mx)
        pennationAngle = self.model.muscleGroup(0).muscle(0).characteristics().pennationAngle().to_mx()
        dt = np.linspace(0, 1, q.shape[1])
        return np.array(
            casadi.Function(
                "Muscle_force",
                [q_mx, qdot_mx, lm_normalized_mx, act_mx, vm_c_normalized_mx],
                [Muscular_force * casadi.cos(pennationAngle)],
            )(q, qdot, lm_normalized, get_control(dt, act), get_control(dt, vm_c_normalized))
        )

    def Tendon_force_dm(self, t0, phases_dt, q: MX, qdot: MX, lm_normalized: MX, act: MX, vm_c_normalized: MX):
        """
        Function which is used to plot the value of the tendon force .
        """

        def get_control(dt_norm, control):
            if control.shape[1] == 1:
                return control[:, 0]

            return (control[:, 0] + (control[:, 1] - control[:, 0]) * dt_norm).T

        q_mx = MX.sym("q", self.nb_q, 1)
        qdot_mx = MX.sym("qdot", self.nb_qdot, 1)
        lm_normalized_mx = MX.sym("lm", self.nb_muscles, 1)
        act_mx = MX.sym("act", self.nb_muscles, 1)
        vm_c_normalized_mx = MX.sym("vm_c", self.nb_muscles, 1)

        _, _, _, Tendon_force = self.Muscles_parameters(q_mx, qdot_mx, lm_normalized_mx, act_mx, vm_c_normalized_mx)

        dt = np.linspace(0, 1, q.shape[1])
        return np.array(
            casadi.Function(
                "Tendon_force", [q_mx, qdot_mx, lm_normalized_mx, act_mx, vm_c_normalized_mx], [Tendon_force]
            )(q, qdot, lm_normalized, get_control(dt, act), get_control(dt, vm_c_normalized))
        )

    def Forces_calculation(
        self,
        tendon_length_normalized: MX,
        MaximalForce: MX,
        activation: MX,
        lm_normalized: MX,
        vm_normalized: MX,
    ):
        """
        Calculation of the value of the muscle force and tendon forces
        ------------------------------------------------------------------------------------------------
        Parameters:
        tendon_length_normalized: value of the tendon length normalized with tendon Slack Length,
        MaximalForce : maximal isometric force of the muscle,
        activation : control of the system,
        lm_normalized : muscle length normalized states of the system,
        vm_normalized: muscle velocity intern,

        ------------------------------------------------------------------------------------------------
        Returns Muscle Force, Tendon Force

        """
        return MaximalForce @ (
            self.fpas(lm_normalized)
            + self.fdamp(vm_normalized)
            + activation @ self.fact(lm_normalized) @ self.fvm(vm_normalized)
        ), MaximalForce @ self.ft(tendon_length_normalized)

    def muscle_joint_torque(self, activations, q, qdot, lm_normalized, vm_c_normalized) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_muscle_size(activations)
        jacobian_length, _, Muscular_force, Tendon_force = self.Muscles_parameters(
            q, qdot, lm_normalized, activations[0], vm_c_normalized
        )
        tau = -casadi.transpose(jacobian_length) @ Muscular_force
        return tau

    def vm_normalized_calculation_newton(
        self,
        lm_normalized,
        tendon_length_normalized: MX,
        pennationAngle: MX,
        MaximalForce: MX,
        activation: MX,
        vm_c_normalized: MX,
        q: MX,
    ):
        """
        Use of Newton method to calculate muscle velocity normalized. Indeed, lm_normalized is a state and the activation
        is a control, so vm_normalized is the only unknown here.
        The equation in which the Newton Method is used is the substraction between the Tendon force and the muscle
        force pennated. The inital guess is the previous vm_normalized.
        calculated.
        ------------------------------------------------------------------------------------------------
        Parameters:
        tendon_length_normalized: value of the tendon length normalized with tendon Slack Length,
        MaximalForce: maximal isometric force of the muscle,
        activation,
        lm_normalized,
        pennationAngle,
        activation
        vm_c_normalized: muscle velocity control which is used as our initial guess here


        ------------------------------------------------------------------------------------------------
        Return muscle velocity normalized intern (Calculated): vm_normalized

        """
        from casadi import Function, rootfinder, vertcat, cos

        vm_normalized_sym = MX.sym("vm_normalized")
        tendon_length_normalized_sym = MX.sym("tendon_length_normalized")
        muscle_length_normalized_sym = MX.sym("muscle_length_normalized")
        q_sym = MX.sym("q")
        act_sym = MX.sym("act")

        Ft = MaximalForce @ self.ft(tendon_length_normalized_sym)
        Fm_pen = (
            MaximalForce
            @ (
                self.fpas(muscle_length_normalized_sym)
                + act_sym @ self.fact(muscle_length_normalized_sym) @ self.fvm(vm_normalized_sym)
                + self.fdamp(vm_normalized_sym)
            )
            * casadi.cos(pennationAngle)
        )

        g = Ft - Fm_pen

        """
        The first variable of the function is the unknown value whereas the others are needed to determine this one. 
        To have more unknown values to calculate, need to use the vertcat function.
        """
        f_vm_normalized = Function(
            "f", [vm_normalized_sym, muscle_length_normalized_sym, tendon_length_normalized_sym, q_sym, act_sym], [g]
        )
        newton_method = rootfinder(
            "newton_method",
            "newton",
            f_vm_normalized,
            {
                "error_on_fail": False,
                "enable_fd": False,
                "print_in": False,
                "print_out": False,
                "max_num_dir": 10,
            },
        )
        vm_normalized = newton_method(vm_c_normalized, lm_normalized, tendon_length_normalized, q, activation)
        return vm_normalized

    def vm_normalized_calculation_linear(
        self,
        lm_normalized,
        tendon_length_normalized: MX,
        pennationAngle: MX,
        activation: MX,
        MaximalForce: MX,
        vm_c_normalized: MX,
        q: MX,
    ):
        """
        To calculate the muscle velocity intern, we linearize the muscle velocity force around the value of muscle
        velocity control. After this, we can calculate the coefficient of the linearized equation.
        Using these coefficient, we can isolate muscle velocity in the differential equation of muscle length.
        y = a * x + b


        ------------------------------------------------------------------------------------------------
        Parameters:
        tendon_length_normalized: value of the tendon length normalized with tendon Slack Length,
        MaximalForce: maximal isometric force of the muscle,
        activation,
        lm_normalized,
        pennationAngle,
        activation
        vm_c_normalized: muscle velocity control which is used as our initial guess here


        ------------------------------------------------------------------------------------------------
        Return muscle velocity normalized intern (Calculated): vm_normalized
        """
        alpha, beta = self.tangent_factor_calculation(vm_c_normalized)
        # alpha, beta = 2.2, 1 # linearize around 0
        alpha, beta = 1, 1
        return (
            self.ft(tendon_length_normalized) / casadi.cos(pennationAngle)
            - self.fpas(lm_normalized)
            - beta * activation[0] @ self.fact(lm_normalized)
        ) / (alpha * activation[0] @ self.fact(lm_normalized) + damp)

    def Muscle_Jacobian(
        self,
        q: MX,
        qdot: MX,
    ):
        """
        Compute the Jacobian of the muscle
        """
        return self.model.musclesLengthJacobian(self.model, q, False).to_mx()

    def vm_calculation(
        self,
        lm_normalized: MX,
        tendon_length_normalized: MX,
        pennationAngle: MX,
        MaximalForce: MX,
        activation: MX,
        vm_c_normalized: MX,
        q: MX,
        qdot: MX,
    ):
        """
        Choice of the method to calculate muscle velocity intern.
        """
        if Method_VM_Calculation.method_used_vm == Method_VM_Calculation.DAMPING_NEWTON:
            return self.vm_normalized_calculation_newton(
                lm_normalized=lm_normalized,
                tendon_length_normalized=tendon_length_normalized,
                pennationAngle=pennationAngle,
                MaximalForce=MaximalForce,
                activation=activation,
                vm_c_normalized=vm_c_normalized,
                q=q,
            )

        elif Method_VM_Calculation.method_used_vm == Method_VM_Calculation.DAMPING_LINEAR:
            return self.vm_normalized_calculation_linear(
                lm_normalized=lm_normalized,
                tendon_length_normalized=tendon_length_normalized,
                pennationAngle=pennationAngle,
                MaximalForce=MaximalForce,
                activation=activation,
                vm_c_normalized=vm_c_normalized,
                q=q,
            )
        elif Method_VM_Calculation.method_used_vm == Method_VM_Calculation.WTHOUT_DAMPING:
            return self.vm_normalized_calculation_without_damping(
                lm_normalized=lm_normalized,
                tendon_length_normalized=tendon_length_normalized,
                pennationAngle=pennationAngle,
                MaximalForce=MaximalForce,
                activation=activation,
                vm_c_normalized=vm_c_normalized,
                q=q,
            )
        else:
            raise ValueError("Method of calculation doesn't exist.")

    def Muscles_parameters(
        self,
        q: MX,
        qdot: MX,
        lm_normalized: MX,
        activation: MX,
        vm_c_normalized: MX,
    ):
        """
        Calculation of the muscle characteristics.
        First we compute the Jacobian.
        Secondly, we read the characteristics of muscles and then compute the muscle forces and muscle velocity intern.
        ------------------------------------------------------------------------------------------------
        Parameters:
        states,
        controls,
        ------------------------------------------------------------------------------------------------
        Return Muscle Jacobian, Muscle forces, and muscle velocities intern (Calculated)
        """
        m = self.model
        Muscular_force = []
        Tendon_force = []
        updated_kinematic_model = m.UpdateKinematicsCustom(q, qdot)
        m.updateMuscles(updated_kinematic_model, q, qdot)
        length_jacobian = self.Muscle_Jacobian(q=q, qdot=qdot)
        for group_idx in range(m.nbMuscleGroups()):
            for muscle_idx in range(m.muscleGroup(group_idx).nbMuscles()):

                updated_kinematic_model = m.UpdateKinematicsCustom(q, qdot)
                m.updateMuscles(updated_kinematic_model, q, qdot)

                musc = m.muscleGroup(group_idx).muscle(muscle_idx)

                optimalLength = musc.characteristics().optimalLength().to_mx()
                tendonSlackLength = musc.characteristics().tendonSlackLength().to_mx()
                pennationAngle = musc.characteristics().pennationAngle().to_mx()
                MaximalForce = musc.characteristics().forceIsoMax().to_mx()

                musculoTendonLength = musc.musculoTendonLength(updated_kinematic_model, q, True).to_mx()
                tendon_length_normalized = (
                    musculoTendonLength - lm_normalized * optimalLength * casadi.cos(pennationAngle)
                ) / tendonSlackLength
                vm_normalized = self.vm_calculation(
                    lm_normalized=lm_normalized,
                    tendon_length_normalized=tendon_length_normalized,
                    pennationAngle=pennationAngle,
                    MaximalForce=MaximalForce,
                    activation=activation,
                    vm_c_normalized=vm_c_normalized,
                    q=q,
                    qdot=qdot,
                )
                Muscular_force_current, Tendon_force_current = self.Forces_calculation(
                    tendon_length_normalized=tendon_length_normalized,
                    MaximalForce=MaximalForce,
                    activation=activation,
                    lm_normalized=lm_normalized,
                    vm_normalized=vm_normalized,
                )
                Muscular_force = vertcat(Muscular_force, Muscular_force_current)
                Tendon_force = vertcat(Tendon_force, Tendon_force_current)
        return length_jacobian, vm_normalized, Muscular_force, Tendon_force

import platform

import biorbd_casadi as biorbd
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedTorque,
    GeneralizedAcceleration,
)

import casadi
from casadi import SX, MX, vertcat, horzcat, norm_fro, Function, jacobian, fabs, norm_2

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
    OdeSolver,
    OdeSolverBase,
    NonLinearProgram,
    Solver,
    DynamicsEvaluation,
    PhaseDynamics,
)

"""https://static-content.springer.com/esm/art%3A10.1007%2Fs10439-016-1591-9/MediaObjects/10439_2016_1591_MOESM1_ESM.pdf"""


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
        Calculation of vm when there is not any damping, so the activation is a singularity here.
        """

        fv_inv = (self.ft(tendon_length_normalized) / casadi.cos(pennationAngle) - self.fpas(lm_normalized)) / (
            activation * self.fact(lm_normalized)
        )

        vm_normalized = 1 / d2 * (casadi.sinh(1 / d1 * (fv_inv - d4)) - d3)
        return vm_normalized

    # Force passive definition = fpce
    # Warning modification of the equation du to sign issue when muscle_length_normalized is under 1
    def fpas(self, muscle_length_normalized):
        offset = (casadi.exp(kpe * (0 - 1) / e0) - 1) / (casadi.exp(kpe) - 1)
        return (casadi.exp(kpe * (muscle_length_normalized - 1) / e0) - 1) / (casadi.exp(kpe) - 1) - offset

    # Force active definition = flce
    def fact(self, muscle_length_normalized):
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

    # Muscle force velocity equation = fvce
    def fvm(self, muscle_velocity_normalized):
        return (
            d1
            * casadi.log(
                (d2 * muscle_velocity_normalized + d3) + casadi.sqrt(((d2 * muscle_velocity_normalized + d3) ** 2) + 1)
            )
            + d4
        )

    # ft(lt) tendon force calculation with tendon length normalized
    def ft(self, tendon_length_normalized):
        """Offset here because without it we have ft(1) < 0 whereas it should be 0."""
        offset = 0.01175075667752834
        return c1 * casadi.exp(kt * (tendon_length_normalized - c2)) - c3 + offset

    def fdamp(self, muscle_velocity_normalized):
        return damp * muscle_velocity_normalized

    def tangent_factor_calculation(self, point):
        """Coefficient of the Tangent equation of muscle velocity force"""
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

        _, vm_normalized, _, _ = self.set_muscles_from_q(q, qdot, lm_normalized, act, vm_c_normalized)

        return vm_normalized

    def compute_vm_normalized_dm(self, t0, phases_dt, q: MX, qdot: MX, lm_normalized: MX, act: MX, vm_c_normalized: MX):
        import numpy as np

        def get_control(dt_norm, control):
            if control.shape[1] == 1:
                return control[:, 0]

            return (control[:, 0] + (control[:, 1] - control[:, 0]) * dt_norm).T

        q_mx = MX.sym("q", self.nb_q, 1)
        qdot_mx = MX.sym("qdot", self.nb_qdot, 1)
        lm_normalized_mx = MX.sym("lm", self.nb_muscles, 1)
        act_mx = MX.sym("act", self.nb_muscles, 1)
        vm_c_normalized_mx = MX.sym("vm_c", self.nb_muscles, 1)

        _, vm_normalized, _, _ = self.set_muscles_from_q(q_mx, qdot_mx, lm_normalized_mx, act_mx, vm_c_normalized_mx)

        dt = np.linspace(0, 1, q.shape[1])
        return np.array(
            casadi.Function("vm_n", [q_mx, qdot_mx, lm_normalized_mx, act_mx, vm_c_normalized_mx], [vm_normalized])(
                q, qdot, lm_normalized, get_control(dt, act), get_control(dt, vm_c_normalized)
            )
        )

    def compute_lt_normalized(self, lm_normalized: MX, musculoTendonLength: MX, pennationAngle: MX, optimalLength: MX):
        optimalLength = MX(0.1)
        pennationAngle = MX(0)
        lm = lm_normalized * optimalLength
        return musculoTendonLength - lm * casadi.cos(pennationAngle)

    def Forces_calculation(
        self,
        tendon_length_normalized: MX,
        MaximalForce: MX,
        activation: MX,
        lm_normalized: MX,
        vm_normalized: MX,
    ):
        return MaximalForce @ (
            self.fpas(lm_normalized)
            + self.fdamp(vm_normalized)
            + activation @ self.fact(lm_normalized) @ self.fvm(vm_normalized)
        ), MaximalForce @ self.ft(tendon_length_normalized)

    def muscle_joint_torque(self, activations, q, qdot, lm_normalized, vm_c_normalized) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_muscle_size(activations)
        jacobian_length, _, Muscular_force, Tendon_force = self.set_muscles_from_q(
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
        Use of Newton method to calculate vm_normalized. Indeed, lm is a state and the activation is a control,
        so vm_normalized is the only unknown here. The equation in which the Newton Method is used is the substraction
        between the Tendon force and the muscle force pennated. The inital guess is the previous vm_normalized
        calculated.
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

        g = (Ft - Fm_pen) ** 2

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
        To calculate the velocity, we linearize the muscle velocity force around the t-1 value of muscle velocity.
        After this, we can calculate the coefficient of the linearized equation. Using this coefficient,
        we can isolate muscle velocity in the differential equation of muscle length.
        y = a * x + b
        """

        # vm_c_normalized = 0
        alpha, beta = self.tangent_factor_calculation(vm_c_normalized)
        return (
            self.ft(tendon_length_normalized) / casadi.cos(pennationAngle)
            - self.fpas(lm_normalized)
            - beta * activation[0] @ self.fact(lm_normalized)
        ) / (alpha * activation[0] @ self.fact(lm_normalized) + damp)

    def set_muscles_from_q(self, q, qdot, lm_normalized, activation, vm_c_normalized):
        m = self.model
        # data = MX(m.nbMuscles(), 1)
        Muscular_force = []
        Tendon_force = []
        updated_kinematic_model = m.UpdateKinematicsCustom(q, qdot)
        m.updateMuscles(updated_kinematic_model, q, qdot)
        length_jacobian = m.musclesLengthJacobian(m, q, False).to_mx()
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

                # vm_normalized = self.vm_normalized_calculation_without_damping(
                # vm_normalized = self.vm_normalized_calculation_linear(
                vm_normalized = self.vm_normalized_calculation_newton(
                    lm_normalized=lm_normalized,
                    tendon_length_normalized=tendon_length_normalized,
                    pennationAngle=pennationAngle,
                    MaximalForce=MaximalForce,
                    activation=activation,
                    vm_c_normalized=vm_c_normalized,
                    q=q,
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
                """If there are Via point : """
                # for k, pts in enumerate(musc.position().pointsInGlobal()):
                #     data.append(pts.to_mx())

        return length_jacobian, vm_normalized, Muscular_force, Tendon_force

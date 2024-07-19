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
        lm: MX | SX,
        muscle_activations: MX | SX,
        fatigue_states: MX | SX = None,
    ):

        activations = []
        for k in range(len(nlp.controls["muscles"])):
            if fatigue_states is not None:
                activations.append(muscle_activations[k] * (1 - fatigue_states[k]))
            else:
                activations.append(muscle_activations[k])
        return nlp.model.muscle_joint_torque(activations, q, qdot, lm)


class BiorbdModel_musculotendon_equilibrium(BiorbdModel):
    """Musculotendon Equilibrium"""

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

    def muscle_joint_torque(self, activations, q, qdot, lm) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_muscle_size(activations)
        jacobian_length, dlm, Muscular_force = self.set_muscles_from_q(q, qdot, lm, activations[0])
        tau = -casadi.transpose(jacobian_length) @ Muscular_force
        # tau2 = -casadi.transpose(jacobian_length) @ Tendon_force
        return tau, dlm

    def muscle_velocity_normalized_calculation(
        self,
        muscle_length_normalized: MX,
        tendon_length_normalized: MX,
        pennationAngle: MX,
        activation: MX,
        previous_vm_normalized: MX,
    ):
        """
        To calculate the velocity, we linearize the muscle velocity force around the t-1 value of muscle velocity.
        After this, we can calculate the coefficient of the linearized equation. Using this coefficient,
        we can isolate muscle velocity in the differential equation of muscle length.
        y = a * x + b
        """

        alpha, beta = self.tangent_factor_calculation(previous_vm_normalized)
        return (
            self.ft(tendon_length_normalized) / casadi.cos(pennationAngle)
            - self.fpas(muscle_length_normalized)
            - beta * activation[0] @ self.fact(muscle_length_normalized)
        ) / (alpha * activation[0] @ self.fact(muscle_length_normalized) + damp)

    def muscle_velocity_calculation_newton(
        self,
        lm: MX,
        optimalLength: MX,
        tendonSlackLength: MX,
        pennationAngle: MX,
        activation: MX,
        musculoTendonLength: MX,
        q: MX,
        dlm_previous_guess: MX = 1,
        dlm_max: MX = 5,
    ):
        """
        Use of Newton method to calculate dlm. Indeed, lm is a state and the activation is a control, so dlm is the only
        unknown here. The equation in which the Newton Method is used is the substraction between the Tendon force and
        the muscle force pennated. The inital guess is the previous dlm calculated.
        """
        from casadi import Function, rootfinder, vertcat, cos

        dlm = MX.sym("dlm")
        tendon_length = musculoTendonLength - lm * casadi.cos(pennationAngle)

        Ft = self.ft((tendon_length) / tendonSlackLength)
        Fm_pen = (
            self.fpas(lm / optimalLength)
            + activation @ self.fact(lm / optimalLength) @ self.fvm(dlm / dlm_max)
            + self.fdamp(dlm / dlm_max)
        ) * casadi.cos(pennationAngle)
        g = (Ft - Fm_pen) ** 2
        """
        The first variable of the function is the unknown value whereas the others are needed to determine this one. To have more 
        unknown values to calculate, need to use the vertcat function.
        """
        f_dlm = Function("f", [dlm, lm, q, activation], [g])
        newton_method = rootfinder(
            "newton_method",
            "newton",
            f_dlm,
            {
                "error_on_fail": False,
                "enable_fd": True,
                # "print_in": True,
                # "print_out": True,
            },
        )
        dlm = newton_method(dlm_previous_guess, lm, q, activation)
        return dlm

    def Muscular_force_dlm_calculation_without_damping(
        self,
        lm,
        musculoTendonLength: MX,
        optimalLength: MX,
        tendonSlackLength: MX,
        pennationAngle: MX,
        maximalForce: MX,
        activation: MX,
        muscle_velocity_max: MX = 5,
    ):
        """
        Calculation of dlm when there is not any damping, so the activation is a singularity here.
        """
        tendon_length = musculoTendonLength - lm * casadi.cos(pennationAngle)

        fv_inv = (
            self.ft(tendon_length / tendonSlackLength) / casadi.cos(pennationAngle) - self.fpas(lm / optimalLength)
        ) / (activation * self.fact(lm / optimalLength))

        dlm = 1 / d2 * (casadi.sinh(1 / d1 * (fv_inv - d4)) - d3)
        Muscular_force = maximalForce @ (
            self.fpas(lm / optimalLength)
            + activation @ self.fact(lm / optimalLength) @ self.fvm(dlm / muscle_velocity_max)
        )

        return dlm, Muscular_force

    def Muscular_force_dlm_calculation_newton_method(
        self,
        lm,
        musculoTendonLength: MX,
        optimalLength: MX,
        tendonSlackLength: MX,
        pennationAngle: MX,
        maximalForce: MX,
        activation: MX,
        muscle_velocity_max: MX = 5,
        q: MX = 0,
    ):
        """
        Calculation of the muscle Force when dlm is calculated with the Newton Method. First, dlm is determined with
        an arbitrary initial guess. Secondly, dlm is calculated reccursively with the previous dlm as initial guess.
        """
        tendon_length = musculoTendonLength - lm * casadi.cos(pennationAngle)
        dlm = self.muscle_velocity_calculation_newton(
            lm=lm,
            optimalLength=optimalLength,
            tendonSlackLength=tendonSlackLength,
            pennationAngle=pennationAngle,
            activation=activation,
            musculoTendonLength=musculoTendonLength,
            dlm_max=MX(muscle_velocity_max),
            q=q,
        )
        dlm = self.muscle_velocity_calculation_newton(
            lm=lm,
            optimalLength=optimalLength,
            tendonSlackLength=tendonSlackLength,
            pennationAngle=pennationAngle,
            activation=activation,
            musculoTendonLength=musculoTendonLength,
            dlm_previous_guess=dlm,
            dlm_max=MX(muscle_velocity_max),
            q=q,
        )

        Muscular_force = maximalForce @ (
            self.fpas(lm / optimalLength)
            - self.fdamp(dlm / muscle_velocity_max)
            + activation @ self.fact(lm / optimalLength) @ self.fvm(dlm / muscle_velocity_max)
        )

        return dlm, Muscular_force

    def Muscular_force_dlm_calculation(
        self,
        lm,
        musculoTendonLength: MX,
        optimalLength: MX,
        tendonSlackLength: MX,
        pennationAngle: MX,
        maximalForce: MX,
        activation: MX,
        muscle_velocity_max: MX = 5,
    ):

        tendon_length = musculoTendonLength - lm * casadi.cos(pennationAngle)

        dlm = (
            self.muscle_velocity_normalized_calculation(
                lm / optimalLength,
                tendon_length / tendonSlackLength,
                pennationAngle,
                activation,
                MX(0),
            )
            * muscle_velocity_max
        )
        dlm = (
            self.muscle_velocity_normalized_calculation(
                lm / optimalLength,
                tendon_length / tendonSlackLength,
                pennationAngle,
                activation,
                dlm / muscle_velocity_max,
                # we need to linearize around the previous vm normalized, so we need to divide by vm_max
            )
            * muscle_velocity_max
        )
        Muscular_force = maximalForce @ (
            self.fpas(lm / optimalLength)
            + self.fdamp(dlm / muscle_velocity_max)
            + activation @ self.fact(lm / optimalLength) @ self.fvm(dlm / muscle_velocity_max)
        )

        return dlm, Muscular_force

    def Muscular_force_calculation_direct_method(
        self,
        lm,
        musculoTendonLength: MX,
        optimalLength: MX,
        tendonSlackLength: MX,
        pennationAngle: MX,
        maximalForce: MX,
        activation: MX,
        muscle_velocity_max: MX = 5,
    ):
        tendon_length = musculoTendonLength - lm * casadi.cos(pennationAngle)
        dlm = 1
        Muscular_force = maximalForce @ (
            self.fpas(lm / optimalLength)
            + self.fdamp(dlm / muscle_velocity_max)
            + activation @ self.fact(lm / optimalLength) @ self.fvm(dlm / muscle_velocity_max)
        )
        return dlm, Muscular_force

    def set_muscles_from_q(self, q, qdot, lm, activation_muscles):
        m = self.model
        # data = MX(m.nbMuscles(), 1)
        Muscular_force = []
        updated_kinematic_model = m.UpdateKinematicsCustom(q)
        m.updateMuscles(updated_kinematic_model, q)
        length_jacobian = m.musclesLengthJacobian(m, q, False).to_mx()
        for group_idx in range(m.nbMuscleGroups()):
            for muscle_idx in range(m.muscleGroup(group_idx).nbMuscles()):
                musc = m.muscleGroup(group_idx).muscle(muscle_idx)
                optimalLength = musc.characteristics().optimalLength().to_mx()
                tendonSlackLength = musc.characteristics().tendonSlackLength().to_mx()
                pennationAngle = musc.characteristics().pennationAngle().to_mx()
                maximalForce = musc.characteristics().forceIsoMax().to_mx()

                muscle_velocity_max = 5
                musculoTendonLength = musc.musculoTendonLength(updated_kinematic_model, q, True).to_mx()
                # musculoTendonLength = fabs(q)

                # dlm, Muscular_force_current = self.Muscular_force_dlm_calculation_without_damping(
                dlm, Muscular_force_current = self.Muscular_force_dlm_calculation(
                    # dlm, Muscular_force_current = self.Muscular_force_dlm_calculation_newton_method(
                    lm,
                    musculoTendonLength,
                    optimalLength,
                    tendonSlackLength,
                    pennationAngle,
                    maximalForce,
                    activation_muscles,
                    muscle_velocity_max,
                    # q,
                )
                Muscular_force = vertcat(Muscular_force, Muscular_force_current)
                """If there are Via point : """
                # for k, pts in enumerate(musc.position().pointsInGlobal()):
                #     data.append(pts.to_mx())

        return length_jacobian, dlm, Muscular_force

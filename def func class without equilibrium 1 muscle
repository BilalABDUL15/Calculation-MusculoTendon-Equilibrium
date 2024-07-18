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
        muscle_activations: MX | SX,
        fatigue_states: MX | SX = None,
    ):

        activations = []
        for k in range(len(nlp.controls["muscles"])):
            if fatigue_states is not None:
                activations.append(muscle_activations[k] * (1 - fatigue_states[k]))
            else:
                activations.append(muscle_activations[k])
        return nlp.model.muscle_joint_torque(activations, q, qdot)


class BiorbdModel_musculotendon_equilibrium(BiorbdModel):
    """Musculotendon without Equilibrium"""

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

    def muscle_joint_torque(self, activations, q, qdot) -> MX:
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_muscle_size(activations)
        jacobian_length, Muscular_force = self.set_muscles_from_q(q, qdot, activations[0])
        tau = -casadi.transpose(jacobian_length) @ Muscular_force
        return tau

    def muscle_velocity_calculation(self, pennationAngle: MX, musculoTendonVelocity: MX):
        """
        Because tendon is rigid, its velocity is equal to zero so muscle velocity is equal to musculotendon velocity
        divided by the cosinus of the pennation Angle.
        vm = vmt / cos(pennationAngle)
        """
        return musculoTendonVelocity / casadi.cos(pennationAngle)

    def Muscular_force_calculation(
        self,
        optimalLength: MX,
        pennationAngle: MX,
        maximalForce: MX,
        muscle_velocity_max: MX,
        muscle_length: MX,
        musculoTendonVelocity: MX,
        activation: MX,
    ):
        muscle_velocity = self.muscle_velocity_calculation(
            pennationAngle,
            musculoTendonVelocity,
        )
        return maximalForce @ (
            activation @ self.fact(muscle_length / optimalLength) * self.fvm(muscle_velocity / muscle_velocity_max)
            + self.fpas(muscle_length / optimalLength)
        )

    def set_muscles_from_q(self, q, qdot, activation_muscles):
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

                musculoTendonVelocity = musc.velocity(updated_kinematic_model, q, qdot, True).to_mx()

                muscle_length = (musculoTendonLength - tendonSlackLength) / casadi.cos(pennationAngle)

                Muscular_force_current = self.Muscular_force_calculation(
                    optimalLength,
                    pennationAngle,
                    maximalForce,
                    muscle_velocity_max,
                    muscle_length,
                    musculoTendonVelocity,
                    activation_muscles,
                )
                Muscular_force = vertcat(Muscular_force, Muscular_force_current)
                """If there are Via point : """
                # for k, pts in enumerate(musc.position().pointsInGlobal()):
                #     data.append(pts.to_mx())

        return length_jacobian, Muscular_force

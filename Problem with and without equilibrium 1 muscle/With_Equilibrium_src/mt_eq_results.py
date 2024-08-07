from definition_func_class import *
from scipy.optimize import newton
import numpy as np
import matplotlib.pyplot as plt


class Musculotendon_equilibrium_results(BiorbdModel_musculotendon_equilibrium):
    def write_result_fic(
        self,
        path_result,
        musculoTendonLength,
        muscle_length_normalized,
        tendon_length_normalized,
        q_tab,
        qdot_tab,
        activation_tab,
        muscle_velocity_normalized,
        Fm_normalized,
        Ft_normalized,
        fpas,
        real_time_to_optimize,
    ):
        """
        Write the output of the optimization problem in a file.
        """
        with open(path_result, "w") as fichier:
            fichier.write("MusculoTendonLength\n")
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
            for vm in muscle_velocity_normalized:
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

            fichier.write("\nPassiveForce\n")
            for f in fpas:
                fichier.write(str(f))
                fichier.write(" ")

            fichier.write("\nRealTime\n")
            fichier.write(str(real_time_to_optimize))

    def calculate_manually_output(
        self,
        sol,
        path,
        LinearContinuous: bool = False,
        Collocation: bool = False,
    ):
        """
        Compute the forces applied to the muscles and tendons manually, by recalculating the muscle velocity with
        the output states and controls.
        """
        musculoTendonLength = []
        tendon_length_normalized = []
        muscle_length_normalized = []
        musculoTendonVelocity = []

        model = biorbd_eigen.Model(path)
        m = model

        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
        from bioptim import TimeAlignment

        states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
        controls = sol.stepwise_controls(to_merge=SolutionMerge.NODES)

        all_q = states["q"]
        all_qdot = states["qdot"]
        all_lm_normalized = states["lm_normalized"]
        all_activation = controls["muscles"]
        q_tab = []
        qdot_tab = []
        activation_tab = []
        muscle_velocity_normalized_tab = []
        Fm_normalized = []
        Ft_normalized = []
        Fm_c_normalized = []
        vm_c_normalized_tab = []
        fpas = []
        """
        Code which calculate the forces manually by recalculating the muscle velocity at each node. At the end, we can
        compare what we've calculated with the output of Bioptim.
        """

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
            muscle_length_normalized += [lm_normalized[0]]
            tendon_length_normalized += [
                (musculoTendonLength[-1] - muscle_length_normalized[-1] * optimalLength * np.cos(pennationAngle))
                / tendonSlackLength
            ]
            q_tab.append(q[0])
            qdot_tab.append(qdot[0])

        for activation in all_activation.T:
            activation_tab.append(activation[0])

        u = 0
        vm_c_normalized = sol.stepwise_controls(to_merge=SolutionMerge.NODES)["vm_c_normalized"].T[0][0]
        muscle_velocity_normalized_tab.append(vm_c_normalized)
        vm_c_normalized_tab.append(vm_c_normalized)
        f = activation_tab[0] * self.fact(muscle_length_normalized[0]) * self.fvm(
            muscle_velocity_normalized_tab[0]
        ) + self.fpas(muscle_length_normalized[0])
        Fm_normalized.append(f)
        Ft_normalized.append(self.ft(tendon_length_normalized[0]))

        Fm_c_normalized.append(f)
        for i in range(1, 201):
            vm_c_normalized = sol.stepwise_controls(to_merge=SolutionMerge.NODES)["vm_c_normalized"].T[u][0]
            vm_c_normalized_tab.append(vm_c_normalized)
            u = i
            p = 3 * i
            if i == 200:
                vm_c_normalized = sol.stepwise_controls(to_merge=SolutionMerge.NODES)["vm_c_normalized"].T[i][0]
                muscle_velocity_normalized_tab.append(vm_c_normalized)
                vm_c_normalized_tab.append(vm_c_normalized)
                f = activation_tab[i] * self.fact(muscle_length_normalized[p]) * self.fvm(
                    muscle_velocity_normalized_tab[-1]
                ) + self.fpas(muscle_length_normalized[p])
                Fm_normalized.append(f)
                Ft_normalized.append(self.ft(tendon_length_normalized[p]))
                fpas.append(self.fpas(muscle_length_normalized[p]))
                Fm_c_normalized.append(f)
            else:

                vm_c_normalized_prev = sol.stepwise_controls(to_merge=SolutionMerge.NODES)["vm_c_normalized"].T[i - 1][
                    0
                ]
                if Method_VM_Calculation.method_used_vm == Method_VM_Calculation.DAMPING_LINEAR:
                    previous_vm_normalized = vm_c_normalized_prev
                    alpha, beta = self.tangent_factor_calculation(previous_vm_normalized)
                    alpha, beta = 2.2, 1
                    vm_normalized = (
                        self.ft(tendon_length_normalized[p]) / np.cos(pennationAngle)
                        - self.fpas(muscle_length_normalized[p])
                        - beta * activation_tab[i] * self.fact(muscle_length_normalized[p])
                    ) / (alpha * activation_tab[i] * self.fact(muscle_length_normalized[p]) + damp)

                elif Method_VM_Calculation.method_used_vm == Method_VM_Calculation.DAMPING_NEWTON:
                    vm_normalized_init = vm_c_normalized_prev
                    vm_normalized = newton(
                        func=lambda vm_n: maximalForce * self.ft(tendon_length_normalized[p])
                        - maximalForce
                        * (
                            self.fpas(muscle_length_normalized[p])
                            + activation_tab[i] * self.fact(muscle_length_normalized[p]) * self.fvm(vm_n)
                            + self.fdamp(vm_n)
                        )
                        * np.cos(pennationAngle),
                        x0=vm_normalized_init,
                        tol=1e-8,
                        rtol=1e-8,
                    )
                muscle_velocity_normalized_tab.append(vm_normalized)
                Fm_normalized.append(
                    (
                        activation_tab[i]
                        * self.fact(muscle_length_normalized[p])
                        * self.fvm(muscle_velocity_normalized_tab[-1])
                        + self.fpas(muscle_length_normalized[p])
                        + self.fdamp(muscle_velocity_normalized_tab[-1])
                    )
                    * np.cos(pennationAngle)
                )
                fpas.append(self.fpas(muscle_length_normalized[p]))
                Ft_normalized.append(self.ft(tendon_length_normalized[p]))
                Fm_c_normalized.append(
                    (
                        activation_tab[i] * self.fact(muscle_length_normalized[p]) * self.fvm(vm_c_normalized_tab[-1])
                        + self.fpas(muscle_length_normalized[p])
                        + self.fdamp(vm_c_normalized_tab[-1])
                    )
                    * np.cos(pennationAngle)
                )
            # print(
            #     "Ft normalized:",
            #     Ft_normalized[-1],
            #     "Fm_pen normalized:",
            #     Fm_normalized[-1] * np.cos(pennationAngle),
            #     "Ecart relatif",
            #     np.abs(Fm_normalized[-1] - Ft_normalized[-1]) / Ft_normalized[-1] * 100,
            #     # "Fm_c_pen normalized:",
            #     # Fm_c_normalized[-1] * np.cos(pennationAngle),
            # )
        return (
            musculoTendonLength,
            muscle_length_normalized,
            tendon_length_normalized,
            muscle_velocity_normalized_tab,
            vm_c_normalized_tab,
            Fm_normalized,
            Fm_c_normalized,
            Ft_normalized,
            fpas,
            q_tab,
            qdot_tab,
            activation_tab,
        )

    def plot_results(
        self,
        musculoTendonLength,
        muscle_length_normalized,
        tendon_length_normalized,
        vm_normalized,
        vm_c_normalized,
        Fm_normalized,
        Fm_c_normalized,
        Ft_normalized,
        fpas,
        Time,
        N,
    ):
        plt.subplot(1, 3, 1)
        plt.plot(muscle_length_normalized, label="lm")
        plt.plot(tendon_length_normalized, label="lt")
        plt.plot(musculoTendonLength, label="lmt")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(vm_normalized, label="vm_normalized")
        plt.plot(vm_c_normalized, label="vm_c_normalized")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(Fm_normalized, label="Fm")
        plt.plot(Fm_c_normalized, label="Fm_c")
        plt.plot(Ft_normalized, label="Ft")
        plt.plot(fpas, label="fpas")
        plt.legend()
        plt.grid(True)

    def show_results(self, sol, path, N, Time, LinearContinuous=False, Collocation=False):
        (
            musculoTendonLength,
            muscle_length_normalized,
            tendon_length_normalized,
            vm_normalized,
            vm_c_normalized,
            Fm_normalized,
            Fm_c_normalized,
            Ft_normalized,
            fpas,
            q,
            qdot,
            activation,
        ) = self.calculate_manually_output(
            sol=sol,
            path=path,
            LinearContinuous=LinearContinuous,
            Collocation=Collocation,
        )

        self.plot_results(
            musculoTendonLength=musculoTendonLength,
            muscle_length_normalized=muscle_length_normalized,
            tendon_length_normalized=tendon_length_normalized,
            vm_normalized=vm_normalized,
            vm_c_normalized=vm_c_normalized,
            Fm_normalized=Fm_normalized,
            Fm_c_normalized=Fm_c_normalized,
            Ft_normalized=Ft_normalized,
            fpas=fpas,
            Time=Time,
            N=N,
        )

        self.write_result_fic(
            path_result=f"/home/mickael/Desktop/{Method_VM_Calculation.method_used_vm}.txt",
            musculoTendonLength=musculoTendonLength,
            muscle_length_normalized=muscle_length_normalized,
            tendon_length_normalized=tendon_length_normalized,
            q_tab=q,
            qdot_tab=qdot,
            activation_tab=activation,
            muscle_velocity_normalized=vm_normalized,
            Fm_normalized=Fm_normalized,
            Ft_normalized=Ft_normalized,
            fpas=fpas,
            real_time_to_optimize=sol.real_time_to_optimize,
        )

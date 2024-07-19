import numpy as np
import matplotlib.pyplot as plt


def read_doc(fichier_name: str):
    musculoTendonLength = []
    tendon_length = []
    muscle_length = []
    activation = []
    q = []
    qdot = []
    muscle_velocity = []
    tendon_force = []
    muscle_force = []
    musculoTendonLength_values = False
    muscle_length_values = False
    tendon_length_values = False
    activation_values = False
    q_values = False
    qdot_values = False
    real_time_value = False
    tendon_force_value = False
    muscle_force_value = False
    muscle_velocity_value = False

    with open(fichier_name, "r") as fic:
        content = fic.read()
    content = content.split()
    for elem in content:
        if elem == "MusculoTendonLength":
            musculoTendonLength_values = True
            muscle_length_values = False
            tendon_length_values = False
            activation_values = False
            q_values = False
            real_time_value = False
            qdot_values = False
            tendon_force_value = False
            muscle_force_value = False
            muscle_velocity_value = False
            continue

        if elem == "Muscle":
            musculoTendonLength_values = False
            muscle_length_values = True
            tendon_length_values = False
            activation_values = False
            q_values = False
            real_time_value = False
            qdot_values = False
            tendon_force_value = False
            muscle_force_value = False
            muscle_velocity_value = False
            continue

        if elem == "Tendon":
            musculoTendonLength_values = False
            muscle_length_values = False
            tendon_length_values = True
            activation_values = False
            q_values = False
            real_time_value = False
            qdot_values = False
            tendon_force_value = False
            muscle_force_value = False
            muscle_velocity_value = False
            continue

        if elem == "Q":
            musculoTendonLength_values = False
            muscle_length_values = False
            tendon_length_values = False
            activation_values = False
            q_values = True
            real_time_value = False
            qdot_values = False
            tendon_force_value = False
            muscle_force_value = False
            muscle_velocity_value = False
            continue

        if elem == "Qdot":
            musculoTendonLength_values = False
            muscle_length_values = False
            tendon_length_values = False
            activation_values = False
            real_time_value = False
            q_values = False
            qdot_values = True
            tendon_force_value = False
            muscle_force_value = False
            muscle_velocity_value = False
            continue

        if elem == "Activation":
            musculoTendonLength_values = False
            muscle_length_values = False
            tendon_length_values = False
            activation_values = True
            q_values = False
            real_time_value = False
            qdot_values = False
            tendon_force_value = False
            muscle_force_value = False
            muscle_velocity_value = False
            continue

        if elem == "RealTime":
            musculoTendonLength_values = False
            muscle_length_values = False
            tendon_length_values = False
            activation_values = False
            q_values = False
            real_time_value = True
            qdot_values = False
            tendon_force_value = False
            muscle_force_value = False
            muscle_velocity_value = False
            continue

        if elem == "TendonForce":
            musculoTendonLength_values = False
            muscle_length_values = False
            tendon_length_values = False
            activation_values = False
            q_values = False
            real_time_value = False
            qdot_values = False
            tendon_force_value = True
            muscle_force_value = False
            muscle_velocity_value = False
            continue

        if elem == "MuscleForce":
            musculoTendonLength_values = False
            muscle_length_values = False
            tendon_length_values = False
            activation_values = False
            q_values = False
            real_time_value = False
            qdot_values = False
            tendon_force_value = False
            muscle_force_value = True
            muscle_velocity_value = False
            continue

        if elem == "MuscleVelocity":
            musculoTendonLength_values = False
            muscle_length_values = False
            tendon_length_values = False
            activation_values = False
            q_values = False
            real_time_value = False
            qdot_values = False
            tendon_force_value = False
            muscle_force_value = False
            muscle_velocity_value = True
            continue

        if tendon_force_value:
            tendon_force.append(float(elem))
            continue

        if muscle_force_value:
            muscle_force.append(float(elem))
            continue

        if muscle_velocity_value:
            muscle_velocity.append(float(elem))
            continue

        if real_time_value:
            RealTime = float(elem)
            continue

        if musculoTendonLength_values:
            musculoTendonLength.append(float(elem))
            continue

        if muscle_length_values:
            muscle_length.append(float(elem))
            continue

        if tendon_length_values:
            tendon_length.append(float(elem))
            continue

        if activation_values:
            activation.append(float(elem))
            continue

        if q_values:
            q.append(float(elem))
            continue

        if qdot_values:
            qdot.append(float(elem))
            continue
    return (
        tendon_force,
        muscle_force,
        musculoTendonLength,
        muscle_length,
        tendon_length,
        muscle_velocity,
        activation,
        q,
        qdot,
        RealTime,
    )


def plot_methods(fichier_name_tab):
    fig, axs = plt.subplots(3, 6, figsize=(15, 10), layout="constrained")
    cpt = 0
    x = np.linspace(0, 0.2, 501)
    z = np.linspace(0, 0.2, 100)
    method = [
        "Without equilibrium",
        "With equilibrium Newton",
        "With equilibrium Linear",
    ]
    RealTime = [0, 0, 0]
    for i in range(0, 18, 6):
        (
            tendon_force,
            muscle_force,
            musculoTendonLength,
            muscle_length,
            tendon_length,
            muscle_velocity,
            activation,
            q,
            qdot,
            RealTime[cpt],
        ) = read_doc(fichier_name_tab[cpt])
        axs.flat[i].plot(x, musculoTendonLength, label="MusculoTendonLength", color="green")

        axs.flat[i].plot(x, muscle_length, label="Muscle Length", color="red")
        axs.flat[i].plot(x, tendon_length, label="Tendon Length", color="blue")
        axs.flat[i].set_title(f"Length: {method[cpt]}")
        axs.flat[i].set_xlabel("Time sec")
        axs.flat[i].set_ylabel("Length m")
        axs.flat[i].legend(loc="best")
        axs.flat[i].grid(True)

        axs.flat[i + 1].plot(z, activation, label="Activation", color="orange")
        axs.flat[i + 1].set_title(f"Activation: {method[cpt]}")
        axs.flat[i + 1].set_xlabel("Time sec")
        axs.flat[i + 1].set_ylabel("Activation Value")
        axs.flat[i + 1].legend(loc="best")
        axs.flat[i + 1].grid(True)

        axs.flat[i + 2].plot(x, q, label="Q", color="cyan")
        axs.flat[i + 2].set_title(f"Q: {method[cpt]}")
        axs.flat[i + 2].set_xlabel("Time sec")
        axs.flat[i + 2].set_ylabel("Q m")
        axs.flat[i + 2].legend(loc="best")
        axs.flat[i + 2].grid(True)

        axs.flat[i + 3].plot(x, qdot, label="Qdot", color="cyan")
        axs.flat[i + 3].set_title(f"Qdot: {method[cpt]}")
        axs.flat[i + 3].set_xlabel("Time sec")
        axs.flat[i + 3].set_ylabel("Qdot m/s")
        axs.flat[i + 3].legend(loc="best")
        axs.flat[i + 3].grid(True)

        axs.flat[i + 4].plot(z, muscle_velocity, label="Muscle Velocity", color="magenta")
        axs.flat[i + 4].set_title(f"Muscle Velocity: {method[cpt]}")
        axs.flat[i + 4].set_xlabel("Time sec")
        axs.flat[i + 4].set_ylabel("Vm m/s")
        axs.flat[i + 4].legend(loc="best")
        axs.flat[i + 4].grid(True)

        axs.flat[i + 5].plot(z, muscle_force, label="Muscle Force", color="red")
        axs.flat[i + 5].plot(z, tendon_force, label="Tendon Force", color="blue")
        axs.flat[i + 5].set_title(f"Forces: {method[cpt]}")
        axs.flat[i + 5].set_xlabel("Time sec")
        axs.flat[i + 5].set_ylabel("Forces in N")
        axs.flat[i + 5].legend(loc="best")
        axs.flat[i + 5].grid(True)
        cpt += 1
        plt.suptitle(f"Real Time of Calculation:{RealTime}.")
    plt.suptitle(
        f"Muscle: TRIlong for a simple elevation movement. Real Time Of Calculation in sec (same order): {RealTime[0]}, {RealTime[1]}, {RealTime[2]}."
    )
    plt.show()
    return


def main():
    fichier_tab = [
        "without_equilibrium.txt",
        "with_equilibrium_newton_method.txt",
        "with_equilibrium_linear_method.txt",
    ]

    plot_methods(fichier_tab)
    return


if __name__ == "__main__":
    main()

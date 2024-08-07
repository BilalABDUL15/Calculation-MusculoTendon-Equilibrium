import numpy as np
import matplotlib.pyplot as plt
from backup_def_ import *


def read_doc(fichier_name: str):
    # Dictionnaire pour stocker les valeurs booléennes
    states = {
        "PassiveForce": False,
        "Muscle": False,
        "Tendon": False,
        "Q": False,
        "Qdot": False,
        "Activation": False,
        "RealTime": False,
        "TendonForce": False,
        "MuscleForce": False,
        "MuscleVelocity": False,
    }

    # Initialisation des listes et variables
    passive_force = []
    muscle_length = []
    tendon_length = []
    activation = []
    q = []
    qdot = []
    tendon_force = []
    muscle_force = []
    muscle_velocity = []
    RealTime = None

    # Lecture du fichier
    with open(fichier_name, "r") as fic:
        content = fic.read().split()

    # Traitement des données
    for elem in content:
        if elem in states:
            for key in states:
                states[key] = False
            states[elem] = True
            continue

        # Traitement des valeurs numériques en fonction de l'état
        if states["TendonForce"]:
            tendon_force.append(float(elem))
        elif states["MuscleForce"]:
            muscle_force.append(float(elem))
        elif states["MuscleVelocity"]:
            muscle_velocity.append(float(elem))
        elif states["RealTime"]:
            RealTime = float(elem)
        elif states["PassiveForce"]:
            passive_force.append(float(elem))
        elif states["Muscle"]:
            muscle_length.append(float(elem))
        elif states["Tendon"]:
            tendon_length.append(float(elem))
        elif states["Activation"]:
            activation.append(float(elem))
        elif states["Q"]:
            q.append(float(elem))
        elif states["Qdot"]:
            qdot.append(float(elem))
    return (
        tendon_force,
        muscle_force,
        passive_force,
        muscle_length,
        tendon_length,
        muscle_velocity,
        activation,
        q,
        qdot,
        RealTime,
    )


def plot_methods(fichier_name_tab):
    nb_method = len(fichier_name_tab)
    fig, axs = plt.subplots(nb_method, 6, figsize=(15, 10), layout="constrained")
    cpt = 0
    x = np.linspace(0, 0.2, 101)
    z = np.linspace(0, 0.2, 200)
    y = np.linspace(0, 0.2, 201)

    method = [
        "Without equilibrium",
        "With equilibrium Newton",
        "With equilibrium Linear",
    ]
    RealTime = [0, 0, 0]
    for i in range(0, 6 * nb_method, 6):
        (
            tendon_force,
            muscle_force,
            passive_force,
            muscle_length,
            tendon_length,
            muscle_velocity,
            activation,
            q,
            qdot,
            RealTime[cpt],
        ) = read_doc(fichier_name_tab[cpt])
        # axs.flat[i].plot(musculoTendonLength, label="MusculoTendonLength", color="green")

        axs.flat[i].plot(muscle_length, label="lm normalized", color="red")
        axs.flat[i].plot(tendon_length, label="lt normalized", color="blue")
        axs.flat[i].set_title(f" {method[cpt]} Length")
        axs.flat[i].set_xlabel("Time sec")
        axs.flat[i].set_ylabel("Length normalized")
        axs.flat[i].legend(loc="best")
        axs.flat[i].grid(True)

        axs.flat[i + 1].plot(activation, label="Activation", color="orange")
        axs.flat[i + 1].set_title(f"Muscle Activation")
        axs.flat[i + 1].set_xlabel("Time sec")
        axs.flat[i + 1].set_ylabel("Activation Value")
        axs.flat[i + 1].legend(loc="best")
        axs.flat[i + 1].grid(True)

        axs.flat[i + 2].plot(q, label="Q", color="cyan")
        axs.flat[i + 2].set_title(f"Joint Position Q")
        axs.flat[i + 2].set_xlabel("Time sec")
        axs.flat[i + 2].set_ylabel("Q m")
        axs.flat[i + 2].legend(loc="best")
        axs.flat[i + 2].grid(True)

        axs.flat[i + 3].plot(qdot, label="Qdot", color="cyan")
        axs.flat[i + 3].set_title(f"Joint Velocity Qdot")
        axs.flat[i + 3].set_xlabel("Time sec")
        axs.flat[i + 3].set_ylabel("Qdot m/s")
        axs.flat[i + 3].legend(loc="best")
        axs.flat[i + 3].grid(True)

        axs.flat[i + 4].plot(muscle_velocity, label="Muscle Velocity normalized", color="magenta")
        axs.flat[i + 4].set_title(f"Muscle Velocity normalized")
        axs.flat[i + 4].set_xlabel("Time sec")
        axs.flat[i + 4].set_ylabel("Vm normalized")
        axs.flat[i + 4].legend(loc="best")
        axs.flat[i + 4].grid(True)

        axs.flat[i + 5].plot(muscle_force, label="Fm pen", color="red")
        axs.flat[i + 5].plot(passive_force, label="fpas pen", color="red", linestyle="dashed")
        axs.flat[i + 5].plot(tendon_force, label="Ft", color="blue")
        axs.flat[i + 5].set_title(f"Forces")
        axs.flat[i + 5].set_xlabel("Time sec")
        axs.flat[i + 5].set_ylabel("Forces normalized")
        axs.flat[i + 5].legend(loc="best")
        axs.flat[i + 5].grid(True)
        cpt += 1
        plt.suptitle(f"Real Time of Calculation:{RealTime}.")
    plt.suptitle(f"Real Time Of Calculation in sec (same order): {RealTime[0]}, {RealTime[1]}, {RealTime[2]}.")
    plt.show()
    return


def main():
    fichier_tab = [
        "/home/mickael/Desktop/without_equilibrium.txt",
        f"/home/mickael/Desktop/{Method_VM_Calculation.DAMPING_NEWTON}.txt",
        f"/home/mickael/Desktop/{Method_VM_Calculation.DAMPING_LINEAR}.txt",
    ]

    plot_methods(fichier_tab)
    return


if __name__ == "__main__":
    main()

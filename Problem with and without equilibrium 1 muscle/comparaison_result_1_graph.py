import numpy as np
import matplotlib.pyplot as plt
from backup_def_ import *


def read_doc(fichier_name: str):

    # Dictionnaire pour stocker les valeurs booléennes
    states = {
        "MusculoTendonLength": False,
        "Muscle": False,
        "Tendon": False,
        "Q": False,
        "Qdot": False,
        "Activation": False,
        "RealTime": False,
        "TendonForce": False,
        "MuscleForce": False,
        "PassiveForce": False,
        "MuscleVelocity": False,
    }

    # Initialisation des listes et variables
    musculoTendonLength = []
    muscle_length = []
    tendon_length = []
    activation = []
    q = []
    qdot = []
    tendon_force = []
    muscle_force = []
    muscle_velocity = []
    passive_force = []
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
        elif states["PassiveForce"]:
            passive_force.append(float(elem))
        elif states["MuscleVelocity"]:
            muscle_velocity.append(float(elem))
        elif states["RealTime"]:
            RealTime = float(elem)
        elif states["MusculoTendonLength"]:
            musculoTendonLength.append(float(elem))
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
        q,
        qdot,
        activation,
        muscle_velocity,
        muscle_length,
        tendon_length,
        tendon_force,
        muscle_force,
        passive_force,
    )


def plot_methods(fichier_name_tab, Time, n_shooting):
    method = [
        "Without equilibrium",
        "With equilibrium Newton",
        "With equilibrium Linear",
    ]
    curve_charact = [
        ["Joint Position Q", ["-0.1", "0.1"]],
        ["Joint Velocity Qdot", ["-1", "1"]],
        ["Muscle Activation", ["-.01", "0.2"]],
        ["Muscle velocity normalized", ["-.25", ".25"]],
        ["Muscle length normalized", ["0", "2"]],
        ["Tendon length normalized", [".99", "1.05"]],
        ["Tendon Force normalized", ["-0.1", "1"]],
        ["Muscle Force normalized pennated", ["-0.1", "1"]],
        ["Passive Force normalized pennated", ["-0.1", "1"]],
    ]
    forces = {
        "Muscle Force normalized pennated": "Fm",
        "Tendon Force normalized": "Ft",
        "Passive Force normalized pennated": "fpas",
    }

    temp = []
    tab = [[] for i in range(len(curve_charact))]
    curve = {}

    for i in range(len(fichier_name_tab)):
        temp.append(read_doc(fichier_name_tab[i]))

    for i in range(len(temp[0])):
        for j in range(3):
            tab[i].append(temp[j][i])

    for i in range(len(tab)):
        curve[curve_charact[i][0]] = tab[i]

    k = 0
    first_ite = True
    cpt_fig = 0
    cpt_col = 0
    row, col = 1, 2
    index_row = 0
    tab_fig = [plt.subplots(row, col, layout="constrained") for _ in range(4)]
    for elem in curve:
        # fig, axs = plt.subplots(nb_method, col, figsize=(15, 10), layout="constrained")
        # index_row, index_col = divmod(k, col)
        fig, axs = tab_fig[cpt_fig]
        if elem != "Passive Force normalized pennated":
            index_row = k % col
        else:
            pass
        for i in range(len(fichier_name_tab)):
            if method[i] == "Without equilibrium":
                linestyle = "solid"
                color = "green"
            elif method[i] == "With equilibrium Newton":
                color = "red"
                linestyle = "dashed"
            elif method[i] == "With equilibrium Linear":
                color = "blue"
                linestyle = "dashdot"
            x = np.linspace(0, Time, len(curve[elem][i]))
            if elem in forces:
                if elem == "Passive Force normalized pennated":
                    axs[index_row].plot(
                        x, curve[elem][i], label=f"{forces[elem]}: {method[i]}", linestyle=linestyle, color="black"
                    )
                else:
                    axs[index_row].plot(
                        x, curve[elem][i], label=f"{forces[elem]}: {method[i]}", linestyle=linestyle, color=color
                    )
                axs[index_row].legend()
            else:
                axs[index_row].plot(x, curve[elem][i], label=f"{method[i]}", linestyle=linestyle, color=color)
            axs[index_row].set_title(elem)
            axs[index_row].set_xlabel("Time (s)")
            if elem == "Joint Position Q":
                axs[index_row].set_ylabel("Joint Position (m)")
            if elem == "Joint Velocity Qdot":
                axs[index_row].set_ylabel("Joint Velocity (m/s)")
            axs[index_row].set_xlim(0, Time)
            axs[index_row].set_ylim(float(curve_charact[k][1][0]), float(curve_charact[k][1][1]))
            if first_ite:
                axs[index_row].legend()
        first_ite = False
        if elem == "Joint Velocity Qdot" or elem == "Muscle velocity normalized" or elem == "Tendon length normalized":
            cpt_fig += 1
            first_ite = True
        cpt_col += 1
        k += 1
    plt.show()
    return


def main():
    fichier_tab = [
        "/home/mickael/Desktop/without_equilibrium.txt",
        f"/home/mickael/Desktop/{Method_VM_Calculation.DAMPING_NEWTON}.txt",
        f"/home/mickael/Desktop/{Method_VM_Calculation.DAMPING_LINEAR}.txt",
    ]
    Time = 0.2  # Time simulation in sec
    N = 100  # Number of shooting points

    plot_methods(fichier_name_tab=fichier_tab, Time=Time, n_shooting=N)
    return


if __name__ == "__main__":
    main()

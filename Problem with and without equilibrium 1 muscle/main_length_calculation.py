from With_Equilibrium_src import with_equilibrium_calculation
from Without_Equilibrium_src import without_equilibrium_calculation
from With_Equilibrium_src.definition_func_class import Method_VM_Calculation
import comparaison_resultats
import comparaison_result_1_graph


def Method_Calculation(
    choice: str,
    path: str,
    n_shooting: int,
    Time: float,
    q0_init: float,
    q0_final: float,
    LinearContinuous: bool = False,
    Collocation: bool = False,
    Method_VM_Calculation_value: Method_VM_Calculation = Method_VM_Calculation.DAMPING_NEWTON,
):
    """
    Choose which method of calculation we want to use. Calculation with or without Equilibrium
    If Without equilibrum, the optimization problem will be solved with a rigid tendon.

    If with equilibrium method selected, the optimization problem will be solved with a non rigid tendon.

    The method of calculation method can be selected between two choices:
        1 - Damping Newton : This method will calculate the muscle velocity intern with a gradient descent algorithm on
        the substraction between the Muscle Force and the Tendon Force.
        2 - Linear : This method will calculate the muscle velocity intern by linearizing the muscle velocity force
        around the last muscle velocity control and integrating the differential equation of the muscle velocity created
        by substractive of the Muscle Force and the Tendon Force.

    ------------------------------------------------------------------------------------------------
    Parameters:
        path: path to the bioMod file,
        n_shooting: number of shooting points,
        Time: duration of the simulation,
        q0_init: initial configuration,
        q0_final: final configuration,
        LinearContinuous: if True, use linear continuous
        Collocation: if True, use collocation
        Method_VM_Calculation_value: method to calculate the muscle velocity intern when the choice
        with equilibrium choice has been selected.
        By default, Damping Newton is selected because this method is more robust and accurate.
    ------------------------------------------------------------------------------------------------

    """
    if choice == "with_equilibrium":
        return with_equilibrium_calculation.problem_resolution(
            n_shooting=n_shooting,
            path=path,
            Time_sim=Time,
            LinearContinuous=LinearContinuous,
            Collocation=Collocation,
            q0_init=q0_init,
            q0_final=q0_final,
            Method_VM_Calculation_value=Method_VM_Calculation_value,
        )
    elif choice == "without_equilibrium":
        return without_equilibrium_calculation.problem_resolution(
            n_shooting=n_shooting,
            path=path,
            Time_sim=Time,
            LinearContinuous=LinearContinuous,
            Collocation=Collocation,
            q0_init=q0_init,
            q0_final=q0_final,
        )
    else:
        raise ValueError("Invalid choice")


def main():

    Method_Calculation(
        choice="with_equilibrium",
        path="models/test.bioMod",
        n_shooting=100,
        Time=0.2,
        LinearContinuous=True,
        Collocation=False,
        q0_init=-0.22,
        q0_final=-0.31,
        Method_VM_Calculation_value=Method_VM_Calculation.DAMPING_LINEAR,
    )

    comparaison_resultats.main()
    comparaison_result_1_graph.main()

    return


if __name__ == "__main__":
    main()

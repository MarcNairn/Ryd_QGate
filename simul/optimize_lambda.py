"""
In this notebook we simulate the dynamics of three Rydberg atoms to efficiently generate
triply ground-to-excited transitions governed by a collective Rabi frequency (transition amplitude) $\overline{\Omega}$
"""
import warnings

from functools import singledispatch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from qutip import *

tqdm.pandas()

# plt.style.use("simul/paperstyle.mplstyle")

# %% Define the 3x3 Hamiltonian matrix
a = basis(3, 0)
b = basis(3, 1)
e = basis(3, 2)


def Hamiltonian_3x3(series: pd.Series):
    # add off-diagonals
    H_3x3 = series['Ω_a'] * e * a.dag()
    H_3x3 += series['Ω_b'] * e * b.dag()
    H_3x3 += H_3x3.dag()

    # add the diagonals
    H_3x3 += -series['δ'] * a.proj()
    H_3x3 += series['δ'] * b.proj()
    H_3x3 += series['Δ'] * e.proj()

    return H_3x3


# %% Functions to analyze a parameter set

def project_3x3_to_2x2(H_3x3):
    P = a.proj() + b.proj()
    Q = qeye(3) - P
    H_traced = P * H_3x3 * P - P * H_3x3 * Q * (Q * H_3x3 * Q + 10e-10 * qeye(3)).inv() * Q * H_3x3 * P
    return Qobj(np.array([[H_traced[0, 0], H_traced[0, 1]], [H_traced[1, 0], H_traced[1, 1]]]))


def calc_Ωeff_Δeff(series: pd.Series):
    """
    Calculate the effective Rabi frequency and detuning after elimination
    """
    H_2x2 = project_3x3_to_2x2(Hamiltonian_3x3(series)).data.to_array()
    Ω_eff = np.abs(H_2x2[0, 1]).real
    Δ_eff = (H_2x2[1, 1] - H_2x2[0, 0]).real
    return pd.Series([Ω_eff, Δ_eff, np.abs(Ω_eff / Δ_eff).real], index=['Ω_eff', 'Δ_eff', 'Ω_eff/Δ_eff'],
                     dtype=np.float64)


def calc_adiabaticity_ab(dataframe: pd.DataFrame, force: bool = False):
    """
    In order to be able to eliminate the intermediate states |a> and |b> we need to have the following conditions:
    δ1 >> Ω1, δ3 >> Ω3.
    If these conditions are met, the result should be close to 0.
    :returns max of Ω1/|δ1| and Ω3/|δ3|
    """
    if dataframe.empty:
        warnings.warn("calc_adiabaticity_ab: Empty dataframe")
        return
    if force == False and 'adiabaticity_ab' in dataframe.columns:
        return
    divis = pd.DataFrame()
    divis['δ/Δ'] = dataframe['δ'] / np.abs(dataframe['Δ'])
    divis['Ω_a/Δ'] = dataframe['Ω_a'] / np.abs(dataframe['δ'])
    divis['Ω_b/Δ'] = dataframe['Ω_b'] / np.abs(dataframe['δ'])
    dataframe['adiabaticity_ab'] = divis.max(axis=1)


def calc_analytic_eff(dataframe: pd.DataFrame):
    if dataframe.empty:
        warnings.warn("calc_analytic_eff: Empty dataframe")
        return
    if ('Ω_ana_eff' in dataframe.columns
            and 'Δ_ana_eff' in dataframe.columns
            and 'Ω_ana_eff/Δ_ana_eff' in dataframe.columns):
        return
    dataframe['Ω_ana_eff'] = dataframe['Ω_a'] * np.conj(dataframe['Ω_b']) / dataframe['Δ']
    dataframe['Δ_ana_eff'] = (2 * dataframe['δ']
                              + dataframe['Ω_b'].abs() ** 2 / dataframe['Δ']
                              - dataframe['Ω_a'].abs() ** 2 / dataframe['Δ'])
    dataframe['Ω_ana_eff/Δ_ana_eff'] = dataframe['Ω_ana_eff'] / dataframe['Δ_ana_eff']


# %% Parameter normalisation to define timescales
punit = 1

"""
Parameters which work for 
"""

# Define the parameter space with lower and upper bounds
free_params_bounds = {
    'Ω_a': [0 * punit, 100 * punit],
    'Ω_b': [0 * punit, 100 * punit],
    'δ': [-1000 * punit, 1000 * punit],
    'Δ': [-1000 * punit, 1000 * punit],
}


def add_new_params(dataframe: pd.DataFrame, length: int):
    # generate the parameters set
    free_params_set = {key: np.random.uniform(*value, length) for key, value in free_params_bounds.items()}
    new_df = pd.DataFrame(free_params_set)

    # remove all detunings which are too close to zero
    delta_threshold = 1e-7
    new_df = new_df[(new_df['δ'].abs() > delta_threshold)
                    & (new_df['Δ'].abs() > delta_threshold)]

    # Calculate some meaningful quantities for a sample
    new_df['δ/Δ'] = new_df['δ'] / new_df['Δ']
    new_df['Ω_a/Δ'] = new_df['Ω_a'] / new_df['Δ']
    new_df['Ω_b/Δ'] = new_df['Ω_b'] / new_df['Δ']

    # FIXME could be done in a more efficient way
    calc_adiabaticity_ab(new_df)
    if not new_df.empty:
        new_df = new_df[new_df['adiabaticity_ab'] < 1e-1]

    new_df[['Ω_eff', 'Δ_eff', 'Ω_eff/Δ_eff']] = new_df.apply(lambda row: calc_Ωeff_Δeff(row), axis=1)
    calc_analytic_eff(new_df)
    new_df['Ω_eff/Ω_ana_eff'] = new_df['Ω_eff'] / new_df['Ω_ana_eff']
    new_df['Δ_eff/Δ_ana_eff'] = new_df['Δ_eff'] / new_df['Δ_ana_eff']

    longer_df = pd.concat([dataframe, new_df], ignore_index=True)
    return longer_df


df = pd.DataFrame()
df = add_new_params(df, int(1e4))
# %% Calculate some statistics
while (len(df) < 1e4):
    df = add_new_params(df, int(1e3))
    print(f"Length of dataframe: {len(df)}")
df.sort_values('Ω_eff/Δ_eff', inplace=True, ignore_index=True, ascending=False)

# %% Plot pairplot of the parameters
sns.pairplot(df,
             hue='Ω_eff/Δ_eff',
             vars=['Ω_eff/Δ_eff', 'Ω_ana_eff/Δ_ana_eff', 'adiabaticity_ab',
                   'Δ', 'Ω_eff/Ω_ana_eff', 'Δ_eff/Δ_ana_eff'],
             diag_kind=None,
             kind='scatter',
             # corner=True,
             # plot_kws={'alpha': 0.8},
             )
plt.show()

# %% Test the projection using time evolution comparison

initial_state = ggg_state

# Time evolution
times = np.linspace(0, 240 * punit, 1000)  # Time range for simulation

# Solve the master equation
result = mesolve(Hamiltonian_3x3(**df.iloc[0].to_dict()), initial_state, times,
                 c_ops=[], e_ops=[], options={'nsteps': 1e8})
# Extract expectation values
P_ggg = expect(ggg_state * ggg_state.dag(), result.states)  # Probability of being in the ground state |gg>
P_rrr = expect(rrr_state * rrr_state.dag(), result.states)  # Probability of being in the Rydberg state |rr>

P_rr = expect(grr_state * grr_state.dag() + rgr_state * rgr_state.dag() * rrg_state * rrg_state.dag(), result.states)
P_r = expect(ggr_state * ggr_state.dag() + grg_state * grg_state.dag() * rgg_state * rgg_state.dag(), result.states)

# Plot the probabilities and coherences
plt.figure()
plt.plot(times, P_ggg, label=r'$\langle \psi | ggg | \psi \rangle$ (Ground State)')
plt.plot(times, P_rrr, label=r'$\langle \psi | rrr | \psi \rangle$ (Rydberg State)')
# plt.plot(times, P_rr, label=r'$\langle \psi | P_{rr} | \psi \rangle$ (Doubly excited)' )
# plt.plot(times, P_r, label=r'$\langle \psi | P_r | \psi \rangle$ (Singly excited)' )
plt.xlabel('Time (arb. units)')
plt.ylabel('Probability')
plt.legend()
plt.tight_layout(pad=0.2)
plt.show()

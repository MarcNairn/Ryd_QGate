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

# %% As in the 2 atom case, we can reduce the problem to work only with the collective states spanned by the individual
# ground/excited states in the basis $$\{|rrr,n+1\rangle, |grr,n\rangle, |rgr,n+2\rangle, |rrg,n\rangle, |rgg,n+1\rangle, |grg,n-1\rangle, |ggr,n+1\rangle, |ggg,n\rangle\}^T$$

# Hamiltonian and detunings for individual atoms

@singledispatch
def Δ_1(n_offset: int, **kwargs):
    return (kwargs['δ1'] - kwargs['Δa']
            + kwargs['Ω1'] ** 2 / kwargs['δ1']
            - kwargs['g_a'] ** 2 * (kwargs['n'] + n_offset + 1) / kwargs['δ1']
            + kwargs['g_b'] ** 2 * (kwargs['n'] + n_offset + 2) / kwargs['Δb']
            )


@Δ_1.register(pd.DataFrame)
def _(dataframe: pd.DataFrame, force: bool = False):
    if force == False and 'Δ_1' in dataframe.columns:
        return
    dataframe['Δ_1'] = (dataframe['δ1'] - dataframe['Δa']
                        + dataframe['Ω1'] ** 2 / dataframe['δ1']
                        - dataframe['g_a'] ** 2 * (dataframe['n'] + 1) / dataframe['δ1']
                        + dataframe['g_b'] ** 2 * (dataframe['n'] + 2) / dataframe['Δb']
                        )


@singledispatch
def Δ_3(n_offset: int, **kwargs):
    return (kwargs['δ3']
            + kwargs['Δb'] +
            kwargs['Ω3'] ** 2 / kwargs['δ3']
            - kwargs['g_b'] ** 2 * (kwargs['n'] + n_offset) / kwargs['δ3']
            + kwargs['g_a'] ** 2 * (kwargs['n'] + n_offset - 1) / kwargs['Δa'])


@Δ_3.register(pd.DataFrame)
def _(dataframe: pd.DataFrame, force: bool = False):
    if force == False and 'Δ_3' in dataframe.columns:
        return
    dataframe['Δ_3'] = (dataframe['δ3'] + dataframe['Δb'] +
                        dataframe['Ω3'] ** 2 / dataframe['δ3']
                        - dataframe['g_b'] ** 2 * (dataframe['n']) / dataframe['δ3']
                        + dataframe['g_a'] ** 2 * (dataframe['n'] - 1) / dataframe['Δa'])


# def H_atom1(**kwargs):
#    # 2x2 matrix for the atom's Hamiltonian
#    sqrt_n1 = np.sqrt(kwargs['n'] + 1)
#
#    H = np.array([[Delta1(**kwargs), kwargs['Ω1'] * kwargs['g_a'] / kwargs['δ1'] * sqrt_n1],
#                  [kwargs['Ω1'] * kwargs['g_a'] / kwargs['δ1'] * sqrt_n1, 0]])
#    return Qobj(H)
#
#
# def H_atom2(**kwargs):
#    # 2x2 matrix for the atom's Hamiltonian
#    sqrt_n = np.sqrt(kwargs['n'])
#    H = np.array([[Delta2(**kwargs), kwargs['Ω2'] * kwargs['g_b'] / kwargs['δ2'] * sqrt_n],
#                  [kwargs['Ω2'] * kwargs['g_b'] / kwargs['δ2'] * sqrt_n, 0]])
#    return Qobj(H)
#
#
# def H_atom3(**kwargs):
#    # 2x2 matrix for the atom's Hamiltonian
#    sqrt_n1 = np.sqrt(kwargs['n'] + 1)
#    H = np.array([[Delta3(**kwargs), kwargs['Ω3'] * kwargs['g_a'] / kwargs['δ3'] * sqrt_n1],
#                  [kwargs['Ω3'] * kwargs['g_a'] / kwargs['δ3'] * sqrt_n1, 0]])
#
#    return Qobj(H)

@singledispatch
def Ω_1t(n_offset: int, **kwargs):
    return kwargs['Ω1'] * kwargs['g_a'] * np.sqrt(kwargs['n'] + n_offset + 1) / kwargs['δ1']


@Ω_1t.register(pd.DataFrame)
def _(dataframe: pd.DataFrame, force: bool = False):
    if force == False and 'Ω_1t' in dataframe.columns:
        return
    dataframe['Ω_1t'] = dataframe['Ω1'] * dataframe['g_a'] * np.sqrt(dataframe['n'] + 1) / dataframe['δ1']


@singledispatch
def Ω_3t(n_offset: int, **kwargs):
    return kwargs['Ω3'] * kwargs['g_b'] * np.sqrt(kwargs['n'] + n_offset) / kwargs['δ3']


@Ω_3t.register(pd.DataFrame)
def _(dataframe: pd.DataFrame, force: bool = False):
    if force == False and 'Ω_3t' in dataframe.columns:
        return
    dataframe['Ω_3t'] = dataframe['Ω3'] * dataframe['g_b'] * np.sqrt(dataframe['n']) / dataframe['δ3']


# %% define the Deltas

def Delta_ggg(**kwargs):
    return (
            -kwargs['Ω2'] ** 2 / kwargs['δ2']
            - Ω_1t(0, **kwargs) ** 2 / Δ_1(0, **kwargs)
            - Ω_3t(0, **kwargs) ** 2 / Δ_3(0, **kwargs)
    )


# def Delta_ggr(**kwargs):
#    return Δ_3(**kwargs)
#
#
# def Delta_grg(**kwargs):
#    return δ_2(**kwargs)
#
#
# def Delta_rgg(**kwargs):
#    return Δ_1(**kwargs)


def Delta_grr(**kwargs):
    return Δ_3(0, **kwargs) + kwargs['δ2']


def Delta_rgr(**kwargs):
    return Δ_1(-1, **kwargs) + Δ_3(1, **kwargs)


def Delta_rrg(**kwargs):
    return Δ_1(0, **kwargs) + kwargs['δ2']


def Delta_rrr(**kwargs):
    return Delta_ggg(**kwargs) + Delta_rrg_prime(**kwargs) + Delta_rgr_prime(**kwargs) + Delta_grr_prime(**kwargs)


def Ω_13t(n_offset, **kwargs):
    return (Ω_1t(n_offset, **kwargs) * Ω_3t(n_offset, **kwargs)
            * (Δ_1(n_offset, **kwargs) + Δ_3(n_offset, **kwargs)) / (
                    Δ_1(n_offset, **kwargs) * Δ_3(n_offset, **kwargs)))


def Ω_12t(n_offset, **kwargs):
    return Ω_1t(n_offset, **kwargs) * kwargs['Ω2'] * (Δ_1(n_offset, **kwargs) + kwargs['δ2']) / (
            Δ_1(n_offset, **kwargs) * kwargs['δ2'])


def Ω_23t(n_offset, **kwargs):
    return Ω_3t(n_offset, **kwargs) * kwargs['Ω2'] * (Δ_3(n_offset, **kwargs) + kwargs['δ2']) / (
            Δ_3(n_offset, **kwargs) * kwargs['δ2'])


def Delta_rrg_prime(**kwargs):
    return (Delta_rrg(**kwargs)
            - kwargs['Ω2'] ** 2 / Δ_1(0, **kwargs)
            - Ω_1t(0, **kwargs) ** 2 / kwargs['δ2']
            )


def Delta_rgr_prime(**kwargs):
    return (Delta_rgr(**kwargs)
            - Ω_1t(-1, **kwargs) * Ω_1t(0, **kwargs) / Δ_3(0, **kwargs)
            - Ω_3t(0, **kwargs) * Ω_3t(1, **kwargs) / Δ_1(0, **kwargs)
            )


def Delta_grr_prime(**kwargs):
    return (Delta_grr(**kwargs)
            - kwargs['Ω2'] ** 2 / Δ_3(0, **kwargs)
            - Ω_3t(0, **kwargs) * Ω_3t(1, **kwargs) / kwargs['δ2']
            )


# %% Define the 5x5 Hamiltonian matrix
ggg_state = basis(5, 0)
rrg_state = basis(5, 1)
rgr_state = basis(5, 2)
grr_state = basis(5, 3)
rrr_state = basis(5, 4)


def Hamiltonian_5x5(series: pd.Series):
    kwargs = series.to_dict()

    H_5x5 = qzero(5)
    # add lower triangular part from bottom
    # 1st row
    H_5x5 += Ω_3t(1, **kwargs) * rrr_state * rrg_state.dag()
    H_5x5 += kwargs['Ω2'] * rrr_state * rgr_state.dag()
    H_5x5 += Ω_1t(-1, **kwargs) * rrr_state * grr_state.dag()
    # 2nd row
    H_5x5 += -kwargs['Ω2'] * Ω_1t(-1, **kwargs) / kwargs['Δ_3'] * grr_state * rgr_state.dag()
    H_5x5 += -kwargs['Ω_1t'] * Ω_3t(1, **kwargs) / kwargs['δ2'] * grr_state * rrg_state.dag()
    H_5x5 += -Ω_23t(0, **kwargs) * grr_state * ggg_state.dag()
    # 3rd row
    H_5x5 += -kwargs['Ω2'] * Ω_3t(0, **kwargs) / kwargs['Δ_1'] * rgr_state * rrg_state.dag()
    H_5x5 += -Ω_13t(0, **kwargs) * rgr_state * ggg_state.dag()
    # 4th row
    H_5x5 += -Ω_12t(0, **kwargs) * rrg_state * ggg_state.dag()

    # add upper triangular part
    H_5x5 += H_5x5.dag()

    # add diagonal part
    H_5x5 += Delta_ggg(**kwargs) * ggg_state.proj()
    H_5x5 += Delta_rrg_prime(**kwargs) * rrg_state.proj()
    H_5x5 += Delta_rgr_prime(**kwargs) * rgr_state.proj()
    H_5x5 += Delta_grr_prime(**kwargs) * grr_state.proj()
    H_5x5 += Delta_rrr(**kwargs) * rrr_state.proj()

    # FIXME set zero energy to the ground state?
    # H_5x5 -= H_5x5[0, 0] * qeye(5)

    return H_5x5


# %% Functions to analyze a parameter set

def project_5x5_to_2x2(H_5x5):
    P = rrr_state.proj().to('CSR') + ggg_state.proj().to('CSR')
    Q = qeye(5).to('CSR') - P
    H_traced = P * H_5x5 * P - P * H_5x5 * Q * (Q * H_5x5 * Q + 10e-10 * qeye(5)).inv(sparse=True) * Q * H_5x5 * P
    return Qobj(np.array([[H_traced[0, 0], H_traced[0, -1]], [H_traced[-1, 0], H_traced[-1, -1]]]))


def calc_ΩR_ΔR(series: pd.Series):
    """
    Calculate the effective Rabi frequency and detuning for the collective states |ggg> and |rrr>
    :returns: |Ω_R|/|Δ_R|
    """
    H_2x2 = project_5x5_to_2x2(Hamiltonian_5x5(series)).data.to_array()
    ΩR = H_2x2[0, 1].real
    ΔR = (H_2x2[0, 0] - H_2x2[1, 1]).real
    return pd.Series([ΩR, ΔR, np.abs(ΩR / ΔR).real], index=['ΩR', 'ΔR', 'ΩR/ΔR'], dtype=np.float64)


def calc_adiabaticity_ab_eff_Rabi(dataframe: pd.DataFrame, force: bool = False):
    """
    In order to be able to eliminate the intermediate states |a> and |b> we need to have the following conditions:
    δ1 >> Ω1, δ3 >> Ω3.
    If these conditions are met, the result should be close to 0.
    :returns max of Ω1/|δ1| and Ω3/|δ3|
    """
    if dataframe.empty:
        warnings.warn("calc_adiabaticity_ab_eff_Rabi: Empty dataframe")
        return
    if force == False and 'adiabaticity_ab_eff_Rabi' in dataframe.columns:
        return
    divis = pd.DataFrame()
    divis['Ω1/δ1'] = dataframe['Ω1'] / np.abs(dataframe['δ1'])
    divis['Ω3/δ3'] = dataframe['Ω3'] / np.abs(dataframe['δ3'])
    dataframe['adiabaticity_ab_eff_Rabi'] = divis.max(axis=1)


def calc_adiabaticity_ab_cavity_coupling(dataframe: pd.DataFrame, force: bool = False):
    """
    In order to be able to eliminate the intermediate states |a> and |b> we need to have the following conditions:
    Δa >> ga, Δb >> gb.
    If these conditions are met, the result should be close to 0.
    :returns max of ga/|Δa| and gb/|Δb|
    """
    if dataframe.empty:
        warnings.warn("calc_adiabaticity_ab_cavity_coupling: Empty dataframe")
        return
    if force == False and 'adiabaticity_ab_cavity_coupling' in dataframe.columns:
        return
    divis = pd.DataFrame()
    divis['da/ga'] = dataframe['g_a'] / dataframe['Δa'].abs()
    divis['db/gb'] = dataframe['g_b'] / dataframe['Δb'].abs()
    dataframe['adiabaticity_ab_cavity_coupling'] = divis.max(axis=1)


def calc_adiabaticity_one_r_eff_Rabi(dataframe: pd.DataFrame, force: bool = False):
    """
    In order to be able to eliminate the states with one excitation we need to have the following conditions:
    Δ_1 >> Ω_1t, δ2 >> Ω2, Δ_3 >> Ω_3t.
    :return: max of |Ω_1t|/|Δ_1|, Ω2/|δ2|, |Ω_3t|/|Δ_3|
    """
    if dataframe.empty:
        warnings.warn("calc_adiabaticity_one_r_eff_Rabi: Empty dataframe")
        return
    if force == False and 'adiabaticity_one_r_eff_Rabi' in dataframe.columns:
        return
    Ω_1t(dataframe, force=force)
    Ω_3t(dataframe, force=force)
    Δ_1(dataframe, force=force)
    Δ_3(dataframe, force=force)

    divis = pd.DataFrame()
    divis['Ω_1t/Δ_1'] = dataframe['Ω_1t'].abs() / dataframe['Δ_1'].abs()
    divis['Ω2/δ2'] = dataframe['Ω2'].abs() / dataframe['δ2'].abs()
    divis['Ω_3t/Δ_3'] = dataframe['Ω_3t'].abs() / dataframe['Δ_3'].abs()
    dataframe['adiabaticity_one_r_eff_Rabi'] = divis.max(axis=1)


# %% Parameter normalisation to define timescales
punit = 1

"""
Parameters which work for 2 atoms (frequencies, detunings, and couplings)

Ω1, Ω2, Ω3 = 56.50 * punit, 60.0 * punit, 112.0 * punit  # Rabi frequencies for atoms
δ1, δ2, δ3 = 663.8 * punit, -742.0 * punit, 716.0 * punit  # Detunings for each atom
Δa, Δb = 722.0 * punit, 800.0 * punit  # Frequency shifts
g_a, g_b = 9.5 * punit, 10.0 * punit  # Coupling strengths
n = 10  # Number of photons in the cavity


Parameters for 3 atoms, which are not too bad:
Ω1                                  32.323232
Ω2                                  37.373737
Ω3                                  22.222222
δ1                                 878.887879
δ2                                 878.887879
δ3                                 313.231313
Δa                                 838.483838
Δb                                -979.697980
g_a                                 10.101010
g_b                                 78.787879
n                                   10.000000
adiabaticity_ab_cavity_coupling      0.080421
adiabaticity_ab_eff_Rabi             0.070945
Ω_1t                                 1.232090
Ω_3t                                17.675908
Δ_1                                -35.718190
Δ_3                               -861.972130
adiabaticity_one_r_eff_Rabi          0.042524
ΩR_ΔR                                0.007356
"""

# Define the parameter space with lower and upper bounds
free_params_bounds = {
    'Ω1': [0 * punit, 100 * punit],
    'Ω2': [0 * punit, 100 * punit],
    'Ω3': [0 * punit, 100 * punit],
    'δ1': [-1000 * punit, 1000 * punit],
    'δ2': [-1000 * punit, 1000 * punit],
    'δ3': [-1000 * punit, 1000 * punit],
    'Δa': [-1000 * punit, 1000 * punit],
    'Δb': [-1000 * punit, 1000 * punit],
    'g_a': [0 * punit, 100 * punit],
    'g_b': [0 * punit, 100 * punit],
    'n': [10, 10],
}


def add_new_params(dataframe: pd.DataFrame, length: int):
    # generate the parameters set
    free_params_set = {key: np.random.uniform(*value, length) for key, value in free_params_bounds.items()}
    new_df = pd.DataFrame(free_params_set)

    # remove all detunings which are too close to zero
    delta_threshold = 1e-3
    new_df = new_df[(new_df['δ1'].abs() > delta_threshold)
                    & (new_df['δ2'].abs() > delta_threshold)
                    & (new_df['δ3'].abs() > delta_threshold)]
    new_df = new_df[(new_df['Δa'].abs() > delta_threshold)
                    & (new_df['Δb'].abs() > delta_threshold)]

    # Calculate some meaningful quantities for a sample
    new_df['Ω1/δ1'] = new_df['Ω1'] / new_df['δ1']
    new_df['Ω2/δ2'] = new_df['Ω2'] / new_df['δ2']
    new_df['Ω3/δ3'] = new_df['Ω3'] / new_df['δ3']

    calc_adiabaticity_ab_cavity_coupling(new_df)
    if not new_df.empty:
        new_df = new_df[new_df['adiabaticity_ab_cavity_coupling'] < 1e-1]
    calc_adiabaticity_ab_eff_Rabi(new_df)
    if not new_df.empty:
        new_df = new_df[new_df['adiabaticity_ab_eff_Rabi'] < 1e-1]
    calc_adiabaticity_one_r_eff_Rabi(new_df)
    if not new_df.empty:
        new_df = new_df[new_df['adiabaticity_one_r_eff_Rabi'] < 1e-1]

    Ω_3t(new_df)
    new_df[['ΩR', 'ΔR', 'ΩR/ΔR']] = new_df.apply(lambda row: calc_ΩR_ΔR(row), axis=1)
    new_df = new_df[new_df['ΩR/ΔR'] > 1e-5]

    longer_df = pd.concat([dataframe, new_df], ignore_index=True)
    return longer_df


df = pd.DataFrame()
df = add_new_params(df, int(1e4))
# %% Calculate some statistics
while (len(df) < 1e3):
    df = add_new_params(df, int(1e4))
    print(f"Length of dataframe: {len(df)}")
df.sort_values('ΩR/ΔR', inplace=True, ignore_index=True, ascending=False)

# %% Plot pairplot of the parameters
sns.pairplot(df,
             hue='ΩR/ΔR',
             vars=['Ω2/δ2', 'adiabaticity_ab_cavity_coupling', 'ΩR/ΔR',
                   'adiabaticity_one_r_eff_Rabi', 'Ω1'],
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
result = mesolve(Hamiltonian_5x5(**df.iloc[0].to_dict()), initial_state, times,
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

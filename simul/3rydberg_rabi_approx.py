"""
In this notebook we simulate the dynamics of three Rydberg atoms to efficiently generate
triply ground-to-excited transitions governed by a collective Rabi frequency (transition amplitude) $\overline{\Omega}$
"""
# %%
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


@singledispatch
def Ω_1t(n_offset: int, **kwargs):
    return kwargs['Ω1'] * kwargs['g_a'] * np.sqrt(kwargs['n'] + n_offset + 1) / np.abs(kwargs['δ1'])


@Ω_1t.register(pd.DataFrame)
def _(dataframe: pd.DataFrame, force: bool = False):
    if force == False and 'Ω_1t' in dataframe.columns:
        return
    dataframe['Ω_1t'] = dataframe['Ω1'] * dataframe['g_a'] * np.sqrt(dataframe['n'] + 1) / np.abs(dataframe['δ1'])


@singledispatch
def Ω_3t(n_offset: int, **kwargs):
    return kwargs['Ω3'] * kwargs['g_b'] * np.sqrt(kwargs['n'] + n_offset) / np.abs(kwargs['δ3'])


@Ω_3t.register(pd.DataFrame)
def _(dataframe: pd.DataFrame, force: bool = False):
    if force == False and 'Ω_3t' in dataframe.columns:
        return
    dataframe['Ω_3t'] = dataframe['Ω3'] * dataframe['g_b'] * np.sqrt(dataframe['n']) / np.abs(dataframe['δ3'])


# %% define the Deltas

def Delta_grr(**kwargs):
    return Δ_3(0, **kwargs) + kwargs['δ2']


def Delta_rgr(**kwargs):
    return Δ_1(-1, **kwargs) + Δ_3(1, **kwargs)


def Delta_rrg(**kwargs):
    return Δ_1(0, **kwargs) + kwargs['δ2']


def Δ_rrr(**kwargs):
    return Δ_1(-1, **kwargs) + kwargs['δ2'] + Δ_3(1, **kwargs)


# %% Define the 5x5 Hamiltonian matrix
ggg_state = basis(8, 0)
rrr_state = basis(8, 1)
rgg_state = basis(8, 2)
grg_state = basis(8, 3)
ggr_state = basis(8, 4)
rrg_state = basis(8, 5)
rgr_state = basis(8, 6)
grr_state = basis(8, 7)


def Hamiltonian_8x8(series: pd.Series):
    kwargs = series.to_dict()

    # Initialize an 8x8 zero matrix
    H_8x8 = qzero(8)

    # Add the off-diagonal elements
    H_8x8 += Ω_1t(0, **kwargs) * rgg_state * ggg_state.dag()
    H_8x8 += kwargs['Ω2'] * grg_state * ggg_state.dag()
    H_8x8 += Ω_3t(0, **kwargs) * ggr_state * ggg_state.dag()

    H_8x8 += Ω_3t(1, **kwargs) * rrg_state * rrr_state.dag()
    H_8x8 += kwargs['Ω2'] * rgr_state * rrr_state.dag()
    H_8x8 += Ω_1t(-1, **kwargs) * grr_state * rrr_state.dag()

    H_8x8 += kwargs['Ω2'] * rrg_state * rgg_state.dag()
    H_8x8 += Ω_3t(1, **kwargs) * rgr_state * rgg_state.dag()

    H_8x8 += Ω_1t(0, **kwargs) * rrg_state * grg_state.dag()
    H_8x8 += Ω_3t(0, **kwargs) * grr_state * grg_state.dag()

    H_8x8 += Ω_1t(-1, **kwargs) * rgr_state * ggr_state.dag()
    H_8x8 += kwargs['Ω2'] * grr_state * ggr_state.dag()

    # Add the Hermitian conjugate (upper triangular part)
    H_8x8 += H_8x8.dag()

    # Add the diagonal elements
    # H_8x8 += Delta_ggg(**kwargs) * ggg_state.proj()
    H_8x8 += Δ_rrr(**kwargs) * rrr_state.proj()
    H_8x8 += Δ_1(0, **kwargs) * rgg_state.proj()
    H_8x8 += kwargs['δ2'] * grg_state.proj()
    H_8x8 += Δ_3(0, **kwargs) * ggr_state.proj()
    H_8x8 += Delta_rrg(**kwargs) * rrg_state.proj()
    H_8x8 += Delta_rgr(**kwargs) * rgr_state.proj()
    H_8x8 += Delta_grr(**kwargs) * grr_state.proj()

    # Optionally normalize the Hamiltonian
    # H_8x8 -= H_8x8[0, 0] * qeye(8)

    return H_8x8


# %% Functions to analyze a parameter set

def project_8x8_to_2x2(H_8x8, approx: bool = False):
    P = rrr_state.proj().to('CSR') + ggg_state.proj().to('CSR')
    Q = qeye(8).to('CSR') - P

    if approx:
        M_Delta = Qobj(np.diag(np.diag((Q * H_8x8 * Q).full())))
        M_Omega = Q * H_8x8 * Q - M_Delta

        M_Deltainv = (M_Delta + 10e-10 * qeye(8)).inv()
        H_traced = P * H_8x8 * P - P * H_8x8 * Q * (M_Deltainv - M_Deltainv * M_Omega * M_Deltainv) * Q * H_8x8 * P
    else:
        H_traced = P * H_8x8 * P - P * H_8x8 * Q * (Q * H_8x8 * Q + 10e-10 * qeye(8)).inv(sparse=True) * Q * H_8x8 * P

    # FIXME could be extracted by using projection operators of the states to be independent on basis definition
    return Qobj(np.array([[H_traced[0, 0], H_traced[0, 1]], [H_traced[1, 0], H_traced[1, 1]]]))


def calc_ΩR_ΔR(series: pd.Series, force=False):
    """
    Calculate the effective Rabi frequency and detuning for the collective states |ggg> and |rrr>
    :returns: |Ω_R|/|Δ_R|
    """
    if force == False and 'ΩR' in series.index:
        return
    H_2x2 = project_8x8_to_2x2(Hamiltonian_8x8(series), approx=False).data.to_array()
    ΩR = np.abs(H_2x2[0, 1].real)
    ΔR = (H_2x2[0, 0] - H_2x2[1, 1]).real
    return pd.Series([ΩR, ΔR, np.abs(ΩR / ΔR).real], index=['ΩR', 'ΔR', 'ΩR/ΔR'], dtype=np.float64)


def calc_adiabaticity_cavity_coupling(dataframe: pd.DataFrame, force: bool = False):
    """
    In order to be able to eliminate the intermediate states |a> and |b> we need to have the following conditions:
    Δa >> ga, Δb >> gb.
    If these conditions are met, the result should be close to 0.
    :returns max of ga/|Δa| and gb/|Δb|
    """
    if dataframe.empty:
        warnings.warn("calc_adiabaticity_cavity_coupling: Empty dataframe")
        return
    if force == False and 'adiabaticity_cavity_coupling' in dataframe.columns:
        return
    divis = pd.DataFrame()
    divis['da/ga'] = dataframe['g_a'] / dataframe['Δa'].abs()
    divis['db/gb'] = dataframe['g_b'] / dataframe['Δb'].abs()
    dataframe['adiabaticity_cavity_coupling'] = divis.max(axis=1)


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


def compare_time_evolution(ds: pd.Series, show_plot=False):
    initial_state = ggg_state

    ds['Ω_3t'] = Ω_3t(0, **ds)
    ds = ds._append(calc_ΩR_ΔR(ds))
    # Time evolution
    times = np.linspace(0, np.amin([4 * np.pi * ds['Ω1'], 10]), 1000)  # Time range for simulation

    # Solve the master equation
    result = sesolve(Hamiltonian_8x8(ds), initial_state, times, e_ops=[], options={'nsteps': 1e8})
    # Extract expectation values
    P_0r = expect(ggg_state.proj(), result.states)  # Probability of being in the ground state |ggg>
    P_1r = expect(rgg_state.proj() + grg_state.proj() + ggr_state.proj(), result.states)  # P of one Rydberg state
    P_2r = expect(rrg_state.proj() + rgr_state.proj() + grr_state.proj(), result.states)  # P of two Rydberg states
    P_3r = expect(rrr_state.proj(), result.states)  # P of three Rydberg states |rrr>

    resul2x2 = sesolve(project_8x8_to_2x2(Hamiltonian_8x8(ds), approx=False), basis(2, 0),
                       times, e_ops=[], options={'nsteps': 1e8})
    # Extract expectation values
    P_ggg2x2 = expect(basis(2, 0).proj(), resul2x2.states)  # Probability of being in the ground state |gg>
    P_rrr2x2 = expect(basis(2, 1).proj(), resul2x2.states)  # Probability of being in the Rydberg state |rr>

    # Plot the probabilities and coherences
    if show_plot:
        plt.figure()
        plt.plot(times, P_0r, label=r'$P (Ground State)')
        plt.plot(times, P_1r, label=r'$P (1 Rydberg State)')
        plt.plot(times, P_2r, label=r'$P (2 Rydberg State)')
        plt.plot(times, P_3r, label=r'$P (3 Rydberg State)')

        plt.plot(times, P_ggg2x2, label=r'$\langle \psi | ggg | \psi \rangle$ (Ground State)2x2')
        plt.plot(times, P_rrr2x2, label=r'$\langle \psi | rrr | \psi \rangle$ (Rydberg State)2x2')

        plt.xlabel('Time Ω_1t')
        plt.ylabel('Probability')
        plt.legend()
        plt.tight_layout(pad=0.2)
        plt.show()
    return np.sum((P_0r - P_ggg2x2) ** 2 + (P_3r - P_rrr2x2) ** 2)


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
    'g_a': [0 * punit, 1000 * punit],
    'g_b': [0 * punit, 1000 * punit],
    'n': [1, 1],
}


# free_params_bounds = {
#    'Ω1': [1 * punit, 1 * punit],
#    'Ω2': [0.01 * punit, 10 * punit],
#    'Ω3': [0.01 * punit, 10 * punit],
#    'δ1': [-1000 * punit, -1000 * punit],
#    'δ2': [-1000 * punit, 1000 * punit],
#    'δ3': [1000 * punit, 1000 * punit],
#    'Δa': [-500 * punit, -500 * punit],
#    'Δb': [500 * punit, 500 * punit],
#    'g_a': [10 * punit, 10 * punit],
#    'g_b': [15 * punit, 15 * punit],
#    'n': [1, 1],
# }


def add_new_params(dataframe: pd.DataFrame, length: int):
    # Generate the parameters set
    free_params_set = {key: np.random.uniform(*value, length) for key, value in free_params_bounds.items()}
    new_df = pd.DataFrame(free_params_set)

    # Remove all detunings which are too close to zero
    delta_threshold = 1e-4
    new_df = new_df[(new_df['δ1'].abs() > delta_threshold)
                    & (new_df['δ2'].abs() > delta_threshold)
                    & (new_df['δ3'].abs() > delta_threshold)]
    new_df = new_df[(new_df['Δa'].abs() > delta_threshold)
                    & (new_df['Δb'].abs() > delta_threshold)]
    if new_df.empty:
        return dataframe

    # Calculate some meaningful quantities for a sample
    new_df['Ω1/δ1'] = new_df['Ω1'] / np.abs(new_df['δ1'])
    new_df = new_df[new_df['Ω1/δ1'] < 1]
    if new_df.empty:
        warnings.warn("All data removed due to Ω1/δ1")
        return dataframe
    new_df['Ω2/δ2'] = new_df['Ω2'] / np.abs(new_df['δ2'])
    new_df = new_df[new_df['Ω2/δ2'] < 1]
    if new_df.empty:
        warnings.warn("All data removed due to Ω2/δ2")
        return dataframe
    new_df['Ω3/δ3'] = new_df['Ω3'] / np.abs(new_df['δ3'])
    new_df = new_df[new_df['Ω3/δ3'] < 1]
    if new_df.empty:
        warnings.warn("All data removed due to Ω3/δ3")
        return dataframe

    calc_adiabaticity_cavity_coupling(new_df)
    new_df = new_df[new_df['adiabaticity_cavity_coupling'] < 1]
    if new_df.empty:
        warnings.warn("All data removed due to adiabaticity_cavity_coupling")
        return dataframe
    calc_adiabaticity_one_r_eff_Rabi(new_df)
    new_df = new_df[new_df['adiabaticity_one_r_eff_Rabi'] < 1]
    if new_df.empty:
        warnings.warn("All data removed due to adiabaticity_one_r_eff_Rabi")
        return dataframe

    Ω_3t(new_df)
    new_df[['ΩR', 'ΔR', 'ΩR/ΔR']] = new_df.apply(lambda row: calc_ΩR_ΔR(row), axis=1)
    new_df = new_df[abs(new_df['ΩR/ΔR']) > 1e-2]
    if new_df.empty:
        warnings.warn("All data removed due to ΩR/ΔR")
        return dataframe
    Δ_1(new_df)
    Δ_3(new_df)
    new_df['resonance'] = new_df['ΔR'].abs() / new_df[['Δ_1', 'δ2', 'Δ_3']].abs().min(axis=1)
    # new_df = new_df[new_df['resonance'] < 2e-1]
    if new_df.empty:
        warnings.warn("All data removed due to resonance")
        return dataframe
    new_df['log ΩR/ΔR'] = np.log10(new_df['ΩR/ΔR'])
    new_df['fidelity'] = new_df.apply(lambda row: compare_time_evolution(ds=row, show_plot=False), axis=1)
    new_df = new_df[new_df['fidelity'] < 1e-1]
    if new_df.empty:
        warnings.warn("All data removed due to fidelity")
        return dataframe
    # new_df['fidelity'] = new_df.apply(lambda row: compare_time_evolution(ds=row, show_plot=True), axis=1)
    new_df['well'] = np.sqrt(
        new_df[['resonance', 'adiabaticity_one_r_eff_Rabi', 'adiabaticity_cavity_coupling']].prod(axis=1))

    longer_df = pd.concat([dataframe, new_df], ignore_index=True)
    return longer_df


df = pd.DataFrame()
df = add_new_params(df, int(20))

# %% Calculate some statistics
while (len(df) < 20):
    df = add_new_params(df, int(5e3))
    print(f"Length of dataframe: {len(df)}")

# %%
df['Delta_rrg'] = df.apply(lambda row: Delta_rrg(**row.to_dict()), axis=1)
df['Delta_rgr'] = df.apply(lambda row: Delta_rgr(**row.to_dict()), axis=1)
df['abs Delta_rgr'] = np.abs(df.apply(lambda row: Delta_rgr(**row.to_dict()), axis=1))
df['Delta_grr'] = df.apply(lambda row: Delta_grr(**row.to_dict()), axis=1)
df['log fidelity'] = np.log10(df['fidelity'])
df['abs Delta_rgr/δ2'] = df['abs Delta_rgr'] / np.abs(df['δ2'])
df['δ1*δ3'] = df['δ1'] * df['δ3']
df.sort_values('fidelity', inplace=True, ignore_index=True, ascending=False)
# %% Plot pairplot of the parameters
sns.pairplot(df[df['ΩR/ΔR'] > 1e-3],
             hue='log fidelity',
             vars=['log fidelity', 'log ΩR/ΔR', 'resonance', 'adiabaticity_cavity_coupling'],
             diag_kind=None,
             kind='scatter',
             # corner=True,
             # plot_kws={'alpha': 0.8},
             palette='flare_r',
             )
plt.show()

# %% Test the projection using time evolution comparison

compare_time_evolution(ds=df.iloc[0], show_plot=True)

# %%

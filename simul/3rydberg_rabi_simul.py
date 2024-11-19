"""
In this notebook we simulate the dynamics of three Rydberg atoms to efficiently generate
triply ground-to-excited transitions governed by a collective Rabi frequency (transition amplitude) $\overline{\Omega}$
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from qutip import *


# plt.style.use("simul/paperstyle.mplstyle")

# %% As in the 2 atom case, we can reduce the problem to work only with the collective states spanned by the individual
# ground/excited states in the basis $$\{|rrr,n+1\rangle, |grr,n\rangle, |rgr,n+2\rangle, |rrg,n\rangle, |rgg,n+1\rangle, |grg,n-1\rangle, |ggr,n+1\rangle, |ggg,n\rangle\}^T$$

# Hamiltonian and detunings for individual atoms

def Δ_1(n_offset=0, **kwargs):
    return (kwargs['δ1'] - kwargs['Δa']
            + kwargs['Ω1'] ** 2 / kwargs['δ1']
            - kwargs['g_a'] ** 2 * (kwargs['n'] + n_offset + 1) / kwargs['δ1']
            + kwargs['g_b'] ** 2 * (kwargs['n'] + n_offset + 2) / kwargs['Δb']
            )


def Δ_2(**kwargs):
    return kwargs['δ2']


def Δ_3(n_offset=0, **kwargs):
    return (kwargs['δ3']
            + kwargs['Δb'] +
            kwargs['Ω3'] ** 2 / kwargs['δ3']
            - kwargs['g_b'] ** 2 * (kwargs['n'] + n_offset) / kwargs['δ3']
            + kwargs['g_a'] ** 2 * (kwargs['n'] + n_offset - 1) / kwargs['Δa'])


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


def Ω_1t(n_offset=0, **kwargs):
    return kwargs['Ω1'] * kwargs['g_a'] * np.sqrt(kwargs['n'] + n_offset + 1) / kwargs['δ1']


def Ω_2t(n_offset=0, **kwargs):
    return kwargs['Ω2']


def Ω_3t(n_offset=0, **kwargs):
    return kwargs['Ω3'] * kwargs['g_b'] * np.sqrt(kwargs['n'] + n_offset) / kwargs['δ3']


# %% define the Deltas

def Delta_ggg(**kwargs):
    return (
            -kwargs['Ω2'] ** 2 / kwargs['δ2']
            - Ω_1t(**kwargs) ** 2 / Δ_1(**kwargs)
            - Ω_3t(**kwargs) ** 2 / Δ_3(**kwargs)
    )


# def Delta_ggr(**kwargs):
#    return Δ_3(**kwargs)
#
#
# def Delta_grg(**kwargs):
#    return Δ_2(**kwargs)
#
#
# def Delta_rgg(**kwargs):
#    return Δ_1(**kwargs)


def Delta_grr(**kwargs):
    return Δ_3(**kwargs) + kwargs['δ2']


def Delta_rgr(**kwargs):
    return Δ_1(-1, **kwargs) + Δ_3(1, **kwargs)


def Delta_rrg(**kwargs):
    return Δ_1(**kwargs) + kwargs['δ2']


def Delta_rrr(**kwargs):
    return Delta_ggg(**kwargs) + Delta_rrg_prime(**kwargs) + Delta_rgr_prime(**kwargs) + Delta_grr_prime(**kwargs)


def Ω_13t(n_offset=0, **kwargs):
    return Ω_1t(n_offset, **kwargs) * Ω_3t(n_offset, **kwargs) * (Δ_1(n_offset, **kwargs) + Δ_3(n_offset, **kwargs)) / (
            Δ_1(n_offset, **kwargs) * Δ_3(n_offset, **kwargs))


def Ω_12t(n_offset=0, **kwargs):
    return Ω_1t(n_offset, **kwargs) * kwargs['Ω2'] * (Δ_1(n_offset, **kwargs) + kwargs['δ2']) / (
            Δ_1(n_offset, **kwargs) * kwargs['δ2'])


def Ω_23t(n_offset=0, **kwargs):
    return Ω_3t(n_offset, **kwargs) * kwargs['Ω2'] * (Δ_3(n_offset, **kwargs) + kwargs['δ2']) / (
            Δ_3(n_offset, **kwargs) * kwargs['δ2'])


def Delta_rrg_prime(**kwargs):
    return (Delta_rrg(**kwargs)
            - kwargs['Ω2'] ** 2 / Δ_1(**kwargs)
            - Ω_1t(**kwargs) ** 2 / kwargs['δ2']
            )


def Delta_rgr_prime(**kwargs):
    return (Delta_rgr(**kwargs)
            - Ω_1t(-1, **kwargs) * Ω_1t(**kwargs) / Δ_3(**kwargs)
            - Ω_3t(**kwargs) * Ω_3t(1, **kwargs) / Δ_1(**kwargs)
            )


def Delta_grr_prime(**kwargs):
    return (Delta_grr(**kwargs)
            - kwargs['Ω2'] ** 2 / Δ_3(**kwargs)
            - Ω_3t(**kwargs) * Ω_3t(1, **kwargs) / kwargs['δ2']
            )


# %% Define the 5x5 Hamiltonian matrix
ggg_state = basis(5, 0)
rrg_state = basis(5, 1)
rgr_state = basis(5, 2)
grr_state = basis(5, 3)
rrr_state = basis(5, 4)


def Hamiltonian_5x5(**kwargs):
    H_5x5 = qzero(5)
    # add lower triangular part from bottom
    # 1st row
    H_5x5 += Ω_3t(1, **kwargs) * rrr_state * rrg_state.dag()
    H_5x5 += Ω_2t(**kwargs) * rrr_state * rgr_state.dag()
    H_5x5 += Ω_1t(-1, **kwargs) * rrr_state * grr_state.dag()
    # 2nd row
    H_5x5 += -Ω_2t(**kwargs) * Ω_1t(-1, **kwargs) / Δ_3(**kwargs) * grr_state * rgr_state.dag()
    H_5x5 += -Ω_1t(**kwargs) * Ω_3t(1, **kwargs) / kwargs['δ2'] * grr_state * rrg_state.dag()
    H_5x5 += -Ω_23t(**kwargs) * grr_state * ggg_state.dag()
    # 3rd row
    H_5x5 += -Ω_2t(**kwargs) * Ω_3t(**kwargs) / Δ_1(**kwargs) * rgr_state * rrg_state.dag()
    H_5x5 += -Ω_13t(**kwargs) * rgr_state * ggg_state.dag()
    # 4th row
    H_5x5 += -Ω_12t(**kwargs) * rrg_state * ggg_state.dag()

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


np.abs(np.round(Hamiltonian_5x5(**df.iloc[5467723].to_dict()).data.to_array(), 2))


# %% Functions to analyze a parameter set

def project_5x5_to_2x2(H_5x5):
    P = rrr_state.proj().to('CSR') + ggg_state.proj().to('CSR')
    Q = qeye(5).to('CSR') - P
    H_traced = P * H_5x5 * P - P * H_5x5 * Q * (Q * H_5x5 * Q + 10e-10 * qeye(5)).inv(sparse=True) * Q * H_5x5 * P
    return Qobj(np.array([[H_traced[0, 0], H_traced[0, -1]], [H_traced[-1, 0], H_traced[-1, -1]]]))


def calc_ΩR_ΔR(**kwargs):
    """
    Calculate the effective Rabi frequency and detuning for the collective states |ggg> and |rrr>
    :returns: |Ω_R|/|Δ_R|
    """
    H_2x2 = project_5x5_to_2x2(Hamiltonian_5x5(**kwargs)).data.to_array()
    return float(np.abs(H_2x2[0, 1]) / np.abs(H_2x2[0, 0] - H_2x2[1, 1]))


def calc_adiabaticity_ab_eff_Rabi(**kwargs):
    """
    In order to be able to eliminate the intermediate states |a> and |b> we need to have the following conditions:
    δ1 >> Ω1, δ3 >> Ω3.
    If these conditions are met, the result should be close to 0.
    :returns max of Ω1/|δ1| and Ω3/|δ3|
    """
    return np.amax([kwargs['Ω1'] / np.abs(kwargs['δ1']),
                    kwargs['Ω3'] / np.abs(kwargs['δ3'])])


def calc_adiabaticity_ab_cavity_coupling(**kwargs):
    """
    In order to be able to eliminate the intermediate states |a> and |b> we need to have the following conditions:
    Δa >> ga, Δb >> gb.
    If these conditions are met, the result should be close to 0.
    :returns max of ga/|Δa| and gb/|Δb|
    """
    return np.amax([kwargs['g_a'] / np.abs(kwargs['Δa']),
                    kwargs['g_b'] / np.abs(kwargs['Δb'])])


def calc_adiabaticity_one_r_eff_Rabi(**kwargs):
    """
    In order to be able to eliminate the states with one excitation we need to have the following conditions:
    Δ_1 >> Ω_1t, δ2 >> Ω2, Δ_3 >> Ω_3t.
    :return: max of |Ω_1t|/|Δ_1|, Ω2/|δ2|, |Ω_3t|/|Δ_3|
    """
    return np.amax([np.abs(Ω_1t(**kwargs)) / np.abs(Δ_1(**kwargs)),
                    kwargs['Ω2'] / np.abs(kwargs['δ2']),
                    np.abs(Ω_3t(**kwargs)) / np.abs(Δ_3(**kwargs))])


# %% Parameter normalisation to define timescales
punit = 1

"""
Parameters which work for 2 atoms (frequencies, detunings, and couplings)

Ω1, Ω2, Ω3 = 56.50 * punit, 60.0 * punit, 112.0 * punit  # Rabi frequencies for atoms
δ1, δ2, δ3 = 663.8 * punit, -742.0 * punit, 716.0 * punit  # Detunings for each atom
Δa, Δb = 722.0 * punit, 800.0 * punit  # Frequency shifts
g_a, g_b = 9.5 * punit, 10.0 * punit  # Coupling strengths
n = 10  # Number of photons in the cavity
"""

free_params_set = {
    'Ω1': np.random.choice(np.linspace(0, 100, 100) * punit, 5),
    'Ω2': np.random.choice(np.linspace(0, 100, 100) * punit, 5),
    'Ω3': np.random.choice(np.linspace(0, 200, 100) * punit, 5),
    'δ1': np.random.choice(np.linspace(-1000, 1000, 100) * punit + 0.1, 5),
    'δ2': np.random.choice(np.linspace(-1000, 1000, 100) * punit + 0.1, 5),
    'δ3': np.random.choice(np.linspace(-1000, 1000, 100) * punit + 0.1, 5),
    'Δa': np.random.choice(np.linspace(-1000, 1000, 100) * punit + 0.1, 5),
    'Δb': np.random.choice(np.linspace(-1000, 1000, 100) * punit + 0.1, 5),
    'g_a': np.random.choice(np.linspace(0, 100, 100) * punit, 5),
    'g_b': np.random.choice(np.linspace(0, 100, 100) * punit, 5),
    'n': [10],
}

index = pd.MultiIndex.from_product(list(free_params_set.values()), names=free_params_set.keys())
df = pd.DataFrame(index=index).reset_index()

# %% Calculate some meaningful quantities for a sample
df_sample = df.sample(frac=0.30, random_state=83)
# df['ΩR_ΔR'] = df.apply(lambda row: calc_ΩR_ΔR(**row.to_dict()), axis=1)

df_sample['adiabaticity_ab_cavity_coupling'] = df_sample.apply(
    lambda row: calc_adiabaticity_ab_cavity_coupling(**row.to_dict()), axis=1)
df_sample = df_sample[df_sample['adiabaticity_ab_cavity_coupling'] < 1e-2]
df_sample['adiabaticity_ab_eff_Rabi'] = df_sample.apply(lambda row: calc_adiabaticity_ab_eff_Rabi(**row.to_dict()),
                                                        axis=1)
df_sample = df_sample[df_sample['adiabaticity_ab_eff_Rabi'] < 1e-2]
df_sample['adiabaticity_one_r_eff_Rabi'] = df_sample.apply(
    lambda row: calc_adiabaticity_one_r_eff_Rabi(**row.to_dict()), axis=1)
df_sample['ΩR_ΔR'] = df_sample.apply(lambda row: calc_ΩR_ΔR(**row.to_dict()), axis=1)

df_sample

# %% Plot pairplot of the parameters
sns.pairplot(df_sample,
             hue='ΩR_ΔR',
             vars=['Ω1', 'Ω2', 'Ω3', 'δ1', 'δ2', 'δ3'],
             diag_kind='kde')
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

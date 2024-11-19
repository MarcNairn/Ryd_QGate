"""
In this notebook we simulate the dynamics of three Rydberg atoms to efficiently generate
triply ground-to-excited transitions governed by a collective Rabi frequency (transition amplitude) $\overline{\Omega}$
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("simul/paperstyle.mplstyle")

# %% Parameter normalisation to define timescales
punit = 1

""" # Parameters (frequencies, detunings, and couplings)
Ω1, Ω2, Ω3 = 20*punit,10*punit,7*punit  # Rabi frequencies for atoms
δ1, δ2, δ3 = 1*punit, -1*punit, 1*punit  # Detunings for each atom
Δa, Δb = 20*punit, 20*punit  # Frequency shifts
g_a, g_b = 1*punit,1*punit  # Coupling strengths """

Ω1, Ω2, Ω3 = 56.50 * punit, 60.0 * punit, 112.0 * punit  # Rabi frequencies for atoms
δ1, δ2, δ3 = 663.8 * punit, -742.0 * punit, 716.0 * punit  # Detunings for each atom
Δa, Δb = 722.0 * punit, 800.0 * punit  # Frequency shifts
g_a, g_b = 9.5 * punit, 10.0 * punit  # Coupling strengths

# Photon number (n) for different transitions
n = 10  # Example photon number for simulation


# %% As in the 2 atom case, we can reduce the problem to work only with the collective states spanned by the individual
# ground/excited states in the basis $$\{|rrr,n+1\rangle, |grr,n\rangle, |rgr,n+2\rangle, |rrg,n\rangle, |rgg,n+1\rangle, |grg,n-1\rangle, |ggr,n+1\rangle, |ggg,n\rangle\}^T$$

# Hamiltonian and detunings for individual atoms

def Delta1(n):
    return δ1 - Δa + Ω1 ** 2 / δ1 - g_a * g_a * (n + 1) / δ1 + g_b * g_b * (n + 2) / Δb


def Delta2(n):
    return δ2 + Δb + Ω2 ** 2 / δ2 - g_b * g_b / δ2 * n - g_a * g_a * (n - 1) / Δa


def Delta3(n):
    return δ3 - Δa + Ω3 ** 2 / δ3 - g_a * g_a * (n + 1) / δ3 + g_b * g_b * (n + 2) / Δb


def H_atom1(n):
    sqrt_n1 = np.sqrt(n + 1)
    # 2x2 matrix for the atom's Hamiltonian
    H = np.array([[Delta1(n), Ω1 * g_a / δ1 * sqrt_n1],
                  [Ω1 * g_a / δ1 * sqrt_n1, 0]])

    return Qobj(H)


def H_atom2(n):
    sqrt_n = np.sqrt(n)
    # 2x2 matrix for the atom's Hamiltonian
    H = np.array([[Delta2(n), Ω2 * g_b / δ2 * sqrt_n],
                  [Ω2 * g_b / δ2 * sqrt_n, 0]])

    return Qobj(H)


def H_atom3(n):
    sqrt_n1 = np.sqrt(n + 1)
    # 2x2 matrix for the atom's Hamiltonian
    H = np.array([[Delta3(n), Ω1 * g_a / δ1 * sqrt_n1],
                  [Ω1 * g_a / δ1 * sqrt_n1, 0]])

    return Qobj(H)


# %% Moving to the combined basis the Hamiltonian takes the form:
#
# $$ H_T =
# \begin{pmatrix}
# \Delta_T(n) & {\Omega}_1(n) & {\Omega}_2(n+2) & {\Omega}_3(n) & 0 & 0 & 0 & 0 \\
# {\Omega}_1(n) & 2(\Delta_2 + \Delta_3) & 0 & 0 & 0 & {\Omega}_3(n-1) & {\Omega}_2(n+1) & 0 \\
# {\Omega}_2(n+2) & 0 & 2(\Delta_1 + \Delta_3) & 0 & {\Omega}_3(n+1) & 0 & {\Omega}_1(n+1) & 0 \\
# {\Omega}_3(n) & 0 & 0 & 2(\Delta_1 + \Delta_2) & {\Omega}_2(n+1) & {\Omega}_1(n-1) & 0 & 0 \\
# 0 & 0 & {\Omega}_3(n+1) & {\Omega}_2(n+1) & \Delta_1 & 0 & 0 & {\Omega}_1(n) \\
# 0 & {\Omega}_3(n-1) & 0 & {\Omega}_1(n-1) & 0 & \Delta_2 & 0 & {\Omega}_2(n) \\
# 0 & {\Omega}_2(n+1) & {\Omega}_1(n+1) & 0 & 0 & 0 & \Delta_3 & {\Omega}_3(n) \\
# 0 & 0 & 0 & 0 & {\Omega}_1(n) & {\Omega}_2(n) & {\Omega}_3(n) & 0
# \end{pmatrix}
# $$
#
# where $\Delta_T(n) = \Delta_1(n-1) + \Delta_2(n+1) + \Delta_3(n-1)$.
# %%
def Omega1t(n):
    return Ω1 * g_a * np.sqrt(n + 1) / δ1


def Omega2t(n):
    return Ω2 * g_b * np.sqrt(n) / δ2


def Omega3t(n):
    return Ω3 * g_a * np.sqrt(n + 1) / δ3


# dets = np.linspace(10, 0, 8)

Deltagg = 0  # dets[0]
Deltagrr = Delta2(n + 1) + Delta1(n - 1)
Deltargr = (Delta1(n - 1) + Delta3(n - 1))  #
Deltarrg = (Delta1(n - 1) + Delta2(n + 1))  #
Deltargg = Delta1(n)
Deltagrg = Delta2(n)  #
Deltaggr = Delta3(n)  #
Deltarrr = Delta1(n) + Delta2(n + 2) + Delta3(n)

H_a = np.array([[Deltarrr, Omega1t(n), Omega2t(n + 2), Omega3t(n), 0, 0, 0, 0],
                [Omega1t(n), Deltagrr, 0, 0, 0, Omega3t(n - 1), Omega2t(n + 1), 0],
                [Omega2t(n), 0, Deltargr, 0, Omega3t(n + 1), 0, Omega1t(n + 1), 0],
                [Omega3t(n), 0, 0, Deltarrg, Omega2t(n + 1), Omega1t(n - 1), 0, 0],
                [0, 0, Omega3t(n + 1), Omega2t(n + 1), Deltargg, 0, 0, Omega1t(n)],
                [0, Omega3t(n - 1), 0, Omega1t(n - 1), 0, Deltagrg, 0, Omega2t(n)],
                [0, Omega2t(n + 1), Omega1t(n + 1), 0, 0, 0, Deltaggr, Omega3t(n)],
                [0, 0, 0, 0, Omega1t(n), Omega2t(n), Omega3t(n), 0]]
               )
Ht = Qobj(H_a)
Ht
# %%
rrr_state = basis(8, 0).to('CSR')
grr_state = basis(8, 1).to('CSR')
rgr_state = basis(8, 2).to('CSR')
rrg_state = basis(8, 3).to('CSR')
rgg_state = basis(8, 4).to('CSR')
grg_state = basis(8, 5).to('CSR')
ggr_state = basis(8, 6).to('CSR')
ggg_state = basis(8, 7).to('CSR')

P = rrr_state.proj() + ggg_state.proj()
Q = qeye(8) - P
# %%
# TODO: Why is there a 10e-10 in the expression?
H_eff = P * Ht * P - P * Ht * Q * (Q * Ht * Q + 10e-10 * qeye(8)).inv(sparse=False) * Q * Ht * P

hinton(H_eff)
plt.show()
# %%
initial_state = ggg_state

# Time evolution
times = np.linspace(0, 240 * punit, 1000)  # Time range for simulation

# Solve the master equation
result = mesolve(Ht, initial_state, times, c_ops=[], e_ops=[], options={'nsteps': 1e8})
# %%
# Extract expectation values
P_ggg = expect(ggg_state * ggg_state.dag(), result.states)  # Probability of being in the ground state |gg>
P_rrr = expect(rrr_state * rrr_state.dag(), result.states)  # Probability of being in the Rydberg state |rr>

P_rr = expect(grr_state * grr_state.dag() + rgr_state * rgr_state.dag() * rrg_state * rrg_state.dag(), result.states)
P_r = expect(ggr_state * ggr_state.dag() + grg_state * grg_state.dag() * rgg_state * rgg_state.dag(), result.states)
# %%
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

# %%
rho_f = result.states[-1] * result.states[-1].dag()
hinton(rho_f)
plt.show()

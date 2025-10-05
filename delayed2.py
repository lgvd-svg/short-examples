import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------
A = 1.0

# Time-dependent coefficients  (vectorised, so they also work for arrays)
B = lambda t: 10.0 #+ 0.5 * np.sin(t)
C = lambda t: 1.0 #+ 0.1 * np.cos(2 * t)
D = lambda t: 0.05 #* (t + 1)
E = lambda t: 0.5 #* np.exp(-0.1 * t)
F = lambda t: 0.5 #+ 0.2 * np.sin(0.5 * t)      # delay tau(t)

# History for t <= t0
t0 = 0.0
history = lambda t: 1.0

# Integration window
t_end = 20.0
# ---------------------------------------------------------

# Global list that stores the whole solution so far
# (filled with the history first)
t_hist = [t0 - 1.0, t0]          # two points to make interpolation happy
v_hist = [history(t0 - 1.0), history(t0)]


def rhs(t, y):
    """
    Right-hand side of the DDE.
    y is a 1-element array because solve_ivp insists on it.
    """
    v = y[0]

    # Time-dependent delay
    tau = F(t)
    tlag = t - tau

    # Evaluate v(t-tau) by linear interpolation of stored data
    # (clip to the leftmost history value to avoid extrapolation)
    v_lag = np.interp(tlag, t_hist, v_hist, left=history(tlag))

    dvdt = (B(t) - C(t) * v - D(t) * v**2 + E(t) * v * v_lag) / A
    return [dvdt]


def solve_dde_onestep(t_span):
    """
    Integrate one small step with solve_ivp, then append the result
    to the global history list.
    """
    sol = solve_ivp(rhs, t_span, [v_hist[-1]],
                    rtol=1e-6, atol=1e-9, max_step=0.05)

    # Append the dense output points to the global list
    t_hist.extend(sol.t[1:].tolist())
    v_hist.extend(sol.y[0, 1:].tolist())


# ---------------------------------------------------------
# 2. Main time-marching loop
# ---------------------------------------------------------
dt_chunk = 0.1          # integrate 0.1 time units at a time

t_curr = t0
while t_curr < t_end:
    t_next = min(t_curr + dt_chunk, t_end)
    solve_dde_onestep([t_curr, t_next])
    t_curr = t_next

# Convert lists to arrays for plotting
t_hist = np.asarray(t_hist)
v_hist = np.asarray(v_hist)

# ---------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------
plt.plot(t_hist, v_hist, lw=1.2)
plt.xlabel('t')
plt.ylabel('v(t)')
plt.title('DDE solution by plain SciPy / step-by-step integration')
plt.show()

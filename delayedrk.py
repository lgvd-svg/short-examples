import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1.  Problem coefficients
# ----------------------------------------------------------
A = 1.0
B = lambda t: 1.0 #+ 0.5*np.sin(t)
C = lambda t: 0.5  #+ 0.1*np.cos(2*t)
D = lambda t: 0.0 # + 0.05*(t+1)
E = lambda t: 0.05 # + np.exp(-0.1*t)
F = lambda t: 1.0 # + 0.2*np.sin(0.5*t)        # delay τ(t)

# ----------------------------------------------------------
# 2.  Discretisation parameters
# ----------------------------------------------------------
t0, t_end = 0.0, 20.0
h  = 0.01                                   # fixed RK-4 step
N  = int(np.ceil((t_end - t0)/h))           # total # steps
t  = np.linspace(t0, t_end, N+1)            # mesh t_0 … t_N
v  = np.empty(N+1)                          # solution array

# ----------------------------------------------------------
# 3.  History (v(t) for t <= t0)
# ----------------------------------------------------------
history = lambda t: 0.0
v[0] = history(t[0])

# ----------------------------------------------------------
# 4.  Helper: linear-interpolated delayed value
# ----------------------------------------------------------
def v_lag(ti):
    """Return v(ti - τ(ti)) using linear interpolation."""
    tau = F(ti)
    tlag = ti - tau
    # clip to the leftmost stored time if tlag < t0
    return np.interp(tlag, t, v, left=history(tlag))

# ----------------------------------------------------------
# 5.  Right-hand side
# ----------------------------------------------------------
def f(ti, vi, vi_lag):
    """dv/dt at time ti."""
    return (B(ti) - C(ti)*vi - D(ti)*vi**2 + E(ti)*vi*vi_lag) / A

# ----------------------------------------------------------
# 6.  Classic RK-4 step
# ----------------------------------------------------------
def rk4_step(i):
    """Advance from index i to i+1."""
    ti   = t[i]
    vi   = v[i]

    k1 = f(ti,            vi,                       v_lag(ti))
    k2 = f(ti + 0.5*h,    vi + 0.5*h*k1,            v_lag(ti + 0.5*h))
    k3 = f(ti + 0.5*h,    vi + 0.5*h*k2,            v_lag(ti + 0.5*h))
    k4 = f(ti + h,        vi + h*k3,                v_lag(ti + h))

    v[i+1] = vi + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ----------------------------------------------------------
# 7.  Time-marching loop
# ----------------------------------------------------------
for i in range(N):
    rk4_step(i)

# ----------------------------------------------------------
# 8.  Plot
# ----------------------------------------------------------
plt.plot(t, v, lw=1.2)
plt.xlabel('t')
plt.ylabel('v(t)')
plt.title('RK-4 solution of the delay differential equation')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parámetros ---
A = 1.7      # Constante A
B = 1.0      # Constante B (debe ser distinta de 0)

# --- Condiciones iniciales ---
v0   = 1.0   # v(0)
vdot0 = 0.0  # v'(0)

# --- Dominio de integración ---
x_span = (0, 1)
x_eval = np.linspace(*x_span, 1000)

# --- Sistema de EDOs de primer orden ---
def sistema(x, y):
    v, vdot = y
    dv_dx   = vdot
    d2v_dx2 = (- A + x * vdot) / B
    return [dv_dx, d2v_dx2]

# --- Resolver ---
sol = solve_ivp(sistema, x_span, [v0, vdot0],
                t_eval=x_eval, method='RK45')

# --- Extraer resultados ---
x   = sol.t
v   = sol.y[0]
vdot = sol.y[1]
T   = -B * vdot        # T(x) = -B * v'(x)
Tdot = np.gradient(T, x)  # T'(x) aproximado

# --- Verificación rápida ---
# v'(x) ≈ A + T'(x)
vprime_check = A + Tdot

# --- Gráficos ---
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(x, v, label='v(x)')
plt.ylabel('v(x)')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(x, vdot, label="v'(x)", color='orange')
plt.ylabel("v'(x)")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(x, T, label='T(x)', color='green')
plt.xlabel('x')
plt.ylabel('T(x)')
plt.grid(True)

plt.tight_layout()
plt.show()
'''
# --- Verificación de consistencia ---
plt.figure()
plt.plot(x, vdot, label="v'(x) numérico")
plt.plot(x, vprime_check, '--', label="A + T'(x)")
plt.legend()
plt.title('Verificación: v\'(x) = A + T\'(x)')
plt.grid(True)
plt.show()
'''

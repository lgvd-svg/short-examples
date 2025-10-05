"""
Ejemplo mínimo de resolución numérica de
v'(t) = A(t) - B(t)v(t) - C(t)v(t)v(t-D(t))
con JiTCDDE.
"""

import numpy as np
from jitcdde import jitcdde, y, t

# -------------------------------------------------
# 1. Definir los coeficientes y el retardo
# -------------------------------------------------
A = lambda t: 1.0                         # fuerza impulsora
B = lambda t: 0.5 + 0.1*np.sin(t)         # pérdidas lineales
C = lambda t: 0.2                         # término cuadrático retardado
D = lambda t: 0.5 + 0.1*np.cos(0.5*t)     # retardo variable (≈0.4–0.6)

# -------------------------------------------------
# 2. Definir la ecuación f(t) = v'(t)
# -------------------------------------------------
# y(0) = v(t)           -> valor actual
# y(0, t-D(t)) = v(t-D(t)) -> valor retardado
f = [ A(t) - B(t)*y(0) - C(t)*y(0)*y(0, t-D(t)) ]

# -------------------------------------------------
# 3. Crear el objeto DDE
# -------------------------------------------------
dde = jitcdde(f)

# -------------------------------------------------
# 4. Condición inicial (constante en [-Dmax,0])
# -------------------------------------------------
Dmax = 10.0  # mayor valor que pueda tomar D(t)
dde.constant_past([0.0], time=0.0)   # v(t)=1 para t<=0

# -------------------------------------------------
# 5. Integrar sobre el intervalo [0, T]
# -------------------------------------------------
T, dt = 20.0, 0.1
times = np.arange(0, T+dt, dt)
sol   = []

dde.step_on_discontinuities()          # maneja saltos por retardo variable
for ti in times:
    sol.append(dde.integrate(ti))

# -------------------------------------------------
# 6. Visualizar
# -------------------------------------------------
import matplotlib.pyplot as plt
plt.plot(times, sol)
plt.xlabel("t")
plt.ylabel("v(t)")
plt.title("Solución de la DDE con retardo variable")
plt.grid(True)
plt.show()

#!/usr/bin/env python3
"""
Solve v'(t) = A(t) + B(t)*v + C(t)*v**2  (pure Python, Tkinter GUI)
"""

import math, tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# ---------- adaptive RK45 ----------
def rk45(f, y0, t_span, tol=1e-6, h0=1e-3, h_max=0.1, h_min=1e-8):
    t, y = t_span[0], y0
    ts, ys = [t], [y]
    a = [0, 0.2, 0.3, 0.6, 1.0, 0.875]
    b = [[], [0.2], [3/40, 9/40], [0.3, -0.9, 1.2],
         [-11/54, 2.5, -70/27, 35/27],
         [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]]
    c  = [37/378, 0, 250/621, 125/594, 0, 512/1771]
    dc = [c[0]-2825/27648, 0, c[2]-18575/48384,
          c[3]-13525/55296, -277/14336, c[5]-0.25]

    while t < t_span[1]:
        h = min(h0, t_span[1]-t)
        k = [h*f(t, y)]
        for i in range(1, 6):
            k.append(h*f(t + a[i]*h, y + sum(b[i][j]*k[j] for j in range(i))))
        y4 = y + sum(cj*kj for cj, kj in zip(c, k))
        y5 = y + sum((cj+dcj)*kj for cj, dcj, kj in zip(c, dc, k))
        err = abs(y5 - y4)
        if err == 0.0:                       # avoid division by zero
            h = min(h*5.0, h_max)
            y, t = y5, t + h
            ts.append(t); ys.append(y)
        else:
            delta = max(1e-4, abs(y4))
            if err <= tol*delta:
                y, t = y5, t + h
                ts.append(t); ys.append(y)
            h = h0 #max(h_min, min(h_max, h*min(5.0, max(0.1, 0.84*(tol*delta/err)**0.25))))
        #h0 = h
    return np.array(ts), np.array(ys)

# ---------- GUI ----------
class OdeGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ODE GUI")
        frm = ttk.LabelFrame(self, text="Parameters")
        frm.pack(fill="x", padx=5, pady=5)
        labs = ["A(t)=", "B(t)=", "C(t)=", "v0=", "t0=", "t1="]
        self.vars = [tk.StringVar(value=v) for v in ["1", "-1", "-0.1", "0.0", "0", "10"]]
        for i, (lab, var) in enumerate(zip(labs, self.vars)):
            ttk.Label(frm, text=lab).grid(row=0, column=2*i)
            ttk.Entry(frm, textvariable=var, width=10).grid(row=0, column=2*i+1)
        ttk.Button(frm, text="Solve", command=self.solve).grid(row=0, column=12, padx=10)

        self.fig = Figure(figsize=(8,4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.solve()

    def solve(self):
        try:
            A = eval("lambda t: " + self.vars[0].get(), {"math": math})
            B = eval("lambda t: " + self.vars[1].get(), {"math": math})
            C = eval("lambda t: " + self.vars[2].get(), {"math": math})
            v0, t0, t1 = map(float, (self.vars[3].get(), self.vars[4].get(), self.vars[5].get()))
            t, v = rk45(lambda t, y: A(t) + B(t)*y + C(t)*y*y, v0, (t0, t1))
            self.ax.clear(); self.ax.plot(t, v); self.ax.set_xlabel("t"); self.ax.set_ylabel("v(t)")
            self.ax.grid(True); self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    OdeGui().mainloop()

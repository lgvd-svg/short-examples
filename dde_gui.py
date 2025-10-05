import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

class DDESolver:
    def __init__(self):
        pass
    
    def solve_dde(self, A, B, C, D, initial_condition, x_start, x_end, step_size):
        """
        Solve the DDE y'(x) = A - B*y(x) - C*y(x-D)*y(x) using 4th-order Runge-Kutta
        
        Parameters:
            A, B, C, D: constants in the DDE
            initial_condition: function y(x) for x ≤ x_start (history function)
            x_start, x_end: interval of integration
            step_size: step size for numerical solution
            
        Returns:
            x_values: array of x values
            y_values: array of corresponding y values
        """
        # Create array of x values
        x_values = np.arange(x_start, x_end + step_size, step_size)
        num_points = len(x_values)
        
        # Initialize y array
        y_values = np.zeros(num_points)
        
        # Set initial condition (history)
        for i in range(num_points):
            x = x_values[i]
            if x <= x_start:
                y_values[i] = initial_condition(x)
        
        # Perform Runge-Kutta integration
        for i in range(1, num_points):
            x = x_values[i]
            y_current = y_values[i-1]
            
            # For delayed term, we need to find y(x - D)
            x_delayed = x - D
            if x_delayed <= x_start:
                y_delayed = initial_condition(x_delayed)
            else:
                # Find the index where x_delayed would be
                # Using linear interpolation between known points
                idx = np.searchsorted(x_values, x_delayed, side='right') - 1
                if idx < 0:
                    y_delayed = initial_condition(x_delayed)
                else:
                    # Linear interpolation
                    x0 = x_values[idx]
                    x1 = x_values[idx+1]
                    y0 = y_values[idx]
                    y1 = y_values[idx+1]
                    y_delayed = y0 + (y1 - y0) * (x_delayed - x0) / (x1 - x0)
            
            # RK4 method
            k1 = A - B * y_current - C * y_delayed * y_current
            y_temp = y_current + 0.5 * step_size * k1
            
            # For k2, we need y(x + h/2 - D)
            x_delayed_k2 = x + 0.5*step_size - D
            if x_delayed_k2 <= x_start:
                y_delayed_k2 = initial_condition(x_delayed_k2)
            else:
                idx = np.searchsorted(x_values, x_delayed_k2, side='right') - 1
                if idx < 0:
                    y_delayed_k2 = initial_condition(x_delayed_k2)
                else:
                    x0 = x_values[idx]
                    x1 = x_values[idx+1]
                    y0 = y_values[idx]
                    y1 = y_values[idx+1]
                    y_delayed_k2 = y0 + (y1 - y0) * (x_delayed_k2 - x0) / (x1 - x0)
            
            k2 = A - B * y_temp - C * y_delayed_k2 * y_temp
            y_temp = y_current + 0.5 * step_size * k2
            
            # k3 uses the same delayed term as k2
            k3 = A - B * y_temp - C * y_delayed_k2 * y_temp
            y_temp = y_current + step_size * k3
            
            # For k4, we need y(x + h - D)
            x_delayed_k4 = x + step_size - D
            if x_delayed_k4 <= x_start:
                y_delayed_k4 = initial_condition(x_delayed_k4)
            else:
                idx = np.searchsorted(x_values, x_delayed_k4, side='right') - 1
                if idx < 0:
                    y_delayed_k4 = initial_condition(x_delayed_k4)
                else:
                    x0 = x_values[idx]
                    x1 = x_values[idx+1]
                    y0 = y_values[idx]
                    y1 = y_values[idx+1]
                    y_delayed_k4 = y0 + (y1 - y0) * (x_delayed_k4 - x0) / (x1 - x0)
            
            k4 = A - B * y_temp - C * y_delayed_k4 * y_temp
            
            # Update y value
            y_values[i] = y_current + (step_size / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return x_values, y_values


class DDEApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DDE Solver - Runge-Kutta Method")
        self.root.geometry("1200x800")
        
        self.solver = DDESolver()
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Parameter inputs
        ttk.Label(input_frame, text="A:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.A_var = tk.DoubleVar(value=1.0)
        ttk.Entry(input_frame, textvariable=self.A_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(input_frame, text="B:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.B_var = tk.DoubleVar(value=0.5)
        ttk.Entry(input_frame, textvariable=self.B_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(input_frame, text="C:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.C_var = tk.DoubleVar(value=0.2)
        ttk.Entry(input_frame, textvariable=self.C_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(input_frame, text="D (delay):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.D_var = tk.DoubleVar(value=1.0)
        ttk.Entry(input_frame, textvariable=self.D_var, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        # Initial condition
        ttk.Label(input_frame, text="Initial y(0):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.y0_var = tk.DoubleVar(value=0.0)
        ttk.Entry(input_frame, textvariable=self.y0_var, width=10).grid(row=4, column=1, padx=5, pady=2)
        
        # Integration parameters
        ttk.Label(input_frame, text="x start:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.x_start_var = tk.DoubleVar(value=0.0)
        ttk.Entry(input_frame, textvariable=self.x_start_var, width=10).grid(row=5, column=1, padx=5, pady=2)
        
        ttk.Label(input_frame, text="x end:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        self.x_end_var = tk.DoubleVar(value=10.0)
        ttk.Entry(input_frame, textvariable=self.x_end_var, width=10).grid(row=6, column=1, padx=5, pady=2)
        
        ttk.Label(input_frame, text="Step size:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)
        self.step_size_var = tk.DoubleVar(value=0.01)
        ttk.Entry(input_frame, textvariable=self.step_size_var, width=10).grid(row=7, column=1, padx=5, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=8, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Solve", command=self.solve).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Data", command=self.export_data).pack(side=tk.LEFT, padx=5)
        
        # Plot frame
        plot_frame = ttk.LabelFrame(main_frame, text="Solution Plot", padding="10")
        plot_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
    def initial_condition(self, x):
        """History function for x ≤ x_start"""
        return self.y0_var.get()
    
    def solve(self):
        try:
            # Get parameters
            A = self.A_var.get()
            B = self.B_var.get()
            C = self.C_var.get()
            D = self.D_var.get()
            x_start = self.x_start_var.get()
            x_end = self.x_end_var.get()
            step_size = self.step_size_var.get()
            
            # Validate inputs
            if x_end <= x_start:
                messagebox.showerror("Error", "x end must be greater than x start")
                return
            if step_size <= 0:
                messagebox.showerror("Error", "Step size must be positive")
                return
            
            # Solve the DDE
            x, y = self.solver.solve_dde(A, B, C, D, self.initial_condition, 
                                        x_start, x_end, step_size)
            
            # Plot the solution
            self.ax.clear()
            self.ax.plot(x, y, 'b-', linewidth=2, label=f'y(x)')
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y(x)')
            self.ax.set_title(f'Solution of y\' = {A} - {B}*y(x) - {C}*y(x-{D})*y(x)')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
            self.canvas.draw()
            
            # Store results for export
            self.current_x = x
            self.current_y = y
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def clear_plot(self):
        self.ax.clear()
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y(x)')
        self.ax.set_title('DDE Solution Plot')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
    
    def export_data(self):
        try:
            if hasattr(self, 'current_x') and hasattr(self, 'current_y'):
                # Create a simple text file with the data
                filename = "dde_solution.txt"
                with open(filename, 'w') as f:
                    f.write("x\ty(x)\n")
                    for i in range(len(self.current_x)):
                        f.write(f"{self.current_x[i]:.6f}\t{self.current_y[i]:.6f}\n")
                messagebox.showinfo("Success", f"Data exported to {filename}")
            else:
                messagebox.showwarning("Warning", "No data to export. Please solve the equation first.")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")

def main():
    root = tk.Tk()
    app = DDEApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

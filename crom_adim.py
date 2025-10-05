import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec

class DimensionlessChromatography:
    """
    Modelo de Columna Cromatográfica Adimensional
    Usando el modelo de platos teóricos y números adimensionales
    """
    
    def __init__(self, n_components=3):
        self.n_components = n_components
        
        # ===========================================
        # PARÁMETROS ADIMENSIONALES
        # ===========================================
        
        # Número de Péclet (Pe) - Razón convección/difusión
        self.Pe = 1000.0
        
        # Número de Stanton (St) - Razón transferencia masa/convección
        self.St = 2.0
        
        # Número de Damköhler (Da) - Razón adsorción/convección
        self.Da = 5.0
        
        # Número de Capacitancia (ε) - Porosidad
        self.epsilon = 0.4  # Porosidad del lecho
        
        # Números de Retención (k') - Factores de capacidad
        self.k_prime = np.array([2.0, 1.0, 0.5])  # k' = (1-ε)/ε * K
        
        # Número de Eficiencia (N) - Número de platos teóricos
        self.N_theoretical = 100
        
        # Número de Asimetría (As) - Factor de forma del pico
        self.As = 1.2
        
        # Parámetros de inyección
        self.t_inj_star = 0.1  # Tiempo de inyección adimensional
        self.C_inj_star = 1.0  # Concentración de inyección adimensional
        
    def dimensionless_adsorption_isotherm(self, C_star, component_idx):
        """
        Isoterma de adsorción adimensional (modelo lineal)
        q* = k' * C*
        """
        return self.k_prime[component_idx] * C_star /(1+ C_star)
    
    def dimensionless_kinetics(self, C_star, q_star, component_idx):
        """
        Cinética de adsorción adimensional
        dq*/dt* = St * (q*_eq - q*)
        """
        q_eq_star = self.dimensionless_adsorption_isotherm(C_star, component_idx)
        return self.St * (q_eq_star - q_star)
    
    def dimensionless_governing_equations(self, t_star, y, z_star):
        """
        Ecuaciones de transporte adimensionales para columna cromatográfica
        Modelo de equilibrio lineal + resistencia a la transferencia de masa
        """
        n_points = len(y) // (2 * self.n_components)
        dydt = np.zeros_like(y)
        
        for comp in range(self.n_components):
            for i in range(n_points):
                # Índices para concentración móvil y estacionaria
                idx_C = comp * n_points + i
                idx_q = (self.n_components + comp) * n_points + i
                
                C_star = y[idx_C]
                q_star = y[idx_q]
                
                if i == 0:
                    # Condición de entrada (inyección tipo pulso)
                    if t_star <= self.t_inj_star:
                        C_in = self.C_inj_star
                    else:
                        C_in = 0.0
                    
                    # Término convectivo (upwind difference)
                    dCdz = (C_star - C_in) / (1/n_points)
                elif i == n_points - 1:
                    # Condición de salida
                    dCdz = (0.0 - C_star) / (1/n_points)
                else:
                    # Puntos internos
                    dCdz = (y[idx_C + 1] - y[idx_C - 1]) / (2/n_points)
                
                # Término difusivo (segunda derivada)
                if i == 0 or i == n_points - 1:
                    d2Cdz2 = 0.0
                else:
                    d2Cdz2 = (y[idx_C + 1] - 2*C_star + y[idx_C - 1]) / (1/n_points)**2
                
                # Ecuación para fase móvil
                dCdt = -dCdz + (1/self.Pe) * d2Cdz2 - self.Da * self.dimensionless_kinetics(C_star, q_star, comp)
                dydt[idx_C] = dCdt
                
                # Ecuación para fase estacionaria
                dqdt = self.dimensionless_kinetics(C_star, q_star, comp)
                dydt[idx_q] = dqdt
        
        return dydt
    
    def solve_dimensionless_chromatography(self, n_points=100, t_final=10.0):
        """
        Resuelve la cromatografía adimensional
        """
        self.n_points = n_points
        z_star = np.linspace(0, 1, n_points)  # Posición adimensional z* = z/L
        
        # Condiciones iniciales (columna limpia)
        y0 = np.zeros(2 * self.n_components * n_points)
        
        # Tiempo adimensional
        t_eval = np.linspace(0, t_final, 1000)
        
        # Resolver ecuaciones
        solution = solve_ivp(
            fun=lambda t, y: self.dimensionless_governing_equations(t, y, z_star),
            t_span=(0, t_final),
            y0=y0,
            t_eval=t_eval,
            method='BDF',
            rtol=1e-6,
            atol=1e-8
        )
        
        return solution, z_star
    
    def calculate_dimensionless_retention_time(self):
        """
        Calcula tiempos de retención adimensionales
        t*_R = 1 + k'
        """
        t_retention = 1.0 + self.k_prime
        return t_retention
    
    def calculate_resolution(self, C_history, t_star):
        """
        Calcula la resolución adimensional entre picos
        Rs = 2*(t_R2 - t_R1) / (w1 + w2)
        """
        resolution_matrix = np.zeros((self.n_components, self.n_components))
        
        for i in range(self.n_components):
            for j in range(i+1, self.n_components):
                # Encontrar tiempos de retención
                t_ri = self.find_retention_time(C_history[i, :], t_star)
                t_rj = self.find_retention_time(C_history[j, :], t_star)
                
                # Encontrar anchos de pico
                w_i = self.find_peak_width(C_history[i, :], t_star, t_ri)
                w_j = self.find_peak_width(C_history[j, :], t_star, t_rj)
                
                # Calcular resolución
                if w_i > 0 and w_j > 0:
                    resolution = 2 * abs(t_rj - t_ri) / (w_i + w_j)
                else:
                    resolution = 0.0
                
                resolution_matrix[i, j] = resolution
                resolution_matrix[j, i] = resolution
        
        return resolution_matrix
    
    def find_retention_time(self, C_profile, t_star):
        """Encuentra el tiempo de retención de un pico"""
        max_idx = np.argmax(C_profile)
        return t_star[max_idx] if max_idx < len(t_star) else 0.0
    
    def find_peak_width(self, C_profile, t_star, t_retention):
        """Encuentra el ancho del pico a la mitad de la altura"""
        if t_retention == 0:
            return 0.0
        
        max_conc = np.max(C_profile)
        half_max = max_conc / 2.0
        
        # Encontrar puntos donde la concentración cruza la mitad del máximo
        above_half = C_profile >= half_max
        if not np.any(above_half):
            return 0.0
        
        # Encontrar primeros y últimos puntos por encima de la mitad
        indices = np.where(above_half)[0]
        if len(indices) == 0:
            return 0.0
        
        start_idx = indices[0]
        end_idx = indices[-1]
        
        if start_idx >= len(t_star) or end_idx >= len(t_star):
            return 0.0
        
        width = t_star[end_idx] - t_star[start_idx]
        return width
    
    def calculate_efficiency(self, C_history, t_star):
        """
        Calcula la eficiencia adimensional (número de platos teóricos)
        N = 16 * (t_R/w)^2
        """
        efficiencies = []
        
        for comp in range(self.n_components):
            t_retention = self.find_retention_time(C_history[comp, :], t_star)
            width = self.find_peak_width(C_history[comp, :], t_star, t_retention)
            
            if width > 0 and t_retention > 0:
                efficiency = 16 * (t_retention / width) ** 2
            else:
                efficiency = 0.0
            
            efficiencies.append(efficiency)
        
        return np.array(efficiencies)

def plot_dimensionless_chromatography_results(t_star, C_outlet, z_star, C_final, resolution, efficiencies, params):
    """Visualiza resultados de la cromatografía adimensional"""
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # 1. Cromatograma adimensional (concentración vs tiempo)
    ax1 = fig.add_subplot(gs[0, :2])
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for comp in range(C_outlet.shape[0]):
        ax1.plot(t_star, C_outlet[comp, :], color=colors[comp % len(colors)],
                linewidth=3, label=f'Componente {comp+1} (k\' = {params["k_prime"][comp]:.2f})')
    
    ax1.set_xlabel('Tiempo Adimensional (t*)')
    ax1.set_ylabel('Concentración Adimensional (C*)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Cromatograma Adimensional - Señal de Salida')
    
    # 2. Perfiles de concentración en la columna (en tiempo final)
    ax2 = fig.add_subplot(gs[1, 0])
    for comp in range(C_final.shape[0]):
        ax2.plot(z_star, C_final[comp, :], color=colors[comp % len(colors)],
                linewidth=2, label=f'Comp {comp+1}')
    
    ax2.set_xlabel('Posición Adimensional (z*)')
    ax2.set_ylabel('Concentración Fase Móvil (C*)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Perfil de Concentración en Columna (t final)')
    
    # 3. Matriz de resolución
    ax3 = fig.add_subplot(gs[1, 1])
    n_comp = C_outlet.shape[0]
    resolution_plot = np.zeros((n_comp, n_comp))
    
    for i in range(n_comp):
        for j in range(n_comp):
            if i != j:
                resolution_plot[i, j] = resolution[i, j]
    
    im = ax3.imshow(resolution_plot, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax3, label='Resolución (Rs)')
    
    # Etiquetas
    ax3.set_xticks(range(n_comp))
    ax3.set_yticks(range(n_comp))
    ax3.set_xticklabels([f'Comp {i+1}' for i in range(n_comp)])
    ax3.set_yticklabels([f'Comp {i+1}' for i in range(n_comp)])
    ax3.set_title('Matriz de Resolución entre Componentes')
    
    # 4. Eficiencia de la columna
    ax4 = fig.add_subplot(gs[1, 2])
    components = [f'Comp {i+1}' for i in range(len(efficiencies))]
    bars = ax4.bar(components, efficiencies, color=colors[:len(efficiencies)])
    ax4.set_ylabel('Número de Platos Teóricos (N)')
    ax4.set_title('Eficiencia de la Columna por Componente')
    ax4.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar, eff in zip(bars, efficiencies):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{eff:.0f}', ha='center', va='bottom')
    
    # 5. Tiempos de retención teóricos vs experimentales
    ax5 = fig.add_subplot(gs[2, 0])
    t_retention_theo = params['t_retention_theo']
    t_retention_exp = [params['find_retention_time'](C_outlet[i, :], t_star) for i in range(n_comp)]
    
    ax5.plot(range(1, n_comp+1), t_retention_theo, 'bo-', linewidth=2, 
             markersize=8, label='Teórico')
    ax5.plot(range(1, n_comp+1), t_retention_exp, 'ro-', linewidth=2,
             markersize=8, label='Experimental')
    ax5.set_xlabel('Componente')
    ax5.set_ylabel('Tiempo Retención Adimensional (t*_R)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Tiempos de Retención Teóricos vs Experimentales')
    
    # 6. Efecto del número de Péclet
    ax6 = fig.add_subplot(gs[2, 1])
    Pe_values = [100, 500, 1000, 2000, 5000]
    peak_widths = []
    
    for Pe in Pe_values:
        temp_chromo = DimensionlessChromatography()
        temp_chromo.Pe = Pe
        sol, _ = temp_chromo.solve_dimensionless_chromatography()
        C_temp = extract_outlet_concentration(sol.y, temp_chromo.n_points, temp_chromo.n_components)
        width = temp_chromo.find_peak_width(C_temp[0, :], sol.t, 
                                          temp_chromo.find_retention_time(C_temp[0, :], sol.t))
        peak_widths.append(width)
    
    ax6.semilogx(Pe_values, peak_widths, 'g-', linewidth=3, marker='o')
    ax6.set_xlabel('Número de Péclet (Pe)')
    ax6.set_ylabel('Ancho de Pico Adimensional (w*)')
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Efecto del Número de Péclet en el Ancho de Pico')
    
    # 7. Selectividad y resolución
    ax7 = fig.add_subplot(gs[2, 2])
    k_prime_range = np.linspace(0.5, 3.0, 20)
    resolutions = []
    
    for k2 in k_prime_range:
        temp_chromo = DimensionlessChromatography()
        temp_chromo.k_prime = np.array([1.0, k2, 0.5])
        sol, _ = temp_chromo.solve_dimensionless_chromatography()
        C_temp = extract_outlet_concentration(sol.y, temp_chromo.n_points, temp_chromo.n_components)
        res_matrix = temp_chromo.calculate_resolution(C_temp, sol.t)
        resolutions.append(res_matrix[0, 1])  # Resolución entre comp 1 y 2
    
    ax7.plot(k_prime_range, resolutions, 'purple', linewidth=3)
    ax7.set_xlabel('Factor de Capacidad Comp 2 (k\'₂)')
    ax7.set_ylabel('Resolución (Rs₁₂)')
    ax7.grid(True, alpha=0.3)
    ax7.set_title('Resolución vs Selectividad')
    
    plt.tight_layout()
    plt.show()

def extract_outlet_concentration(y, n_points, n_components):
    """Extrae la concentración de salida de la solución"""
    n_time = y.shape[1]
    C_outlet = np.zeros((n_components, n_time))
    
    for comp in range(n_components):
        # La concentración de salida está en el último punto espacial
        outlet_idx = (comp + 1) * n_points - 1
        C_outlet[comp, :] = y[outlet_idx, :]
    
    return C_outlet

def extract_final_profile(y, n_points, n_components):
    """Extrae el perfil final de concentración en la columna"""
    n_vars = y.shape[0]
    C_final = np.zeros((n_components, n_points))
    
    for comp in range(n_components):
        start_idx = comp * n_points
        end_idx = (comp + 1) * n_points
        C_final[comp, :] = y[start_idx:end_idx, -1]
    
    return C_final

def dimensionless_optimization():
    """Optimización de parámetros adimensionales para máxima resolución"""
    print("\n" + "="*60)
    print("OPTIMIZACIÓN DE PARÁMETROS ADIMENSIONALES")
    print("="*60)
    
    def objective_function(x):
        # x = [Pe, St, Da]
        Pe, St, Da = x
        
        chromo = DimensionlessChromatography()
        chromo.Pe = Pe
        chromo.St = St
        chromo.Da = Da
        
        sol, _ = chromo.solve_dimensionless_chromatography(t_final=8.0)
        C_outlet = extract_outlet_concentration(sol.y, chromo.n_points, chromo.n_components)
        resolution_matrix = chromo.calculate_resolution(C_outlet, sol.t)
        
        # Maximizar la resolución mínima entre componentes adyacentes
        min_resolution = np.min(resolution_matrix[resolution_matrix > 0]) if np.any(resolution_matrix > 0) else 0
        
        # Penalizar valores extremos de parámetros
        penalty = 0
        if Pe < 100 or Pe > 10000:
            penalty += 1000
        if St < 0.1 or St > 10:
            penalty += 1000
        if Da < 0.1 or Da > 20:
            penalty += 1000
        
        return -min_resolution + penalty  # Negativo para maximizar
    
    # Optimización
    initial_guess = [1000.0, 2.0, 5.0]
    bounds = [(100, 10000), (0.1, 10.0), (0.1, 20.0)]
    
    result = minimize(objective_function, initial_guess, method='L-BFGS-B', bounds=bounds)
    
    print(f"Parámetros óptimos encontrados:")
    print(f"  Péclet (Pe): {result.x[0]:.2f}")
    print(f"  Stanton (St): {result.x[1]:.2f}")
    print(f"  Damköhler (Da): {result.x[2]:.2f}")
    print(f"  Resolución mínima: {-result.fun:.3f}")
    
    return result.x

def main():
    # ===========================================
    # SIMULACIÓN DE CROMATOGRAFÍA ADIMENSIONAL
    # ===========================================
    print("Inicializando Columna Cromatográfica Adimensional...")
    chromo = DimensionlessChromatography(n_components=3)
    
    print("\n" + "="*60)
    print("PARÁMETROS ADIMENSIONALES")
    print("="*60)
    print(f"Número de Péclet (Pe): {chromo.Pe}")
    print(f"Número de Stanton (St): {chromo.St}")
    print(f"Número de Damköhler (Da): {chromo.Da}")
    print(f"Porosidad (ε): {chromo.epsilon}")
    print(f"Factores de capacidad (k'): {chromo.k_prime}")
    print(f"Tiempo inyección (t*_inj): {chromo.t_inj_star}")
    
    # ===========================================
    # RESOLVER CROMATOGRAFÍA ADIMENSIONAL
    # ===========================================
    print("\nResolviendo ecuaciones de transporte adimensionales...")
    solution, z_star = chromo.solve_dimensionless_chromatography(t_final=8.0)
    
    # ===========================================
    # PROCESAR RESULTADOS
    # ===========================================
    t_star = solution.t
    C_outlet = extract_outlet_concentration(solution.y, chromo.n_points, chromo.n_components)
    C_final = extract_final_profile(solution.y, chromo.n_points, chromo.n_components)
    
    # Calcular métricas de desempeño
    resolution_matrix = chromo.calculate_resolution(C_outlet, t_star)
    efficiencies = chromo.calculate_efficiency(C_outlet, t_star)
    t_retention_theo = chromo.calculate_dimensionless_retention_time()
    
    # ===========================================
    # RESULTADOS NUMÉRICOS
    # ===========================================
    print("\n" + "="*60)
    print("RESULTADOS DE LA CROMATOGRAFÍA ADIMENSIONAL")
    print("="*60)
    
    for comp in range(chromo.n_components):
        t_ret_exp = chromo.find_retention_time(C_outlet[comp, :], t_star)
        width = chromo.find_peak_width(C_outlet[comp, :], t_star, t_ret_exp)
        print(f"Componente {comp+1} (k' = {chromo.k_prime[comp]:.2f}):")
        print(f"  Tiempo retención teórico: {t_retention_theo[comp]:.3f}")
        print(f"  Tiempo retención experimental: {t_ret_exp:.3f}")
        print(f"  Ancho de pico: {width:.3f}")
        print(f"  Eficiencia (N): {efficiencies[comp]:.0f}")
        print()
    
    print("Resolución entre componentes:")
    for i in range(chromo.n_components):
        for j in range(i+1, chromo.n_components):
            print(f"  Rs({i+1},{j+1}) = {resolution_matrix[i,j]:.3f}")
    
    # ===========================================
    # VISUALIZACIÓN
    # ===========================================
    print("\nGenerando gráficas adimensionales...")
    params = {
        'k_prime': chromo.k_prime,
        't_retention_theo': t_retention_theo,
        'find_retention_time': chromo.find_retention_time
    }
    
    plot_dimensionless_chromatography_results(t_star, C_outlet, z_star, C_final, 
                                            resolution_matrix, efficiencies, params)
    
    # ===========================================
    # OPTIMIZACIÓN
    # ===========================================
    optimal_params = dimensionless_optimization()
    
    # Simular con parámetros óptimos
    print("\nSimulando con parámetros óptimos...")
    chromo_opt = DimensionlessChromatography()
    chromo_opt.Pe, chromo_opt.St, chromo_opt.Da = optimal_params
    sol_opt, _ = chromo_opt.solve_dimensionless_chromatography(t_final=8.0)
    C_outlet_opt = extract_outlet_concentration(sol_opt.y, chromo_opt.n_points, chromo_opt.n_components)
    res_opt = chromo_opt.calculate_resolution(C_outlet_opt, sol_opt.t)
    
    print(f"Resolución mínima con parámetros óptimos: {np.min(res_opt[res_opt > 0]):.3f}")

if __name__ == "__main__":
    main()

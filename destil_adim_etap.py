import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec

class DimensionlessDistillation:
    """
    Modelo de Destilación Multicomponente Adimensional
    Usando números adimensionales y el método de McCabe-Thiele generalizado
    """
    
    def __init__(self, n_components=3):
        self.n_components = n_components
        
        # ===========================================
        # PARÁMETROS ADIMENSIONALES
        # ===========================================
        
        # Número de Stanton de transferencia de masa
        self.St_m = 2.0
        
        # Número de Reflujo (R) - Razón reflujo/alimentación
        self.R = 2.0
        
        # Número de Stripping (S) - Razón vaporización/alimentación  
        self.S = 2.0
        
        # Número de Eficiencia de Plato (η)
        self.eta_plate = 0.7
        
        # Número de Presión Relativa (Π)
        self.Pi = 1.0  # Presión adimensional P/P_ref
        
        # Número de Volatilidad Relativa (α)
        self.alpha = np.array([4.0, 2.0, 1.0])  # Volatilidades relativas
        
        # Fracción alimentación adimensional (q)
        self.q = 0.5  # 0 < q < 1 (líquido saturado)
        
        # Composición alimentación adimensional
        self.zF = np.array([0.4, 0.3, 0.3])  # Suma = 1.0
        
        # Número de platos
        self.N_stages = 10
        
    def dimensionless_vapor_liquid_equilibrium(self, x_star, stage_temp_star):
        """
        Relación de equilibrio vapor-líquido adimensional
        y*_i = (α_i * x*_i) / (Σ α_j * x*_j)
        """
        numerator = self.alpha * x_star
        denominator = np.sum(numerator)
        
        if denominator == 0:
            return x_star.copy()
        
        y_star_eq = numerator / denominator
        return y_star_eq
    
    def dimensionless_operating_line_rectifying(self, x_star, R, xD_star):
        """
        Línea de operación adimensional para sección de rectificación
        y* = [R/(R+1)] * x* + [1/(R+1)] * xD*
        """
        term1 = (R / (R + 1)) * x_star
        term2 = (1 / (R + 1)) * xD_star
        return term1 + term2
    
    def dimensionless_operating_line_stripping(self, x_star, S, xB_star):
        """
        Línea de operación adimensional para sección de agotamiento
        y* = [(S+1)/S] * x* - [1/S] * xB*
        """
        term1 = ((S + 1) / S) * x_star
        term2 = (1 / S) * xB_star
        return term1 - term2
    
    def dimensionless_murphree_efficiency(self, y_star_actual, y_star_prev, y_star_eq):
        """
        Eficiencia de Murphree adimensional
        EM = (y*_actual - y*_prev) / (y*_eq - y*_prev)
        """
        denominator = y_star_eq - y_star_prev
        mask = np.abs(denominator) > 1e-10
        efficiency = np.zeros_like(y_star_actual)
        efficiency[mask] = (y_star_actual[mask] - y_star_prev[mask]) / denominator[mask]
        return efficiency
    
    def solve_dimensionless_column(self):
        """
        Resuelve la columna de destilación adimensional usando método de McCabe-Thiele
        """
        # Inicializar matrices de composición
        x_star = np.zeros((self.N_stages, self.n_components))
        y_star = np.zeros((self.N_stages, self.n_components))
        
        # Condiciones de contorno (estimaciones iniciales)
        xD_star_guess = np.array([0.9, 0.08, 0.02])  # Destilado rico en componente 1
        xB_star_guess = np.array([0.05, 0.35, 0.6])  # Fondos rico en componente 3
        
        # Encontrar condiciones de operación consistentes
        def column_equations(vars):
            xD, xB = vars[:self.n_components], vars[self.n_components:2*self.n_components]
            
            # Balances globales de materia
            total_balance = np.sum(xD) - 1.0  # Normalización
            total_balance += np.sum(xB) - 1.0
            
            # Balances por componente
            comp_balance = self.zF - (0.5 * xD + 0.5 * xB)  # D = B = 0.5*F adimensional
            
            return np.concatenate([ [total_balance], comp_balance, [np.sum(xD)-1, np.sum(xB)-1] ])
        
        # Resolver balances
        initial_guess = np.concatenate([xD_star_guess, xB_star_guess])
        solution = fsolve(column_equations, initial_guess, xtol=1e-8)
        
        xD_star = solution[:self.n_components]
        xB_star = solution[self.n_components:2*self.n_components]
        
        # Normalizar
        xD_star = xD_star / np.sum(xD_star)
        xB_star = xB_star / np.sum(xB_star)
        
        print(f"Composición destilado adimensional: {xD_star}")
        print(f"Composición fondos adimensional: {xB_star}")
        
        # Ubicación del plato de alimentación (adimensional)
        feed_stage = int(self.N_stages * self.q)
        
        # Inicializar composiciones
        x_star[0, :] = xB_star  # Fondos
        y_star[0, :] = self.dimensionless_vapor_liquid_equilibrium(xB_star, 0.8)
        
        # Calcular composiciones etapa por etapa
        for stage in range(1, self.N_stages):
            if stage < feed_stage:
                # Sección de agotamiento
                y_star_prev = y_star[stage-1, :]
                x_star_eq_guess = x_star[stage-1, :]  # Guess inicial
                
                def stripping_equations(x_curr):
                    y_eq = self.dimensionless_vapor_liquid_equilibrium(x_curr, 0.5)
                    y_op = self.dimensionless_operating_line_stripping(x_curr, self.S, xB_star)
                    return y_eq - y_op
                
                x_sol = fsolve(stripping_equations, x_star_eq_guess, xtol=1e-8)
                x_star[stage, :] = x_sol / np.sum(x_sol)
                y_star[stage, :] = self.dimensionless_operating_line_stripping(x_star[stage, :], self.S, xB_star)
                
            else:
                # Sección de rectificación
                y_star_prev = y_star[stage-1, :]
                x_star_eq_guess = x_star[stage-1, :]
                
                def rectifying_equations(x_curr):
                    y_eq = self.dimensionless_vapor_liquid_equilibrium(x_curr, 0.5)
                    y_op = self.dimensionless_operating_line_rectifying(x_curr, self.R, xD_star)
                    return y_eq - y_op
                
                x_sol = fsolve(rectifying_equations, x_star_eq_guess, xtol=1e-8)
                x_star[stage, :] = x_sol / np.sum(x_sol)
                y_star[stage, :] = self.dimensionless_operating_line_rectifying(x_star[stage, :], self.R, xD_star)
        
        # Aplicar eficiencia de plato
        for stage in range(1, self.N_stages):
            y_star_eq = self.dimensionless_vapor_liquid_equilibrium(x_star[stage, :], 0.5)
            y_star_actual = y_star[stage-1, :] + self.eta_plate * (y_star_eq - y_star[stage-1, :])
            y_star[stage, :] = y_star_actual / np.sum(y_star_actual)
        
        # Última etapa - condensador
        x_star[-1, :] = xD_star
        y_star[-1, :] = xD_star  # Reflujo total
        
        return x_star, y_star, xD_star, xB_star, feed_stage
    
    def calculate_dimensionless_performance(self, x_star, y_star, xD_star, xB_star):
        """
        Calcula indicadores de desempeño adimensionales
        """
        # Pureza del destilado (componente más volátil)
        distillate_purity = xD_star[0]
        
        # Recuperación del componente clave
        recovery_key = (0.5 * xD_star[0]) / self.zF[0]  # D*xD / F*zF
        
        # Eficiencia de separación adimensional
        separation_index = (xD_star[0] - xB_star[0]) / (1 - xB_star[0])
        
        # Número de platos teóricos adimensionales
        theoretical_stages = self.N_stages * self.eta_plate
        
        return {
            'distillate_purity': distillate_purity,
            'recovery': recovery_key,
            'separation_index': separation_index,
            'theoretical_stages': theoretical_stages
        }

def plot_dimensionless_distillation_results(stages, x_star, y_star, feed_stage, performance):
    """Visualiza resultados de la destilación adimensional"""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    stage_numbers = np.arange(len(stages))
    
    # 1. Perfiles de composición líquida adimensional
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['red', 'blue', 'green']
    labels = ['Componente 1 (Liviano)', 'Componente 2', 'Componente 3 (Pesado)']
    
    for i in range(x_star.shape[1]):
        ax1.plot(stage_numbers, x_star[:, i], color=colors[i], 
                linewidth=3, marker='o', markersize=4, label=labels[i])
    
    ax1.axvline(x=feed_stage, color='black', linestyle='--', linewidth=2, label='Alimentación')
    ax1.set_xlabel('Número de Plato Adimensional')
    ax1.set_ylabel('Composición Líquida (x*)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Perfil de Composición Líquida Adimensional')
    
    # 2. Perfiles de composición vapor adimensional
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(y_star.shape[1]):
        ax2.plot(stage_numbers, y_star[:, i], color=colors[i], 
                linewidth=3, marker='s', markersize=4, label=labels[i])
    
    ax2.axvline(x=feed_stage, color='black', linestyle='--', linewidth=2, label='Alimentación')
    ax2.set_xlabel('Número de Plato Adimensional')
    ax2.set_ylabel('Composición Vapor (y*)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Perfil de Composición Vapor Adimensional')
    
    # 3. Diagrama de McCabe-Thiele adimensional (para componente clave)
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Línea de equilibrio
    x_eq = np.linspace(0, 1, 100)
    y_eq = distillation.alpha[0] * x_eq / (1 + (distillation.alpha[0] - 1) * x_eq)
    ax3.plot(x_eq, y_eq, 'b-', linewidth=2, label='Equilibrio')
    
    # Línea operación rectificación
    x_rect = np.linspace(0, 1, 100)
    y_rect = distillation.dimensionless_operating_line_rectifying(x_rect, distillation.R, performance['xD_star'][0])
    ax3.plot(x_rect, y_rect, 'r-', linewidth=2, label='Operación Rectificación')
    
    # Línea operación agotamiento
    x_strip = np.linspace(0, 1, 100)
    y_strip = distillation.dimensionless_operating_line_stripping(x_strip, distillation.S, performance['xB_star'][0])
    ax3.plot(x_strip, y_strip, 'g-', linewidth=2, label='Operación Agotamiento')
    
    # Escalones de McCabe-Thiele
    for i in range(len(stages)-1):
        ax3.plot([x_star[i, 0], x_star[i, 0]], [y_star[i, 0], y_star[i+1, 0]], 'k-', alpha=0.5)
        ax3.plot([x_star[i, 0], x_star[i+1, 0]], [y_star[i+1, 0], y_star[i+1, 0]], 'k-', alpha=0.5)
    
    ax3.set_xlabel('x* (Componente Clave)')
    ax3.set_ylabel('y* (Componente Clave)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Diagrama de McCabe-Thiele Adimensional')
    
    # 4. Indicadores de desempeño
    ax4 = fig.add_subplot(gs[1, 0])
    metrics = ['Pureza Destilado', 'Recuperación', 'Índice Separación']
    values = [performance['distillate_purity'], performance['recovery'], performance['separation_index']]
    
    bars = ax4.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax4.set_ylabel('Valor Adimensional')
    ax4.set_title('Indicadores de Desempeño Adimensionales')
    
    # Agregar valores en las barras
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 5. Efecto de la eficiencia
    ax5 = fig.add_subplot(gs[1, 1])
    eta_values = np.linspace(0.3, 1.0, 50)
    purity_values = []
    
    for eta in eta_values:
        temp_distillation = DimensionlessDistillation()
        temp_distillation.eta_plate = eta
        x_temp, y_temp, xD_temp, xB_temp, _ = temp_distillation.solve_dimensionless_column()
        purity = xD_temp[0]
        purity_values.append(purity)
    
    ax5.plot(eta_values, purity_values, 'purple', linewidth=3)
    ax5.set_xlabel('Eficiencia de Plato (η)')
    ax5.set_ylabel('Pureza Destilado (x*_D)')
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Efecto de la Eficiencia en la Pureza')
    
    # 6. Balance de materia adimensional
    ax6 = fig.add_subplot(gs[1, 2])
    components = ['Comp 1', 'Comp 2', 'Comp 3']
    feed = distillation.zF
    distillate = performance['xD_star']
    bottoms = performance['xB_star']
    
    x_pos = np.arange(len(components))
    width = 0.25
    
    ax6.bar(x_pos - width, feed, width, label='Alimentación (z*)', alpha=0.7)
    ax6.bar(x_pos, distillate, width, label='Destilado (xD*)', alpha=0.7)
    ax6.bar(x_pos + width, bottoms, width, label='Fondos (xB*)', alpha=0.7)
    
    ax6.set_xlabel('Componentes')
    ax6.set_ylabel('Composición Adimensional')
    ax6.set_title('Balance de Materia por Componente')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def dimensionless_sensitivity_analysis():
    """Análisis de sensibilidad de parámetros adimensionales"""
    print("\n" + "="*60)
    print("ANÁLISIS DE SENSIBILIDAD ADIMENSIONAL")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sensibilidad al número de reflujo
    R_values = np.linspace(0.5, 5.0, 20)
    purity_R = []
    recovery_R = []
    
    for R in R_values:
        temp_distill = DimensionlessDistillation()
        temp_distill.R = R
        x_temp, y_temp, xD_temp, xB_temp, _ = temp_distill.solve_dimensionless_column()
        perf = temp_distill.calculate_dimensionless_performance(x_temp, y_temp, xD_temp, xB_temp)
        purity_R.append(perf['distillate_purity'])
        recovery_R.append(perf['recovery'])
    
    axes[0, 0].plot(R_values, purity_R, 'b-', linewidth=3, label='Pureza Destilado')
    axes[0, 0].set_xlabel('Número de Reflujo (R)')
    axes[0, 0].set_ylabel('Pureza (x*_D)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Sensibilidad al Número de Reflujo')
    axes[0, 0].legend()
    
    axes_twin = axes[0, 0].twinx()
    axes_twin.plot(R_values, recovery_R, 'r-', linewidth=3, label='Recuperación')
    axes_twin.set_ylabel('Recuperación')
    axes_twin.legend(loc='lower right')
    
    # Sensibilidad a la volatilidad relativa
    alpha_values = np.linspace(1.5, 6.0, 20)
    purity_alpha = []
    
    for alpha in alpha_values:
        temp_distill = DimensionlessDistillation()
        temp_distill.alpha = np.array([alpha, 2.0, 1.0])
        x_temp, y_temp, xD_temp, xB_temp, _ = temp_distill.solve_dimensionless_column()
        perf = temp_distill.calculate_dimensionless_performance(x_temp, y_temp, xD_temp, xB_temp)
        purity_alpha.append(perf['distillate_purity'])
    
    axes[0, 1].plot(alpha_values, purity_alpha, 'g-', linewidth=3)
    axes[0, 1].set_xlabel('Volatilidad Relativa (α₁)')
    axes[0, 1].set_ylabel('Pureza Destilado (x*_D)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Sensibilidad a la Volatilidad Relativa')
    
    # Sensibilidad al número de Stanton
    St_values = np.linspace(0.5, 4.0, 20)
    purity_St = []
    
    for St in St_values:
        temp_distill = DimensionlessDistillation()
        temp_distill.St_m = St
        x_temp, y_temp, xD_temp, xB_temp, _ = temp_distill.solve_dimensionless_column()
        perf = temp_distill.calculate_dimensionless_performance(x_temp, y_temp, xD_temp, xB_temp)
        purity_St.append(perf['distillate_purity'])
    
    axes[1, 0].plot(St_values, purity_St, 'orange', linewidth=3)
    axes[1, 0].set_xlabel('Número de Stanton (St)')
    axes[1, 0].set_ylabel('Pureza Destilado (x*_D)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Sensibilidad al Número de Stanton')
    
    # Sensibilidad a la condición de alimentación
    q_values = np.linspace(0.1, 0.9, 20)
    purity_q = []
    
    for q in q_values:
        temp_distill = DimensionlessDistillation()
        temp_distill.q = q
        x_temp, y_temp, xD_temp, xB_temp, _ = temp_distill.solve_dimensionless_column()
        perf = temp_distill.calculate_dimensionless_performance(x_temp, y_temp, xD_temp, xB_temp)
        purity_q.append(perf['distillate_purity'])
    
    axes[1, 1].plot(q_values, purity_q, 'purple', linewidth=3)
    axes[1, 1].set_xlabel('Condición Alimentación (q)')
    axes[1, 1].set_ylabel('Pureza Destilado (x*_D)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Sensibilidad a la Condición de Alimentación')
    
    plt.tight_layout()
    plt.show()

def main():
    # ===========================================
    # SIMULACIÓN DE DESTILACIÓN ADIMENSIONAL
    # ===========================================
    print("Inicializando Columna de Destilación Adimensional...")
    global distillation
    distillation = DimensionlessDistillation(n_components=3)
    
    print("\n" + "="*60)
    print("PARÁMETROS ADIMENSIONALES")
    print("="*60)
    print(f"Número de Reflujo (R): {distillation.R}")
    print(f"Número de Stripping (S): {distillation.S}")
    print(f"Número de Stanton (St): {distillation.St_m}")
    print(f"Eficiencia de Plato (η): {distillation.eta_plate}")
    print(f"Volatilidades Relativas (α): {distillation.alpha}")
    print(f"Condición Alimentación (q): {distillation.q}")
    print(f"Composición Alimentación (z*): {distillation.zF}")
    
    # ===========================================
    # RESOLVER COLUMNA ADIMENSIONAL
    # ===========================================
    print("\nResolviendo columna de destilación adimensional...")
    x_star, y_star, xD_star, xB_star, feed_stage = distillation.solve_dimensionless_column()
    
    # ===========================================
    # CÁLCULO DE INDICADORES DE DESEMPEÑO
    # ===========================================
    performance = distillation.calculate_dimensionless_performance(x_star, y_star, xD_star, xB_star)
    performance['xD_star'] = xD_star
    performance['xB_star'] = xB_star
    
    # ===========================================
    # RESULTADOS NUMÉRICOS
    # ===========================================
    print("\n" + "="*60)
    print("RESULTADOS DE LA DESTILACIÓN ADIMENSIONAL")
    print("="*60)
    print(f"Composición destilado: {xD_star}")
    print(f"Composición fondos: {xB_star}")
    print(f"Pureza destilado (componente 1): {performance['distillate_purity']:.4f}")
    print(f"Recuperación componente 1: {performance['recovery']:.4f}")
    print(f"Índice de separación: {performance['separation_index']:.4f}")
    print(f"Platos teóricos equivalentes: {performance['theoretical_stages']:.1f}")
    print(f"Plato de alimentación: {feed_stage}")
    
    # ===========================================
    # VISUALIZACIÓN
    # ===========================================
    print("\nGenerando gráficas adimensionales...")
    stages = np.arange(distillation.N_stages)
    plot_dimensionless_distillation_results(stages, x_star, y_star, feed_stage, performance)
    
    # ===========================================
    # ANÁLISIS DE SENSIBILIDAD
    # ===========================================
    dimensionless_sensitivity_analysis()

if __name__ == "__main__":
    main()
